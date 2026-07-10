"""HybridTieredMemoryStore：四级层级记忆后端实现。

- WORKING: in-process LRU cache.
- TASK: SQLite rows.
- PROJECT: Markdown raw archive + SQLite searchable index.
- GLOBAL: SQLite rows, with optional Redis mirroring when ``redis_url`` is set.

Retrieval uses embedding similarity over compact summaries/index text. Long
PROJECT text is archived as Markdown and can be expanded through ``raw_ref``.
"""

from __future__ import annotations

import json
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Optional

from ._types import (
    ArchiveExpandPolicy,
    MediaRoute,
    MemoryContext,
    MemoryItem,
    MemoryMedium,
    MemoryScope,
    MemoryUpdateAction,
    RetrievalMode,
    WakeupLevel,
    WakeupProfile,
    WakeupResult,
    wakeup_profile,
)
from ._utils import (
    _clip_text,
    _cosine,
    _format_markdown_memory,
    _indent_block,
    _memory_index_text,
    _recency_score,
    _rough_token_count,
    _safe_filename,
    _sparse_score,
    _stringify,
    _summarize_text,
    _tokenize,
)
from .base import MemoryStore
from .embedding import EmbeddingModel, HashingEmbeddingModel
from .judge import MemoryLLMJudge


class HybridTieredMemoryStore(MemoryStore):
    """Tiered memory backend.

    - WORKING: in-process LRU cache.
    - TASK: SQLite rows.
    - PROJECT: Markdown raw archive + SQLite searchable index.
    - GLOBAL: SQLite rows, with optional Redis mirroring when ``redis_url`` is set.

    Retrieval uses embedding similarity over compact summaries/index text. Long
    PROJECT text is archived as Markdown and can be expanded through ``raw_ref``.
    """

    def __init__(
        self,
        root_dir: str | Path = ".memory",
        *,
        sqlite_path: Optional[str | Path] = None,
        working_max_items: int = 256,
        embedding_model: Optional[EmbeddingModel] = None,
        long_text_threshold: int = 2000,
        summary_max_chars: int = 1200,
        redis_url: Optional[str] = None,
        redis_default_ttl: int = 3600,
        postgres_dsn: Optional[str] = None,
        graph_backend: Optional[Any] = None,
        enable_memory_update: bool = True,
        memory_llm_judge: Optional[MemoryLLMJudge] = None,
        memory_update_scopes: Optional[List[MemoryScope]] = None,
        memory_search_top_k: int = 5,
        memory_search_min_similarity: float = 0.3,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir = self.root_dir / "project_archive"
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir = self.root_dir / "audit"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.audit_jsonl = self.audit_dir / "memories.jsonl"
        self.sqlite_path = Path(sqlite_path) if sqlite_path else self.root_dir / "memory.sqlite3"
        self.working_max_items = working_max_items
        self.long_text_threshold = long_text_threshold
        self.summary_max_chars = summary_max_chars
        self.embedding_model = embedding_model or HashingEmbeddingModel()
        self.redis_default_ttl = redis_default_ttl
        self._working: "OrderedDict[str, MemoryItem]" = OrderedDict()
        self._redis = self._connect_redis(redis_url)
        self._postgres = self._connect_postgres(postgres_dsn)
        self._graph_backend = graph_backend
        # Memory update configuration (mem0-style, LLM-based)
        self.enable_memory_update = enable_memory_update
        self.memory_llm_judge = memory_llm_judge
        self.memory_update_scopes = memory_update_scopes or [
            MemoryScope.PROJECT,
            MemoryScope.GLOBAL,
        ]
        self.memory_search_top_k = memory_search_top_k
        self.memory_search_min_similarity = memory_search_min_similarity
        # Persistent SQLite connection (WAL mode for better concurrency)
        self._sqlite_conn = sqlite3.connect(
            str(self.sqlite_path), check_same_thread=False,
        )
        self._sqlite_conn.row_factory = sqlite3.Row
        self._sqlite_conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def write(self, item: MemoryItem, *, context: Optional[MemoryContext] = None) -> None:
        item.scope_id = item.scope_id or (context.id_for(item.scope) if context else None)
        self._prepare_item(item)
        # mem0-style: check for memory updates on PROJECT/GLOBAL scopes
        if (
            self.enable_memory_update
            and item.scope in self.memory_update_scopes
            and not item.metadata.get("_skip_update_check")
        ):
            updated = self._check_and_update_memory(item, context=context)
            if updated:
                return
        self._dispatch_write(item)

    def route(self, item: MemoryItem) -> MediaRoute:
        """Choose storage media for a memory item.

        Metadata overrides:
        - ``media``: explicit list of medium names.
        - ``temperature``: hot | warm | cold.
        - ``ttl_seconds``: Redis TTL for warm data.
        - ``graph``: truthy enables graph backend write.
        - ``audit``: truthy forces cold JSONL/Markdown archive.
        """
        explicit = item.metadata.get("media")
        if explicit:
            media = [
                medium if isinstance(medium, MemoryMedium) else MemoryMedium(str(medium))
                for medium in explicit
            ]
            return MediaRoute(media=list(dict.fromkeys(media)), ttl_seconds=item.metadata.get("ttl_seconds"))

        media: List[MemoryMedium] = []
        temperature = item.metadata.get("temperature")
        if item.scope == MemoryScope.WORKING or temperature == "hot":
            media.append(MemoryMedium.HOT)
        if temperature == "warm" or item.metadata.get("ttl_seconds") is not None:
            media.append(MemoryMedium.WARM)
        if item.scope in (MemoryScope.TASK, MemoryScope.PROJECT, MemoryScope.GLOBAL):
            media.extend([MemoryMedium.STRUCTURED, MemoryMedium.SEMANTIC])
        if (
            item.scope == MemoryScope.PROJECT
            or temperature == "cold"
            or item.metadata.get("audit")
            or item.token_count >= self.long_text_threshold
        ):
            media.append(MemoryMedium.COLD)
        if item.scope == MemoryScope.GLOBAL:
            media.append(MemoryMedium.WARM)
        if item.metadata.get("graph") or item.metadata.get("entity") or item.metadata.get("relations"):
            media.append(MemoryMedium.GRAPH)
        return MediaRoute(
            media=list(dict.fromkeys(media)),
            ttl_seconds=item.metadata.get("ttl_seconds", self.redis_default_ttl),
        )

    def read(
        self,
        query: str,
        *,
        scope: Optional[MemoryScope] = None,
        context: Optional[MemoryContext] = None,
        top_k: int = 5,
        retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
        temporal_weight: float = 0.0,
    ) -> List[MemoryItem]:
        effective_scope = scope or MemoryScope.TASK
        query_embedding = self.embedding_model.embed(query)
        query_tokens = _tokenize(query)
        if effective_scope == MemoryScope.WORKING:
            candidates = self._read_working(context=context)
        else:
            candidates = self._read_sqlite(effective_scope, context=context)
        ranked = sorted(
            candidates,
            key=lambda item: (
                self._score_item(
                    item,
                    query_embedding=query_embedding,
                    query_tokens=query_tokens,
                    retrieval_mode=retrieval_mode,
                    temporal_weight=temporal_weight,
                ),
                item.importance,
                item.ts,
            ),
            reverse=True,
        )
        return ranked[:top_k]

    def cascade_read(
        self,
        query: str,
        *,
        narrowest: MemoryScope = MemoryScope.WORKING,
        context: Optional[MemoryContext] = None,
        top_k: int = 5,
        retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
        temporal_weight: float = 0.0,
    ) -> List[MemoryItem]:
        hierarchy = MemoryScope.hierarchy()
        start_idx = hierarchy.index(narrowest)
        collected: List[MemoryItem] = []
        for scope in hierarchy[start_idx:]:
            remaining = top_k - len(collected)
            if remaining <= 0:
                break
            items = self.read(
                query,
                scope=scope,
                context=context,
                top_k=remaining,
                retrieval_mode=retrieval_mode,
                temporal_weight=temporal_weight,
            )
            collected.extend(items)
        return collected

    def wake(
        self,
        query: str,
        *,
        profile: WakeupProfile | int | str = WakeupLevel.STANDARD,
        context: Optional[MemoryContext] = None,
    ) -> WakeupResult:
        effective_profile = (
            profile if isinstance(profile, WakeupProfile) else wakeup_profile(profile)
        )
        items = self._read_scopes(
            query,
            scopes=effective_profile.scopes,
            context=context,
            top_k=effective_profile.top_k,
            retrieval_mode=effective_profile.retrieval_mode,
            temporal_weight=effective_profile.temporal_weight,
        )
        context_text = self.format_wakeup_context(items, effective_profile)
        return WakeupResult(profile=effective_profile, items=items, context_text=context_text)

    def clear(
        self,
        scope: Optional[MemoryScope] = None,
        *,
        context: Optional[MemoryContext] = None,
    ) -> None:
        if scope is None or scope == MemoryScope.WORKING:
            if scope is None or context is None:
                self._working.clear()
            else:
                scope_id = context.id_for(MemoryScope.WORKING)
                for key in [k for k, v in self._working.items() if v.scope_id == scope_id]:
                    self._working.pop(key, None)
        if scope == MemoryScope.WORKING:
            return
        conn = self._get_conn()
        if scope is None:
            conn.execute("DELETE FROM memories")
        else:
            scope_id = context.id_for(scope) if context else None
            if scope_id:
                conn.execute(
                    "DELETE FROM memories WHERE scope = ? AND scope_id = ?",
                    (scope.value, scope_id),
                )
            else:
                conn.execute("DELETE FROM memories WHERE scope = ?", (scope.value,))
        conn.commit()

    def expand(self, item: MemoryItem) -> str:
        """Return archived raw text when available, otherwise stringify content."""
        if item.raw_ref:
            path = Path(item.raw_ref)
            if not path.is_absolute():
                path = self.root_dir / path
            if path.exists():
                return path.read_text(encoding="utf-8")
        return _stringify(item.content)

    def format_wakeup_context(
        self, items: List[MemoryItem], profile: WakeupProfile
    ) -> str:
        if not items:
            return ""
        lines = [f"Relevant memory ({profile.name}, level={profile.level.value}):"]
        for idx, item in enumerate(items, 1):
            summary = item.summary or _stringify(item.content)
            ref = f" ref={item.raw_ref}" if item.raw_ref else ""
            lines.append(f"{idx}. [{item.scope.value}]{ref} {summary}")
            should_expand = (
                profile.expand_policy == ArchiveExpandPolicy.EXPAND_ALL
                or (
                    profile.expand_policy == ArchiveExpandPolicy.PRE_EXPAND_TOP
                    and idx <= profile.pre_expand_limit
                )
            )
            if should_expand and item.raw_ref:
                lines.append("   Expanded archive:")
                lines.append(_indent_block(_clip_text(self.expand(item), 4000), "   "))
        if profile.expand_policy == ArchiveExpandPolicy.ON_DEMAND:
            lines.append("Archived items are summarized only; request expansion by ref if needed.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Internal: DB init
    # ------------------------------------------------------------------ #
    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                scope_id TEXT,
                content_json TEXT NOT NULL,
                summary TEXT NOT NULL,
                raw_ref TEXT,
                modality TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                importance REAL NOT NULL,
                token_count INTEGER NOT NULL,
                tags_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                ts REAL NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories(scope, scope_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_ts ON memories(ts)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                scope_id TEXT,
                embedding_json TEXT NOT NULL,
                index_text TEXT NOT NULL,
                ts REAL NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_embedding_scope "
            "ON memory_embeddings(scope, scope_id)"
        )
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Return the persistent SQLite connection.

        The connection uses WAL journal mode for better concurrency and is
        created once during ``__init__`` to avoid the overhead of opening a
        new connection on every operation.
        """
        return self._sqlite_conn

    def close(self) -> None:
        """Close the persistent SQLite connection."""
        try:
            self._sqlite_conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Internal: item preparation
    # ------------------------------------------------------------------ #
    def _prepare_item(self, item: MemoryItem) -> None:
        text = _stringify(item.content)
        item.summary = item.summary or _summarize_text(text, self.summary_max_chars)
        item.token_count = item.token_count or _rough_token_count(text)
        item.embedding = item.embedding or self.embedding_model.embed(
            " ".join([item.summary, " ".join(item.tags), _stringify(item.metadata)])
        )

    # ------------------------------------------------------------------ #
    # Internal: WORKING (LRU)
    # ------------------------------------------------------------------ #
    def _write_working(self, item: MemoryItem) -> None:
        self._working[item.id] = item
        self._working.move_to_end(item.id)
        while len(self._working) > self.working_max_items:
            self._working.popitem(last=False)

    def _read_working(self, *, context: Optional[MemoryContext]) -> List[MemoryItem]:
        scope_id = context.id_for(MemoryScope.WORKING) if context else None
        result: List[MemoryItem] = []
        for key, item in self._working.items():
            if scope_id is None or item.scope_id == scope_id:
                result.append(item)
        # Touch matched items to mark them as recently used (true LRU)
        for item in result:
            self._working.move_to_end(item.id)
        return result

    # ------------------------------------------------------------------ #
    # Internal: COLD archive
    # ------------------------------------------------------------------ #
    def _archive_cold(self, item: MemoryItem) -> None:
        self._append_jsonl_audit(item)
        self._archive_markdown_raw_if_needed(item)

    def _append_jsonl_audit(self, item: MemoryItem) -> None:
        with self.audit_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item.to_dict(), ensure_ascii=False, default=str) + "\n")

    def _archive_markdown_raw_if_needed(self, item: MemoryItem) -> None:
        text = _stringify(item.content)
        if item.raw_ref:
            return
        if item.scope != MemoryScope.PROJECT and len(text) < self.long_text_threshold:
            return
        safe_scope = _safe_filename(item.scope_id or "default")
        folder = self.archive_dir / safe_scope
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{item.id}.md"
        path.write_text(_format_markdown_memory(item, text), encoding="utf-8")
        item.raw_ref = str(path.relative_to(self.root_dir))
        item.content = item.summary
        item.metadata = {**item.metadata, "archived": True}

    # ------------------------------------------------------------------ #
    # Internal: SQLite
    # ------------------------------------------------------------------ #
    def _write_sqlite(self, item: MemoryItem) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO memories (
                id, scope, scope_id, content_json, summary, raw_ref, modality,
                embedding_json, importance, token_count, tags_json, metadata_json, ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                item.scope.value,
                item.scope_id,
                json.dumps(item.content, ensure_ascii=False, default=str),
                item.summary,
                item.raw_ref,
                item.modality,
                json.dumps(item.embedding or []),
                item.importance,
                item.token_count,
                json.dumps(item.tags, ensure_ascii=False),
                json.dumps(item.metadata, ensure_ascii=False, default=str),
                item.ts,
            ),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO memory_embeddings (
                memory_id, scope, scope_id, embedding_json, index_text, ts
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                item.scope.value,
                item.scope_id,
                json.dumps(item.embedding or []),
                " ".join([item.summary, " ".join(item.tags)]),
                item.ts,
            ),
        )
        conn.commit()

    def _read_sqlite(
        self,
        scope: MemoryScope,
        *,
        context: Optional[MemoryContext],
    ) -> List[MemoryItem]:
        scope_id = context.id_for(scope) if context else None
        sql = "SELECT * FROM memories WHERE scope = ?"
        params: List[Any] = [scope.value]
        if scope_id is not None:
            sql += " AND scope_id = ?"
            params.append(scope_id)
        rows = self._get_conn().execute(sql, params).fetchall()
        return [self._row_to_item(row) for row in rows]

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            content=json.loads(row["content_json"]),
            summary=row["summary"],
            raw_ref=row["raw_ref"],
            scope=MemoryScope(row["scope"]),
            scope_id=row["scope_id"],
            modality=row["modality"],
            embedding=json.loads(row["embedding_json"]),
            importance=float(row["importance"]),
            token_count=int(row["token_count"]),
            tags=json.loads(row["tags_json"]),
            metadata=json.loads(row["metadata_json"]),
            ts=float(row["ts"]),
        )

    # ------------------------------------------------------------------ #
    # Internal: cross-scope retrieval
    # ------------------------------------------------------------------ #
    def _read_scopes(
        self,
        query: str,
        *,
        scopes: List[MemoryScope],
        context: Optional[MemoryContext],
        top_k: int,
        retrieval_mode: RetrievalMode,
        temporal_weight: float,
    ) -> List[MemoryItem]:
        query_embedding = self.embedding_model.embed(query)
        query_tokens = _tokenize(query)
        candidates: List[MemoryItem] = []
        for scope in scopes:
            if scope == MemoryScope.WORKING:
                candidates.extend(self._read_working(context=context))
            else:
                candidates.extend(self._read_sqlite(scope, context=context))
        ranked = sorted(
            candidates,
            key=lambda item: (
                self._score_item(
                    item,
                    query_embedding=query_embedding,
                    query_tokens=query_tokens,
                    retrieval_mode=retrieval_mode,
                    temporal_weight=temporal_weight,
                ),
                item.importance,
                item.ts,
            ),
            reverse=True,
        )
        deduped: List[MemoryItem] = []
        seen = set()
        for item in ranked:
            if item.id in seen:
                continue
            seen.add(item.id)
            deduped.append(item)
            if len(deduped) >= top_k:
                break
        return deduped

    def _score_item(
        self,
        item: MemoryItem,
        *,
        query_embedding: List[float],
        query_tokens: List[str],
        retrieval_mode: RetrievalMode,
        temporal_weight: float,
    ) -> float:
        dense = _cosine(query_embedding, item.embedding or [])
        sparse = _sparse_score(query_tokens, _tokenize(_memory_index_text(item)))
        if retrieval_mode == RetrievalMode.DENSE:
            score = dense
        elif retrieval_mode == RetrievalMode.SPARSE:
            score = sparse
        else:
            score = 0.65 * dense + 0.35 * sparse
        if retrieval_mode == RetrievalMode.HYBRID_TEMPORAL and temporal_weight:
            score += temporal_weight * _recency_score(item.ts)
        return score

    # ------------------------------------------------------------------ #
    # Internal: optional backends (Redis / Postgres / Graph)
    # ------------------------------------------------------------------ #
    def _connect_redis(self, redis_url: Optional[str]) -> Any:
        if not redis_url:
            return None
        try:
            import redis  # type: ignore

            return redis.Redis.from_url(redis_url)
        except Exception:
            return None

    def _connect_postgres(self, postgres_dsn: Optional[str]) -> Any:
        if not postgres_dsn:
            return None
        try:
            import psycopg  # type: ignore

            return psycopg.connect(postgres_dsn)
        except Exception:
            return None

    def _write_redis(self, item: MemoryItem, *, ttl_seconds: Optional[int]) -> None:
        if self._redis is None:
            return
        try:
            key = f"memory:{item.scope.value}:{item.scope_id or 'default'}:{item.id}"
            payload = json.dumps(item.to_dict(), ensure_ascii=False, default=str)
            if ttl_seconds:
                self._redis.setex(key, int(ttl_seconds), payload)
            else:
                self._redis.set(key, payload)
        except Exception:
            return

    def _write_graph(self, item: MemoryItem) -> None:
        if self._graph_backend is None:
            return
        payload = item.to_dict()
        for method_name in ("add_memory", "add", "write"):
            method = getattr(self._graph_backend, method_name, None)
            if callable(method):
                try:
                    method(payload)
                except TypeError:
                    method(item)
                except Exception:
                    return
                return

    # ------------------------------------------------------------------ #
    # Internal: write dispatch
    # ------------------------------------------------------------------ #
    def _dispatch_write(self, item: MemoryItem) -> None:
        """Route a prepared item to the appropriate storage backends.

        This is the single source of truth for the write-path routing logic,
        shared by :meth:`write` and :meth:`_replace_memory`.

        .. note:: ``item`` must already be prepared via :meth:`_prepare_item`.
        """
        route = self.route(item)
        item.metadata = {
            **item.metadata,
            "media_route": [medium.value for medium in route.media],
        }
        if route.contains(MemoryMedium.HOT):
            self._write_working(item)
        if route.contains(MemoryMedium.COLD):
            self._archive_cold(item)
        if route.contains(MemoryMedium.STRUCTURED) or route.contains(MemoryMedium.SEMANTIC):
            self._write_sqlite(item)
        if route.contains(MemoryMedium.WARM):
            self._write_redis(item, ttl_seconds=route.ttl_seconds)
        if route.contains(MemoryMedium.GRAPH):
            self._write_graph(item)

    # ------------------------------------------------------------------ #
    # mem0-style memory update / replace mechanism (LLM-based)
    # ------------------------------------------------------------------ #
    def _check_and_update_memory(
        self, new_item: MemoryItem, *, context: Optional[MemoryContext]
    ) -> bool:
        """Check if new memory should update/replace an existing one using LLM.

        This implements a mem0-style memory management strategy:
        1. Search for similar memories in the same scope using embedding similarity
        2. If similar memories found, ask the LLM to decide the action
        3. Execute the LLM's decision (ADD/UPDATE/DELETE/NOOP)

        :param new_item: The new memory item being written.
        :param context: Optional memory context.
        :return: True if the memory was handled (no further write needed), False to proceed with normal write.
        """
        # Ensure we have an LLM judge
        if self.memory_llm_judge is None:
            return False

        # Find similar memories in the same scope
        similar = self._find_similar_in_scope(
            new_item,
            scope=new_item.scope,
            context=context,
            top_k=self.memory_search_top_k,
        )
        if not similar:
            return False

        # Prepare data for LLM judge
        new_memory_text = _stringify(new_item.content)
        if new_item.summary:
            new_memory_text = f"[Summary] {new_item.summary}\n[Content] {new_memory_text}"
        if new_item.tags:
            new_memory_text += f"\n[Tags] {', '.join(new_item.tags)}"

        existing_memories = []
        for candidate, score in similar:
            existing_memories.append({
                "id": candidate.id,
                "content": _stringify(candidate.content),
                "summary": candidate.summary,
                "tags": candidate.tags,
            })

        # Ask LLM for decision
        try:
            decisions = self.memory_llm_judge.judge(new_memory_text, existing_memories)
        except Exception:
            # If LLM fails, fall back to normal write
            return False

        if not decisions:
            return False

        # Build a lookup for quick access
        similar_map = {c.id: c for c, _ in similar}

        # Process decisions — priority: UPDATE > DELETE > ADD > NOOP
        has_add = False
        for decision in decisions:
            mem_id = decision.get("id", "")
            action = decision.get("action", "noop")
            if mem_id not in similar_map:
                continue

            old_item = similar_map[mem_id]

            if action == MemoryUpdateAction.UPDATE.value:
                self._replace_memory(old_item, new_item, context=context)
                return True  # Memory was updated, skip normal write
            elif action == MemoryUpdateAction.DELETE.value:
                # New memory is redundant with existing one, discard it
                return True  # Memory handled (discarded), skip normal write
            elif action == MemoryUpdateAction.ADD.value:
                has_add = True
            # NOOP: skip this decision, continue checking others

        # If ADD was requested (and no UPDATE/DELETE), let normal write proceed
        return has_add

    def _find_similar_in_scope(
        self,
        item: MemoryItem,
        *,
        scope: MemoryScope,
        context: Optional[MemoryContext],
        top_k: int = 3,
        min_similarity: Optional[float] = None,
    ) -> List[tuple]:
        """Find similar memories in the same scope, excluding the item itself.

        :param min_similarity: Minimum cosine similarity threshold. Candidates
            below this score are filtered out to avoid sending irrelevant memories
            to the LLM judge.  Defaults to ``self.memory_search_min_similarity``.
        :return: List of (MemoryItem, similarity_score) tuples, sorted by score desc.
        """
        threshold = min_similarity if min_similarity is not None else self.memory_search_min_similarity
        query_embedding = item.embedding or self.embedding_model.embed(
            " ".join([item.summary, " ".join(item.tags), _stringify(item.metadata)])
        )
        if scope == MemoryScope.WORKING:
            candidates = self._read_working(context=context)
        else:
            candidates = self._read_sqlite(scope, context=context)

        scored: List[tuple] = []
        for candidate in candidates:
            if candidate.id == item.id:
                continue
            candidate_embedding = candidate.embedding or []
            if not candidate_embedding:
                continue
            score = _cosine(query_embedding, candidate_embedding)
            if score < threshold:
                continue
            scored.append((candidate, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _replace_memory(
        self,
        old_item: MemoryItem,
        new_item: MemoryItem,
        *,
        context: Optional[MemoryContext],
    ) -> None:
        """Replace an old memory with a new one.

        Deletes the old memory and writes the new one, preserving a reference
        to the replaced memory ID in metadata.
        """
        new_item.metadata = {
            **new_item.metadata,
            "_replaced_memory_id": old_item.id,
            "_replaced_memory_ts": old_item.ts,
        }
        self.delete(old_item.id)
        # Skip the update check since we already did it
        new_item.metadata["_skip_update_check"] = True
        self._dispatch_write(new_item)
        new_item.metadata.pop("_skip_update_check", None)

    def delete(self, memory_id: str) -> None:
        """Delete a single memory by ID from all storage backends."""
        # Remove from working memory
        self._working.pop(memory_id, None)
        # Remove from SQLite
        conn = self._get_conn()
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,))
        conn.commit()
        # Remove from Redis (best effort, use SCAN to avoid blocking)
        if self._redis is not None:
            try:
                pattern = f"memory:*:*:{memory_id}"
                for key in self._redis.scan_iter(match=pattern, count=100):
                    self._redis.delete(key)
            except Exception:
                pass
