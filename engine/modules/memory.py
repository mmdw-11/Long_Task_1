"""记忆模块：四级层级记忆的读写与检索。

``MemoryStore`` 抽象了一套四级层级记忆存储接口：

- **WORKING**（工作级）: 当前节点单次执行内的临时上下文，执行结束即可丢弃。
- **TASK**（任务级）: 一次图执行（run）内的跨节点共享记忆，run 结束后可归档或丢弃。
- **PROJECT**（项目级）: 同一编排定义（workflow）跨多次 run 的持久化记忆。
- **GLOBAL**（全局级）: 跨项目的全局知识，长期持久化。

层级从窄到宽排列：WORKING → TASK → PROJECT → GLOBAL。
检索时支持**级联查找**：从指定层级开始，逐级向上扩展搜索范围，直至收集到
足够结果或穷尽所有层级。

本文件仅提供接口与空实现桩（``NoOpMemoryStore`` 写入丢弃、读取返回空）。
"""

from __future__ import annotations

import enum
import time
from abc import ABC, abstractmethod
import hashlib
import json
import math
import re
import sqlite3
import uuid
from collections import OrderedDict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol


class MemoryScope(str, enum.Enum):
    """记忆作用域（层级从窄到宽）。"""

    WORKING = "working"    # 工作级：当前节点执行内
    TASK = "task"          # 任务级：一次图执行（run）内
    PROJECT = "project"    # 项目级：同一编排定义跨 run
    GLOBAL = "global"      # 全局级：跨项目

    @staticmethod
    def hierarchy() -> List["MemoryScope"]:
        """返回从窄到宽的层级顺序。"""
        return [
            MemoryScope.WORKING,
            MemoryScope.TASK,
            MemoryScope.PROJECT,
            MemoryScope.GLOBAL,
        ]


class MemoryMedium(str, enum.Enum):
    """Storage media used inside each memory scope."""

    HOT = "hot"                # In-process dict / LRU.
    WARM = "warm"              # Redis, optionally shared and TTL-based.
    COLD = "cold"              # Markdown / JSONL audit archive.
    STRUCTURED = "structured"  # SQLite / PostgreSQL row store.
    SEMANTIC = "semantic"      # Vector index, currently SQLite embedding rows.
    GRAPH = "graph"            # Zep / Graphiti style graph memory.


@dataclass
class MediaRoute:
    """Resolved media destinations for a memory item."""

    media: List[MemoryMedium]
    ttl_seconds: Optional[int] = None

    def contains(self, medium: MemoryMedium) -> bool:
        return medium in self.media


@dataclass
class MemoryContext:
    """记忆上下文：标识当前操作所属的各级作用域 id。

    未设置的层级视为 ``None``，表示该层级不参与精确匹配。
    级联检索时，从 ``narrowest`` 指定的层级开始逐级向上。

    :param working_id:  当前节点执行标识（如 ``"step_3:researcher"``）。
    :param task_id:     当前图执行标识（如一次 run 的 uuid）。
    :param project_id:  当前编排/项目标识（如 orchestrator 名称或 id）。
    :param global_id:   全局命名空间（通常为固定值，如 ``"default"``）。
    """

    working_id: Optional[str] = None
    task_id: Optional[str] = None
    project_id: Optional[str] = None
    global_id: Optional[str] = "default"

    def id_for(self, scope: MemoryScope) -> Optional[str]:
        """返回指定层级对应的 id。"""
        return {
            MemoryScope.WORKING: self.working_id,
            MemoryScope.TASK: self.task_id,
            MemoryScope.PROJECT: self.project_id,
            MemoryScope.GLOBAL: self.global_id,
        }.get(scope)


@dataclass
class MemoryItem:
    """一条记忆。

    :param content:  记忆内容（任意可序列化结构）。
    :param scope:    该条记忆写入的作用域层级。
    :param scope_id: 该条记忆绑定的作用域实例 id（由 MemoryContext 提供）。
    :param tags:     标签列表，用于过滤与分类检索。
    :param metadata: 附加元信息。
    :param ts:       写入时间戳。
    """

    content: Any
    scope: MemoryScope = MemoryScope.TASK
    scope_id: Optional[str] = None
    id: str = field(default_factory=lambda: f"mem-{uuid.uuid4().hex}")
    summary: str = ""
    raw_ref: Optional[str] = None
    modality: str = "text"
    embedding: Optional[List[float]] = None
    importance: float = 0.0
    token_count: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "summary": self.summary,
            "raw_ref": self.raw_ref,
            "scope": self.scope.value,
            "scope_id": self.scope_id,
            "modality": self.modality,
            "embedding": self.embedding,
            "importance": self.importance,
            "token_count": self.token_count,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "ts": self.ts,
        }


class MemoryStore(ABC):
    """四级层级记忆存储接口。

    实现者应按 ``MemoryScope`` 的四个层级分别维护存储（可以是不同的后端），
    并支持级联检索。
    """

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #
    @abstractmethod
    def write(self, item: MemoryItem, *, context: Optional[MemoryContext] = None) -> None:
        """写入一条记忆。

        :param item:    待写入的记忆条目。
        :param context: 可选的上下文；若提供，可用 ``context.id_for(item.scope)``
            自动填充 ``item.scope_id``。
        """
        raise NotImplementedError

    def append(
        self,
        content: Any,
        scope: MemoryScope = MemoryScope.TASK,
        *,
        context: Optional[MemoryContext] = None,
        tags: Optional[List[str]] = None,
        **metadata: Any,
    ) -> MemoryItem:
        """便捷写入：从内容构造 ``MemoryItem`` 并写入。"""
        scope_id = context.id_for(scope) if context else None
        item = MemoryItem(
            content=content,
            scope=scope,
            scope_id=scope_id,
            tags=list(tags or []),
            metadata=dict(metadata),
        )
        self.write(item, context=context)
        return item

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    @abstractmethod
    def read(
        self,
        query: str,
        *,
        scope: Optional[MemoryScope] = None,
        context: Optional[MemoryContext] = None,
        top_k: int = 5,
    ) -> List[MemoryItem]:
        """检索相关记忆。

        :param query:   检索关键词 / 查询条件。
        :param scope:   限定作用域层级；为 None 时等价于 ``MemoryScope.TASK``。
        :param context: 可选上下文，用于按 scope_id 精确过滤。
        :param top_k:   最多返回条数。
        :return: 匹配的记忆列表（按相关性 / 时间排序）。
        """
        raise NotImplementedError

    def cascade_read(
        self,
        query: str,
        *,
        narrowest: MemoryScope = MemoryScope.WORKING,
        context: Optional[MemoryContext] = None,
        top_k: int = 5,
    ) -> List[MemoryItem]:
        """级联检索：从 ``narrowest`` 层级开始逐级向上查找，直至收集到
        足够结果或穷尽 GLOBAL。

        默认实现依次调用 :meth:`read` 并合并结果；子类可覆盖以优化性能。

        :param narrowest: 起始（最窄）层级。
        :param context:   可选上下文。
        :param top_k:     总共最多返回条数。
        """
        hierarchy = MemoryScope.hierarchy()
        start_idx = hierarchy.index(narrowest)
        collected: List[MemoryItem] = []
        for scope in hierarchy[start_idx:]:
            remaining = top_k - len(collected)
            if remaining <= 0:
                break
            items = self.read(query, scope=scope, context=context, top_k=remaining)
            collected.extend(items)
        return collected

    # ------------------------------------------------------------------ #
    # 清空
    # ------------------------------------------------------------------ #
    @abstractmethod
    def clear(
        self,
        scope: Optional[MemoryScope] = None,
        *,
        context: Optional[MemoryContext] = None,
    ) -> None:
        """清空记忆。

        :param scope:   限定层级；None 表示清空所有层级。
        :param context: 可选上下文；若提供且 scope 不为 None，则只清空该
            scope_id 对应的记忆（不影响同层级其他实例）。
        """
        raise NotImplementedError


class NoOpMemoryStore(MemoryStore):
    """空实现桩：写入丢弃、读取返回空列表。"""

    def write(self, item: MemoryItem, *, context: Optional[MemoryContext] = None) -> None:
        return None

    def read(
        self,
        query: str,
        *,
        scope: Optional[MemoryScope] = None,
        context: Optional[MemoryContext] = None,
        top_k: int = 5,
    ) -> List[MemoryItem]:
        return []

    def clear(
        self,
        scope: Optional[MemoryScope] = None,
        *,
        context: Optional[MemoryContext] = None,
    ) -> None:
        return None


class EmbeddingModel(Protocol):
    """Small protocol for pluggable local or remote embedding providers."""

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError


class HashingEmbeddingModel:
    """Deterministic local embedding based on feature hashing.

    This is intentionally dependency-free. It gives usable semantic-ish keyword
    matching for tests and local development, and can be replaced by an OpenAI,
    sentence-transformers, Mem0, or Zep embedding adapter later.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        return _normalize(vector)


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
        self._init_db()

    def write(self, item: MemoryItem, *, context: Optional[MemoryContext] = None) -> None:
        item.scope_id = item.scope_id or (context.id_for(item.scope) if context else None)
        self._prepare_item(item)
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
    ) -> List[MemoryItem]:
        effective_scope = scope or MemoryScope.TASK
        query_embedding = self.embedding_model.embed(query)
        if effective_scope == MemoryScope.WORKING:
            candidates = self._read_working(context=context)
        else:
            candidates = self._read_sqlite(effective_scope, context=context)
        ranked = sorted(
            candidates,
            key=lambda item: (
                _cosine(query_embedding, item.embedding or []),
                item.importance,
                item.ts,
            ),
            reverse=True,
        )
        return ranked[:top_k]

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
        with self._connect() as conn:
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

    def expand(self, item: MemoryItem) -> str:
        """Return archived raw text when available, otherwise stringify content."""
        if item.raw_ref:
            path = Path(item.raw_ref)
            if not path.is_absolute():
                path = self.root_dir / path
            if path.exists():
                return path.read_text(encoding="utf-8")
        return _stringify(item.content)

    def _init_db(self) -> None:
        with self._connect() as conn:
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

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _prepare_item(self, item: MemoryItem) -> None:
        text = _stringify(item.content)
        item.summary = item.summary or _summarize_text(text, self.summary_max_chars)
        item.token_count = item.token_count or _rough_token_count(text)
        item.embedding = item.embedding or self.embedding_model.embed(
            " ".join([item.summary, " ".join(item.tags), _stringify(item.metadata)])
        )

    def _write_working(self, item: MemoryItem) -> None:
        self._working[item.id] = item
        self._working.move_to_end(item.id)
        while len(self._working) > self.working_max_items:
            self._working.popitem(last=False)

    def _read_working(self, *, context: Optional[MemoryContext]) -> List[MemoryItem]:
        scope_id = context.id_for(MemoryScope.WORKING) if context else None
        return [
            item
            for item in self._working.values()
            if scope_id is None or item.scope_id == scope_id
        ]

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

    def _write_sqlite(self, item: MemoryItem) -> None:
        with self._connect() as conn:
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
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
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


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def _normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _cosine(left: List[float], right: List[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    return sum(left[i] * right[i] for i in range(size))


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _summarize_text(text: str, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _rough_token_count(text: str) -> int:
    return max(1, len(_tokenize(text)))


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "default"


def _format_markdown_memory(item: MemoryItem, raw_text: str) -> str:
    metadata = {
        "id": item.id,
        "scope": item.scope.value,
        "scope_id": item.scope_id,
        "tags": item.tags,
        "ts": item.ts,
        **item.metadata,
    }
    return (
        "---\n"
        f"{json.dumps(metadata, ensure_ascii=False, indent=2)}\n"
        "---\n\n"
        f"# Memory {item.id}\n\n"
        "## Summary\n\n"
        f"{item.summary}\n\n"
        "## Raw\n\n"
        f"{raw_text}\n"
    )
