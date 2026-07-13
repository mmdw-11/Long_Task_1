import pytest

from engine import HybridTieredMemoryStore, MemoryContext, MemoryScope, Orchestrator
from engine import WakeupLevel, wakeup_profile
from engine.modules.memory import MemoryItem, MemoryMedium, HashingEmbeddingModel, MemoryUpdateAction


class MockEmbeddingModel:
    """A mock embedding model for testing memory update logic.

    Produces deterministic embeddings based on content keywords,
    allowing us to control similarity scores in tests.
    """

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list:
        """Generate embedding based on keyword hashing."""
        import hashlib
        import math

        vector = [0.0] * self.dimensions
        words = text.lower().split()
        for word in words:
            digest = hashlib.sha256(word.encode()).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        # Normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector


class MockLLMJudge:
    """Mock LLM judge for testing memory update logic.

    Returns predetermined decisions based on configurable rules.
    """

    def __init__(self, default_action: str = "add") -> None:
        self.default_action = default_action
        self._rules: list = []  # list of (keyword_in_new, keyword_in_existing, action)

    def add_rule(self, new_contains: str, existing_contains: str, action: str) -> None:
        """Add a rule: if new memory contains keyword and existing contains keyword, return action."""
        self._rules.append((new_contains, existing_contains, action))

    def judge(self, new_memory_text: str, existing_memories: list) -> list:
        results = []
        for mem in existing_memories:
            action = self.default_action
            for new_kw, exist_kw, rule_action in self._rules:
                if new_kw.lower() in new_memory_text.lower() and exist_kw.lower() in mem.get("content", "").lower():
                    action = rule_action
                    break
            results.append({"id": mem["id"], "action": action})
        return results


class AlwaysUpdateJudge:
    """Mock judge that always returns UPDATE for the first existing memory."""

    def judge(self, new_memory_text: str, existing_memories: list) -> list:
        if not existing_memories:
            return []
        results = [{"id": existing_memories[0]["id"], "action": "update"}]
        for mem in existing_memories[1:]:
            results.append({"id": mem["id"], "action": "noop"})
        return results


class AlwaysDeleteJudge:
    """Mock judge that always returns DELETE (new memory is redundant)."""

    def judge(self, new_memory_text: str, existing_memories: list) -> list:
        return [{"id": mem["id"], "action": "delete"} for mem in existing_memories]


class AlwaysAddJudge:
    """Mock judge that always returns ADD (keep both)."""

    def judge(self, new_memory_text: str, existing_memories: list) -> list:
        return [{"id": mem["id"], "action": "add"} for mem in existing_memories]


def test_working_memory_uses_lru(tmp_path):
    store = HybridTieredMemoryStore(tmp_path, working_max_items=2)
    ctx = MemoryContext(working_id="node-a")

    store.append("first apple", MemoryScope.WORKING, context=ctx)
    store.append("second banana", MemoryScope.WORKING, context=ctx)
    store.append("third cherry", MemoryScope.WORKING, context=ctx)

    results = store.read("first", scope=MemoryScope.WORKING, context=ctx, top_k=5)
    contents = [item.content for item in results]
    assert "first apple" not in contents
    assert set(contents) == {"second banana", "third cherry"}


def test_task_memory_persists_to_sqlite_and_uses_embedding_search(tmp_path):
    store = HybridTieredMemoryStore(tmp_path)
    ctx = MemoryContext(task_id="task-1")

    store.append("weather report about heavy rain", MemoryScope.TASK, context=ctx)
    store.append("python sqlite memory backend design", MemoryScope.TASK, context=ctx)

    results = store.read("sqlite backend", scope=MemoryScope.TASK, context=ctx, top_k=1)
    assert results[0].content == "python sqlite memory backend design"

    reopened = HybridTieredMemoryStore(tmp_path)
    persisted = reopened.read("heavy rain", scope=MemoryScope.TASK, context=ctx, top_k=1)
    assert persisted[0].content == "weather report about heavy rain"


def test_project_long_text_archives_raw_markdown_and_indexes_summary(tmp_path):
    store = HybridTieredMemoryStore(tmp_path, long_text_threshold=40, summary_max_chars=32)
    ctx = MemoryContext(project_id="project-1")
    raw = "architecture " * 20 + "embedding retrieval markdown archive"

    item = store.append(raw, MemoryScope.PROJECT, context=ctx, tags=["design"])

    assert item.raw_ref is not None
    assert item.metadata["archived"] is True
    archive_path = tmp_path / item.raw_ref
    assert archive_path.exists()
    assert "embedding retrieval markdown archive" in archive_path.read_text(encoding="utf-8")

    results = store.read("architecture retrieval", scope=MemoryScope.PROJECT, context=ctx, top_k=1)
    assert results[0].id == item.id
    assert "architecture" in results[0].summary
    assert "embedding retrieval markdown archive" in store.expand(results[0])
    assert MemoryMedium.COLD.value in item.metadata["media_route"]


def test_global_memory_uses_sqlite_when_redis_is_absent(tmp_path):
    store = HybridTieredMemoryStore(tmp_path)
    ctx = MemoryContext(global_id="org")

    item = store.append("global lesson: always add regression tests", MemoryScope.GLOBAL, context=ctx)

    results = store.read("regression tests", scope=MemoryScope.GLOBAL, context=ctx, top_k=1)
    assert results[0].scope == MemoryScope.GLOBAL
    assert "regression tests" in results[0].content
    assert MemoryMedium.WARM.value in item.metadata["media_route"]


def test_cascade_read_searches_from_narrow_to_broad(tmp_path):
    store = HybridTieredMemoryStore(tmp_path)
    ctx = MemoryContext(working_id="w", task_id="t", project_id="p", global_id="g")

    store.append("task sqlite clue", MemoryScope.TASK, context=ctx)
    store.append("project markdown clue", MemoryScope.PROJECT, context=ctx)

    results = store.cascade_read("clue", narrowest=MemoryScope.WORKING, context=ctx, top_k=2)
    assert [item.scope for item in results] == [MemoryScope.TASK, MemoryScope.PROJECT]


@pytest.mark.asyncio
async def test_hook_injects_relevant_memory_into_node_state(tmp_path):
    store = HybridTieredMemoryStore(tmp_path)
    ctx = MemoryContext(task_id="run-1", project_id="project-1", global_id="org")
    store.append("sqlite migration lesson", MemoryScope.PROJECT, context=ctx)

    orch = Orchestrator()
    orch.set_memory(store)
    orch.set_memory_options(top_k=1)
    node = orch.create_agent("worker")
    orch.set_entry(node)

    graph = orch.build_graph()
    state = await graph.ainvoke(
        {"input": "please use sqlite", "task_id": "run-1", "project_id": "project-1"}
    )

    assert "sqlite migration lesson" in state["worker"]


@pytest.mark.asyncio
async def test_hook_writes_node_output_to_task_memory(tmp_path):
    store = HybridTieredMemoryStore(tmp_path)
    orch = Orchestrator()
    orch.set_memory(store)
    node = orch.create_agent("writer")
    orch.set_entry(node)

    graph = orch.build_graph()
    await graph.ainvoke({"input": "draft memory note", "task_id": "run-2"})

    ctx = MemoryContext(task_id="run-2")
    results = store.read("draft memory note", scope=MemoryScope.TASK, context=ctx)
    assert results
    assert results[0].metadata["node"] == "writer"


def test_media_route_can_be_explicit(tmp_path):
    store = HybridTieredMemoryStore(tmp_path)
    item = MemoryItem(
        "manual warm audit memory",
        metadata={"media": ["hot", "cold"], "audit": True},
    )

    store.write(item)

    assert item.id in store._working
    assert item.metadata["media_route"] == ["hot", "cold"]
    assert (tmp_path / "audit" / "memories.jsonl").exists()


def test_sqlite_has_separate_semantic_index_table(tmp_path):
    store = HybridTieredMemoryStore(tmp_path)
    ctx = MemoryContext(task_id="semantic-task")
    item = store.append("semantic embedding table row", MemoryScope.TASK, context=ctx)

    conn = store._get_conn()
    row = conn.execute(
        "SELECT memory_id, index_text FROM memory_embeddings WHERE memory_id = ?",
        (item.id,),
    ).fetchone()

    assert row["memory_id"] == item.id
    assert "semantic embedding" in row["index_text"]


def test_graph_route_calls_optional_backend(tmp_path):
    class FakeGraph:
        def __init__(self):
            self.payloads = []

        def add_memory(self, payload):
            self.payloads.append(payload)

    graph = FakeGraph()
    store = HybridTieredMemoryStore(tmp_path, graph_backend=graph)
    item = store.append(
        "alice manages project apollo",
        MemoryScope.GLOBAL,
        graph=True,
        entity="alice",
    )

    assert MemoryMedium.GRAPH.value in item.metadata["media_route"]
    assert graph.payloads[0]["id"] == item.id


def test_wakeup_level_zero_is_silent_and_narrow(tmp_path):
    store = HybridTieredMemoryStore(tmp_path)
    ctx = MemoryContext(task_id="t", project_id="p")
    store.append("task dense clue", MemoryScope.TASK, context=ctx)
    store.append("project dense clue", MemoryScope.PROJECT, context=ctx)

    result = store.wake("dense clue", profile=WakeupLevel.SILENT, context=ctx)

    assert result.profile.top_k == 3
    assert {item.scope for item in result.items} == {MemoryScope.TASK}
    assert "Expanded archive" not in result.context_text


def test_wakeup_level_two_pre_expands_top_archive(tmp_path):
    store = HybridTieredMemoryStore(tmp_path, long_text_threshold=20, summary_max_chars=24)
    ctx = MemoryContext(project_id="p")
    raw = "apollo archive " * 20
    store.append(raw, MemoryScope.PROJECT, context=ctx)

    result = store.wake("apollo archive", profile=WakeupLevel.DEEP, context=ctx)

    assert result.profile.pre_expand_limit == 2
    assert "Expanded archive" in result.context_text
    assert "apollo archive" in result.context_text


def test_wakeup_level_three_expands_all_and_uses_temporal_profile(tmp_path):
    store = HybridTieredMemoryStore(
        tmp_path, long_text_threshold=20, summary_max_chars=24,
    )
    ctx = MemoryContext(project_id="p")
    old = store.append("phoenix incident " * 20, MemoryScope.PROJECT, context=ctx)
    old.ts -= 10_000
    store.write(old, context=ctx)
    new = store.append("phoenix incident recent " * 20, MemoryScope.PROJECT, context=ctx)

    result = store.wake("phoenix incident", profile=WakeupLevel.RECOVERY, context=ctx)

    assert result.profile.temporal_weight > 0
    assert result.items[0].id == new.id
    assert result.context_text.count("Expanded archive") >= 2


def test_wakeup_profile_aliases():
    assert wakeup_profile("default").level == WakeupLevel.STANDARD
    assert wakeup_profile("deep").top_k == 8


# ------------------------------------------------------------------ #
# Tests for delete method
# ------------------------------------------------------------------ #
def test_delete_removes_memory_from_sqlite(tmp_path):
    store = HybridTieredMemoryStore(tmp_path)
    ctx = MemoryContext(task_id="task-del")
    item = store.append("memory to delete", MemoryScope.TASK, context=ctx)

    results = store.read("memory to delete", scope=MemoryScope.TASK, context=ctx, top_k=1)
    assert len(results) == 1
    assert results[0].id == item.id

    store.delete(item.id)

    results_after = store.read("memory to delete", scope=MemoryScope.TASK, context=ctx, top_k=5)
    assert all(r.id != item.id for r in results_after)


def test_delete_removes_memory_from_working(tmp_path):
    store = HybridTieredMemoryStore(tmp_path)
    ctx = MemoryContext(working_id="w-del")
    item = store.append("working memory delete", MemoryScope.WORKING, context=ctx)

    assert item.id in store._working
    store.delete(item.id)
    assert item.id not in store._working


def test_delete_removes_from_embedding_table(tmp_path):
    store = HybridTieredMemoryStore(tmp_path)
    ctx = MemoryContext(task_id="task-emb-del")
    item = store.append("embedding table delete test", MemoryScope.TASK, context=ctx)

    conn = store._get_conn()
    row = conn.execute(
        "SELECT memory_id FROM memory_embeddings WHERE memory_id = ?",
        (item.id,),
    ).fetchone()
    assert row is not None

    store.delete(item.id)

    conn = store._get_conn()
    row = conn.execute(
        "SELECT memory_id FROM memory_embeddings WHERE memory_id = ?",
        (item.id,),
    ).fetchone()
    assert row is None


# ------------------------------------------------------------------ #
# Tests for mem0-style memory update mechanism (LLM-based)
# ------------------------------------------------------------------ #
def test_memory_update_replaces_when_llm_says_update(tmp_path):
    """When LLM judge returns UPDATE, the old memory should be replaced."""
    judge = AlwaysUpdateJudge()
    store = HybridTieredMemoryStore(
        tmp_path,
        embedding_model=MockEmbeddingModel(),
        memory_llm_judge=judge,
    )
    ctx = MemoryContext(project_id="proj-update")

    old_item = store.append(
        "user prefers python programming language",
        MemoryScope.PROJECT,
        context=ctx,
        tags=["preference"],
    )

    new_item = store.append(
        "user prefers python programming language for backend development",
        MemoryScope.PROJECT,
        context=ctx,
        tags=["preference"],
    )

    results = store.read("python programming", scope=MemoryScope.PROJECT, context=ctx, top_k=5)
    ids = [r.id for r in results]
    assert old_item.id not in ids
    assert new_item.id in ids


def test_memory_update_discards_when_llm_says_delete(tmp_path):
    """When LLM judge returns DELETE, the new memory is redundant and discarded."""
    judge = AlwaysDeleteJudge()
    store = HybridTieredMemoryStore(
        tmp_path,
        embedding_model=MockEmbeddingModel(),
        memory_llm_judge=judge,
    )
    ctx = MemoryContext(global_id="org-delete")

    old_item = store.append(
        "database migration scheduled for monday",
        MemoryScope.GLOBAL,
        context=ctx,
        tags=["schedule"],
    )

    new_item = store.append(
        "database migration scheduled for monday duplicate",
        MemoryScope.GLOBAL,
        context=ctx,
        tags=["schedule"],
    )

    results = store.read("database migration", scope=MemoryScope.GLOBAL, context=ctx, top_k=5)
    ids = [r.id for r in results]
    # Old memory should still exist, new one should be discarded
    assert old_item.id in ids
    assert new_item.id not in ids


def test_memory_update_adds_when_llm_says_add(tmp_path):
    """When LLM judge returns ADD, both memories should coexist."""
    judge = AlwaysAddJudge()
    store = HybridTieredMemoryStore(
        tmp_path,
        embedding_model=MockEmbeddingModel(),
        memory_llm_judge=judge,
    )
    ctx = MemoryContext(project_id="proj-add")

    item1 = store.append(
        "python web development with fastapi",
        MemoryScope.PROJECT,
        context=ctx,
        tags=["python"],
    )

    item2 = store.append(
        "machine learning model training pipeline",
        MemoryScope.PROJECT,
        context=ctx,
        tags=["ml"],
    )

    results = store.read("development", scope=MemoryScope.PROJECT, context=ctx, top_k=5)
    ids = [r.id for r in results]
    assert item1.id in ids
    assert item2.id in ids


def test_memory_update_disabled_skips_llm_check(tmp_path):
    """When enable_memory_update=False, no LLM check should happen."""
    judge = AlwaysUpdateJudge()
    store = HybridTieredMemoryStore(
        tmp_path,
        embedding_model=MockEmbeddingModel(),
        enable_memory_update=False,
        memory_llm_judge=judge,
    )
    ctx = MemoryContext(project_id="proj-disabled")

    item1 = store.append(
        "configuration setting value alpha",
        MemoryScope.PROJECT,
        context=ctx,
        tags=["config"],
    )

    item2 = store.append(
        "configuration setting value beta",
        MemoryScope.PROJECT,
        context=ctx,
        tags=["config"],
    )

    conn = store._get_conn()
    rows = conn.execute(
        "SELECT id FROM memories WHERE scope = ? AND scope_id = ?",
        (MemoryScope.PROJECT.value, "proj-disabled"),
    ).fetchall()
    ids = [r["id"] for r in rows]
    # Both should exist since update is disabled
    assert item1.id in ids
    assert item2.id in ids


def test_memory_update_task_scope_not_affected(tmp_path):
    """TASK scope should not trigger memory update (only PROJECT/GLOBAL by default)."""
    judge = AlwaysUpdateJudge()
    store = HybridTieredMemoryStore(
        tmp_path,
        embedding_model=MockEmbeddingModel(),
        memory_llm_judge=judge,
    )
    ctx = MemoryContext(task_id="task-nocheck")

    item1 = store.append(
        "task result alpha computation",
        MemoryScope.TASK,
        context=ctx,
        tags=["result"],
    )

    item2 = store.append(
        "task result beta computation",
        MemoryScope.TASK,
        context=ctx,
        tags=["result"],
    )

    results = store.read("task result", scope=MemoryScope.TASK, context=ctx, top_k=5)
    ids = [r.id for r in results]
    # Both should exist since TASK scope is not in update scopes
    assert item1.id in ids
    assert item2.id in ids


def test_memory_update_no_judge_means_no_check(tmp_path):
    """When no memory_llm_judge is set, memory update should be skipped."""
    store = HybridTieredMemoryStore(
        tmp_path,
        embedding_model=MockEmbeddingModel(),
        enable_memory_update=True,
        memory_llm_judge=None,  # No judge
    )
    ctx = MemoryContext(project_id="proj-nojudge")

    item1 = store.append(
        "memory without judge alpha",
        MemoryScope.PROJECT,
        context=ctx,
        tags=["test"],
    )

    item2 = store.append(
        "memory without judge beta",
        MemoryScope.PROJECT,
        context=ctx,
        tags=["test"],
    )

    conn = store._get_conn()
    rows = conn.execute(
        "SELECT id FROM memories WHERE scope = ? AND scope_id = ?",
        (MemoryScope.PROJECT.value, "proj-nojudge"),
    ).fetchall()
    ids = [r["id"] for r in rows]
    # Both should exist since no LLM judge is configured
    assert item1.id in ids
    assert item2.id in ids


def test_replaced_memory_preserves_reference_in_metadata(tmp_path):
    """When a memory replaces another, the old ID should be recorded in metadata."""
    judge = AlwaysUpdateJudge()
    store = HybridTieredMemoryStore(
        tmp_path,
        embedding_model=MockEmbeddingModel(),
        memory_llm_judge=judge,
    )
    ctx = MemoryContext(project_id="proj-ref")

    old_item = store.append(
        "api endpoint configuration version one",
        MemoryScope.PROJECT,
        context=ctx,
        tags=["api"],
    )

    new_item = store.append(
        "api endpoint configuration version two",
        MemoryScope.PROJECT,
        context=ctx,
        tags=["api"],
    )

    # The new item should have metadata referencing the old one
    assert new_item.metadata.get("_replaced_memory_id") == old_item.id


def test_mock_llm_judge_with_rules(tmp_path):
    """Test MockLLMJudge with configurable rules."""
    judge = MockLLMJudge(default_action="add")
    judge.add_rule("backend", "frontend", "update")

    store = HybridTieredMemoryStore(
        tmp_path,
        embedding_model=MockEmbeddingModel(),
        memory_llm_judge=judge,
    )
    ctx = MemoryContext(project_id="proj-rules")

    old_item = store.append(
        "user is a frontend developer",
        MemoryScope.PROJECT,
        context=ctx,
        tags=["role"],
    )

    # This should trigger the rule (new contains 'backend', existing contains 'frontend')
    new_item = store.append(
        "user is now a backend developer",
        MemoryScope.PROJECT,
        context=ctx,
        tags=["role"],
    )

    results = store.read("developer", scope=MemoryScope.PROJECT, context=ctx, top_k=5)
    ids = [r.id for r in results]
    assert old_item.id not in ids
    assert new_item.id in ids


# ------------------------------------------------------------------ #
# Test for BGEM3EmbeddingModel (skip if not installed)
# ------------------------------------------------------------------ #
def test_bge_m3_embedding_model_import():
    """Test that BGEM3EmbeddingModel can be imported."""
    from engine.modules.memory import BGEM3EmbeddingModel
    assert BGEM3EmbeddingModel is not None


def test_bge_m3_embedding_model_produces_vectors():
    """Test that BGEM3EmbeddingModel produces correct dimension vectors."""
    try:
        import FlagEmbedding  # noqa: F401
    except ImportError:
        pytest.skip("FlagEmbedding not installed")
    from engine.modules.memory import BGEM3EmbeddingModel
    model = BGEM3EmbeddingModel()
    vector = model.embed("test sentence")
    assert isinstance(vector, list)
    assert len(vector) == 1024  # BGE-M3 produces 1024-dim vectors
