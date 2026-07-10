from engine import HybridTieredMemoryStore, MemoryContext, MemoryScope


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


def test_global_memory_uses_sqlite_when_redis_is_absent(tmp_path):
    store = HybridTieredMemoryStore(tmp_path)
    ctx = MemoryContext(global_id="org")

    store.append("global lesson: always add regression tests", MemoryScope.GLOBAL, context=ctx)

    results = store.read("regression tests", scope=MemoryScope.GLOBAL, context=ctx, top_k=1)
    assert results[0].scope == MemoryScope.GLOBAL
    assert "regression tests" in results[0].content


def test_cascade_read_searches_from_narrow_to_broad(tmp_path):
    store = HybridTieredMemoryStore(tmp_path)
    ctx = MemoryContext(working_id="w", task_id="t", project_id="p", global_id="g")

    store.append("task sqlite clue", MemoryScope.TASK, context=ctx)
    store.append("project markdown clue", MemoryScope.PROJECT, context=ctx)

    results = store.cascade_read("clue", narrowest=MemoryScope.WORKING, context=ctx, top_k=2)
    assert [item.scope for item in results] == [MemoryScope.TASK, MemoryScope.PROJECT]
