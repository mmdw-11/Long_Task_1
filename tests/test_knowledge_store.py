from engine.modules.knowledge import KnowledgeStore, chunk_text


def test_knowledge_store_upload_chunk_and_retrieve(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    base = store.create_base({"name": "员工手册", "chunk_size": 80}, "user-a")
    document = store.add_document(base.id, "user-a", "manual.md", "# 假期\n\n员工每年享有十天带薪年假。".encode("utf-8"), ["hr"])
    result = store.retrieve([base.id], "带薪年假有几天", owner="user-a", mode="hybrid", threshold=0, labels=["hr"])
    assert document.parse_status == "completed"
    assert result["documents"]
    assert result["citations"][0]["filename"] == "manual.md"


def test_delete_document_clears_index(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    base = store.create_base({"name": "资料"}, "user-a")
    doc = store.add_document(base.id, "user-a", "a.txt", "唯一检索文本".encode())
    store.delete_document(base.id, doc.id, "user-a")
    assert not store.list_chunks(base.id)
    assert store.statistics(base.id)["document_count"] == 0


def test_chunk_strategies_are_nonempty():
    text = "# 标题\n\n第一段内容足够长。\n\n第二段内容足够长。"
    for strategy in ("smart", "paragraph", "heading", "page", "regex"):
        assert chunk_text(text, strategy, 100, 10)
