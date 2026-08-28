from fastapi.testclient import TestClient

from engine.modules.product_ops import ApplicationStore, MemoryBankStore, ToolCatalogStore
from engine.modules.workflows import RunStore, WorkflowStore
from engine.server.app import create_app


def make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    monkeypatch.setenv("MEMORY_DATA_ROOT", str(tmp_path / "memory_data"))
    return TestClient(create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        tool_catalog_store=ToolCatalogStore(tmp_path / "tools"),
        application_store=ApplicationStore(tmp_path / "apps"),
        memory_bank_store=MemoryBankStore(tmp_path / "banks"),
    ))


def test_primary_and_reference_binding_and_delete_guard(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    primary = client.post("/api/memory-banks", json={"name":"Primary"}).json()
    reference = client.post("/api/memory-banks", json={"name":"Reference"}).json()
    app = client.post("/api/apps", json={
        "name":"Memory Agent",
        "memory_bank_ids":[primary["id"], reference["id"]],
        "primary_memory_bank_id":primary["id"],
    }).json()
    assert app["primary_memory_bank_id"] == primary["id"]
    assert client.delete(f"/api/memory-banks/{reference['id']}").status_code == 409


def test_memory_crud_and_scope_stats(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    bank = client.post("/api/memory-banks", json={"name":"Customer Memory"}).json()
    created = client.post(f"/api/memory-banks/{bank['id']}/memories", json={
        "content":"Customer prefers concise Chinese reports", "scope":"project", "scope_id":"app-a",
    })
    assert created.status_code == 200
    listed = client.get(f"/api/memory-banks/{bank['id']}/memories?scope=project&query=Chinese").json()
    assert listed["total"] == 1
    detail = client.get(f"/api/memory-banks/{bank['id']}").json()
    assert detail["scope_counts"]["project"] == 1
    assert client.delete(f"/api/memory-banks/{bank['id']}/memories/{created.json()['id']}").status_code == 200


def test_application_run_writes_only_primary_bank(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    primary = client.post("/api/memory-banks", json={"name":"Writable"}).json()
    reference = client.post("/api/memory-banks", json={"name":"Read only"}).json()
    app = client.post("/api/apps", json={
        "name":"Memory Writer", "memory_bank_ids":[primary["id"],reference["id"]],
        "primary_memory_bank_id":primary["id"],
    }).json()
    response = client.post(f"/api/apps/{app['id']}/runs", json={"input":{"input":"remember this run"}})
    assert response.status_code == 200
    primary_items = client.get(f"/api/memory-banks/{primary['id']}/memories").json()["items"]
    reference_items = client.get(f"/api/memory-banks/{reference['id']}/memories").json()["items"]
    assert primary_items
    assert reference_items == []


def test_markdown_skill_file_import(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    response = client.post(
        "/api/skills/import/file?filename=meeting.md",
        content="# Meeting Notes\n\nSummarize decisions and owners.",
        headers={"Content-Type":"text/markdown"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "published"
    duplicate = client.post(
        "/api/skills/import/file?filename=meeting-copy.md",
        content="# Meeting Notes\n\nSummarize decisions and owners.",
        headers={"Content-Type":"text/markdown"},
    )
    assert duplicate.json()["id"] == response.json()["id"]
    invalid = client.post("/api/skills/import/file?filename=bad.txt", content="hello")
    assert invalid.status_code == 400
