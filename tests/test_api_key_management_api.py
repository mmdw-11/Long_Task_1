"""验证控制台 API Key 管理接口的创建、列表和禁用流程。"""

from fastapi.testclient import TestClient

from engine.modules.product_ops import ApiKeyStore
from engine.modules.workflows import RunStore, WorkflowStore
from engine.server.app import create_app


def test_api_key_secret_is_returned_once(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        api_key_store=ApiKeyStore(tmp_path / "api_keys"),
    )
    client = TestClient(app)

    created = client.post("/api/api-keys", json={"name": "演示密钥"}).json()
    listed = client.get("/api/api-keys").json()

    assert created["secret"].startswith("af-")
    assert listed[0]["prefix"] == created["prefix"]
    assert "secret" not in listed[0]


def test_api_key_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        api_key_store=ApiKeyStore(tmp_path / "api_keys"),
    )
    client = TestClient(app)

    created = client.post("/api/api-keys", json={"name": "演示密钥"}).json()
    updated = client.put(f"/api/api-keys/{created['id']}", json={"enabled": False}).json()

    assert updated["enabled"] is False
