"""验证应用中心接口会把百炼式应用流程落到现有工作流后端。"""

from fastapi.testclient import TestClient

from engine.modules.product_ops import ApplicationStore, ToolCatalogStore
from engine.modules.workflows import RunStore, WorkflowStore
from engine.server.app import create_app


def test_create_application_creates_workflow_and_entry_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        tool_catalog_store=ToolCatalogStore(tmp_path / "tools"),
        application_store=ApplicationStore(tmp_path / "apps"),
    )
    client = TestClient(app)

    created = client.post(
        "/api/apps",
        json={
            "name": "邮件助手",
            "description": "自动起草邮件",
            "system_prompt": "你是邮件助手",
        },
    ).json()
    workflow = client.get(f"/api/workflows/{created['workflow_id']}").json()

    assert created["entry_agent_id"]
    assert workflow["metadata"]["application_id"] == created["id"]
    assert workflow["graph"]["entry"] == created["entry_agent_id"]
    assert workflow["graph"]["agents"][0]["sys_prompt"] == "你是邮件助手"


def test_update_application_syncs_entry_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        tool_catalog_store=ToolCatalogStore(tmp_path / "tools"),
        application_store=ApplicationStore(tmp_path / "apps"),
    )
    client = TestClient(app)

    created = client.post("/api/apps", json={"name": "邮件助手"}).json()
    updated = client.put(
        f"/api/apps/{created['id']}",
        json={"name": "邮件审核助手", "system_prompt": "先审核再回复", "tool_ids": ["tool-a"]},
    ).json()
    workflow = client.get(f"/api/workflows/{updated['workflow_id']}").json()
    agent = workflow["graph"]["agents"][0]

    assert updated["name"] == "邮件审核助手"
    assert agent["name"] == "邮件审核助手"
    assert agent["sys_prompt"] == "先审核再回复"
    assert agent["config"]["tool_ids"] == ["tool-a"]
