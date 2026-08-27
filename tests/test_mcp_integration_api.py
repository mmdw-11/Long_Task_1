from fastapi.testclient import TestClient

from engine.modules.mcp_integration import MCPConfigStore
from engine.modules.workflows import RunStore, WorkflowStore
from engine.server.app import create_app


def test_agent_mcp_connect_save_and_runtime_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    monkeypatch.setenv("MCP_ALLOW_LOCALHOST", "1")
    monkeypatch.delenv("DEVICE_BASE_URL", raising=False)
    monkeypatch.delenv("DEVICE_MODEL", raising=False)

    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        mcp_config_store=MCPConfigStore(tmp_path / "mcp"),
    )
    client = TestClient(app)

    agent_id = client.post(
        "/api/agents",
        json={"name": "mcp_worker", "description": "MCP 工具 服务 接口"},
    ).json()["id"]
    client.post("/api/graph/entry", json={"agent_id": agent_id})

    tested = client.post(
        "/api/mcp/test",
        json={"endpoint": "http://127.0.0.1:8000/mcp/demo", "auth_type": "none"},
    )
    assert tested.status_code == 200
    discovered = tested.json()["tools"]
    assert {tool["name"] for tool in discovered} == {"preview_email", "lookup_demo"}

    saved = client.post(
        f"/api/agents/{agent_id}/mcp",
        json={
            "name": "本地演示 MCP",
            "endpoint": "http://127.0.0.1:8000/mcp/demo",
            "auth_type": "none",
            "enabled_tools": ["lookup_demo"],
            "discovered_tools": discovered,
        },
    )
    assert saved.status_code == 200
    item = saved.json()["item"]
    assert item["server"]["has_auth_secret"] is False
    assert [tool["name"] for tool in item["tools"] if tool["enabled_for_agent"]] == ["lookup_demo"]

    listed = client.get(f"/api/agents/{agent_id}/mcp").json()["items"]
    assert len(listed) == 1
    assert listed[0]["server"]["name"] == "本地演示 MCP"

    created = client.post("/api/runs", json={"input": {"input": "请使用 MCP 工具 服务 接口 查询演示数据"}}).json()
    record = client.get(f"/api/runs/{created['id']}").json()
    tool_event = next(event for event in record["events"] if event["type"] == "tool_result")

    assert tool_event["tool_call"]["name"] == "lookup_demo"
    assert tool_event["tool_call"]["status"] == "succeeded"
    assert "preview_email" not in str(record["events"])
