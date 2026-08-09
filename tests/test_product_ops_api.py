"""验证产品运维接口。

这些接口为前端提供系统状态与工具目录，不依赖真实模型服务即可工作。
"""

from fastapi.testclient import TestClient

from engine.modules.product_ops import ToolCatalogStore
from engine.modules.workflows import RunStore, WorkflowStore
from engine.server.app import create_app


def test_system_status_endpoint_returns_capabilities(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    monkeypatch.setenv("DEVICE_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("DEVICE_MODEL", "qwen2.5:0.5b-instruct")

    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        tool_catalog_store=ToolCatalogStore(tmp_path / "tools"),
    )
    client = TestClient(app)

    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["capabilities"]["tool_catalog"] is True
    assert data["models"]["device"]["configured"] is True
    assert data["storage"]["tool_root"].endswith("tools")


def test_tool_catalog_crud(tmp_path):
    app = create_app(tool_catalog_store=ToolCatalogStore(tmp_path / "tools"))
    client = TestClient(app)

    created = client.post(
        "/api/tools",
        json={
            "name": "calendar_create",
            "display_name": "创建日历事件",
            "description": "向本地日历写入 ICS 事件",
            "category": "calendar",
            "tags": ["calendar", "ics"],
        },
    )
    assert created.status_code == 200
    tool_id = created.json()["id"]

    listed = client.get("/api/tools")
    assert listed.status_code == 200
    assert listed.json()[0]["name"] == "calendar_create"

    updated = client.put(
        f"/api/tools/{tool_id}",
        json={"enabled": False, "display_name": "创建会议日历事件"},
    )
    assert updated.status_code == 200
    assert updated.json()["enabled"] is False
    assert updated.json()["display_name"] == "创建会议日历事件"

    filtered = client.get("/api/tools?enabled=false")
    assert filtered.status_code == 200
    assert filtered.json()[0]["id"] == tool_id

    deleted = client.delete(f"/api/tools/{tool_id}")
    assert deleted.status_code == 200
    assert client.get("/api/tools").json() == []
