"""验证产品运维接口。

这些接口为前端提供系统状态与工具目录，不依赖真实模型服务即可工作。
"""

from fastapi.testclient import TestClient

from engine.modules.product_ops import ToolCatalogStore
from engine.modules.skills import SkillRepository, SkillStatus
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


def test_system_export_snapshot_can_hide_large_sections(tmp_path):
    # 系统快照用于备份和排障；查询参数允许前端按场景裁剪大字段。
    workflows = WorkflowStore(tmp_path / "workflows")
    runs = RunStore(tmp_path / "runs")
    skills = SkillRepository(tmp_path / "skills")
    tools = ToolCatalogStore(tmp_path / "tools")
    workflows.create(name="演示流程", graph={"agents": [], "edges": []})
    runs.create(input={"input": "hello"}, recursion_limit=3)
    skills.create(
        name="导出技能",
        status=SkillStatus.PUBLISHED,
        content="# 导出技能\n\n## 适用场景\n- 导出。\n\n## 执行步骤\n1. 收集数据。\n\n## 校验方式\n- 检查数量。",
    )
    tools.create(name="demo_tool", display_name="演示工具")

    app = create_app(
        workflow_store=workflows,
        run_store=runs,
        skill_repository=skills,
        tool_catalog_store=tools,
    )
    client = TestClient(app)

    full_snapshot = client.get("/api/system/export")
    assert full_snapshot.status_code == 200
    assert full_snapshot.json()["summary"] == {
        "workflows": 1,
        "runs": 1,
        "skills": 1,
        "tools": 1,
    }
    assert "content" in full_snapshot.json()["skills"][0]

    slim_snapshot = client.get(
        "/api/system/export?include_runs=false&include_skill_content=false"
    )
    assert slim_snapshot.status_code == 200
    assert slim_snapshot.json()["summary"]["runs"] == 0
    assert slim_snapshot.json()["runs"] == []
    assert "content" not in slim_snapshot.json()["skills"][0]
