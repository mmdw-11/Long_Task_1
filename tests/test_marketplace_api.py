"""验证百炼式市场模板会真实落入项目的工具、技能、应用和记忆库。"""

from fastapi.testclient import TestClient

from engine.modules.product_ops import ApplicationStore, MemoryBankStore, ToolCatalogStore
from engine.modules.skills import SkillRepository
from engine.modules.workflows import RunStore, WorkflowStore
from engine.server.app import create_app


def test_marketplace_installation_and_memory_bank(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        skill_repository=SkillRepository(tmp_path / "skills"),
        tool_catalog_store=ToolCatalogStore(tmp_path / "tools"),
        application_store=ApplicationStore(tmp_path / "apps"),
        memory_bank_store=MemoryBankStore(tmp_path / "memory-banks"),
    )
    client = TestClient(app)

    mcp = client.post("/api/marketplace/mcp/email/install")
    skill = client.post("/api/marketplace/skills/email-writer/install")
    application = client.post("/api/marketplace/apps/email-assistant/install")
    memory = client.post("/api/memory-banks", json={"name": "客户沟通记忆库"})

    assert mcp.status_code == 200
    assert mcp.json()["tool"]["metadata"]["needs_configuration"] is True
    assert skill.status_code == 200
    assert skill.json()["skill"]["name"] == "商务邮件撰写"
    assert application.status_code == 200
    assert application.json()["application"]["workflow_id"]
    assert memory.status_code == 200
    assert client.get("/api/memory-banks").json()[0]["name"] == "客户沟通记忆库"
