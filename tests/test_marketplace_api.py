"""验证百炼式市场模板会真实落入项目的工具、技能、应用和记忆库。"""

from fastapi.testclient import TestClient

from engine.modules.product_ops import ApplicationStore, ConsoleResourceStore, MemoryBankStore, ToolCatalogStore
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
        console_resource_store=ConsoleResourceStore(tmp_path / "resources"),
    )
    client = TestClient(app)

    mcp = client.post("/api/marketplace/mcp/local-demo/install")
    skill = client.post("/api/marketplace/skills/email-writer/install")
    application = client.post("/api/marketplace/apps/email-assistant/install")
    memory = client.post("/api/memory-banks", json={"name": "客户沟通记忆库"})

    assert mcp.status_code == 200
    assert mcp.json()["tool"]["metadata"]["needs_configuration"] is False
    assert mcp.json()["tool"]["metadata"]["mcp_url"].endswith("/mcp/demo")
    assert len(mcp.json()["connection"]["tools"]) == 2
    assert all(item["metadata"]["connection_id"] == mcp.json()["connection"]["id"] for item in mcp.json()["connection"]["tools"])
    repeated = client.post("/api/marketplace/mcp/local-demo/install")
    assert repeated.status_code == 200
    assert repeated.json()["installed"] is False
    assert len(client.get("/api/tool-connections").json()) == 1
    market = {item["slug"]: item for item in client.get("/api/marketplace/mcp").json()}
    assert market["local-demo"]["installed"] is True
    assert market["local-demo"]["availability"] == "ready"
    assert market["web-search"]["requires_configuration"] is True
    assert skill.status_code == 200
    assert skill.json()["skill"]["name"] == "商务邮件撰写"
    assert application.status_code == 200
    assert application.json()["application"]["workflow_id"]
    assert memory.status_code == 200
    assert client.get("/api/memory-banks").json()[0]["name"] == "客户沟通记忆库"


def test_marketplace_separates_real_remote_mcp_from_builtin_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    monkeypatch.setenv("SEED_BUILTIN_TOOLS", "1")
    app = create_app(tool_catalog_store=ToolCatalogStore(tmp_path / "tools"))
    client = TestClient(app)

    market = {item["slug"]: item for item in client.get("/api/marketplace/mcp").json()}

    assert market["context7"]["mcp_url"] == "https://mcp.context7.com/mcp"
    assert market["context7"]["availability"] == "ready"
    assert market["github"]["requires_configuration"] is True
    builtins = client.get("/api/tools").json()
    assert {item["name"] for item in builtins} >= {"current_time", "calculator", "task_note"}
    assert all(item["metadata"].get("source") == "builtin" for item in builtins)


def test_component_and_console_resources_are_actionable(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"), run_store=RunStore(tmp_path / "runs"),
        tool_catalog_store=ToolCatalogStore(tmp_path / "tools"), application_store=ApplicationStore(tmp_path / "apps"),
        console_resource_store=ConsoleResourceStore(tmp_path / "resources"),
    )
    client = TestClient(app)
    component = client.post("/api/components/todo-node/install")
    knowledge = client.post("/api/resources/knowledge-bases", json={"name": "项目资料"})

    assert component.status_code == 200
    assert client.get("/api/resources/components").json()[0]["name"] == "TODO 规划组件"
    assert knowledge.status_code == 200
    assert client.get("/api/resources/knowledge-bases").json()[0]["name"] == "项目资料"


def test_custom_mcp_is_grouped_without_automatically_binding_an_app(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    monkeypatch.setattr("engine.server.app.validate_remote_url", lambda url: url)
    monkeypatch.setattr("engine.server.app.discover_mcp_tools", lambda *_: [
        {"name":"search","title":"检索","description":"查询资料","inputSchema":{"type":"object"}},
        {"name":"summarize","title":"摘要","description":"生成摘要","inputSchema":{"type":"object"}},
    ])
    app = create_app(
        tool_catalog_store=ToolCatalogStore(tmp_path / "tools"),
        application_store=ApplicationStore(tmp_path / "apps"),
    )
    client = TestClient(app)
    created = client.post("/api/tool-connections/mcp", json={"url":"https://tools.example.com/mcp","name":"资料服务"})
    assert created.status_code == 200
    connection = created.json()["connection"]
    assert connection["name"] == "资料服务"
    assert len(connection["tools"]) == 2
    assert all(item["metadata"]["connection_id"] == connection["id"] for item in connection["tools"])
    application = client.post("/api/apps", json={"name":"未授权工具的应用"}).json()
    assert application["tool_ids"] == []
