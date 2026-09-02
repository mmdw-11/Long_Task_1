"""验证应用中心接口会把百炼式应用流程落到现有工作流后端。"""

import time

from fastapi.testclient import TestClient

from engine.modules.product_ops import ApplicationStore, ToolCatalogStore
from engine.modules.model_connections import ModelConnectionStore
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
            "avatar_url": "asset://agent-avatar",
            "knowledge_base_ids": ["kb-a"],
            "memory_bank_ids": ["memory-a"],
        },
    ).json()
    workflow = client.get(f"/api/workflows/{created['workflow_id']}").json()

    assert created["entry_agent_id"]
    assert workflow["metadata"]["application_id"] == created["id"]
    assert workflow["graph"]["entry"] == created["entry_agent_id"]
    assert workflow["graph"]["agents"][0]["sys_prompt"] == "你是邮件助手"
    assert created["avatar_url"] == "asset://agent-avatar"
    assert created["knowledge_base_ids"] == ["kb-a"]
    assert created["memory_bank_ids"] == ["memory-a"]


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
    skill_id = client.get("/api/skills").json()[0]["id"]
    updated = client.put(
        f"/api/apps/{created['id']}",
        json={
            "name": "邮件审核助手",
            "system_prompt": "先审核再回复",
            "tool_ids": ["tool-a"],
            "skill_ids": [skill_id],
            "knowledge_base_ids": ["kb-a"],
            "memory_bank_ids": ["memory-a"],
        },
    ).json()
    workflow = client.get(f"/api/workflows/{updated['workflow_id']}").json()
    agent = workflow["graph"]["agents"][0]

    assert updated["name"] == "邮件审核助手"
    assert agent["name"] == "邮件审核助手"
    assert agent["sys_prompt"] == "先审核再回复"
    assert agent["config"]["tool_ids"] == ["tool-a"]
    assert agent["config"]["skill_ids"] == [skill_id]
    assert agent["config"]["knowledge_base_ids"] == ["kb-a"]
    assert agent["config"]["memory_bank_ids"] == ["memory-a"]


def test_application_defaults_keep_legacy_records_compatible(tmp_path):
    store = ApplicationStore(tmp_path / "apps")
    record = store.create(name="旧应用")

    loaded = store.get(record.id)

    assert loaded.avatar_url == ""
    assert loaded.knowledge_base_ids == []
    assert loaded.memory_bank_ids == []


def test_application_name_is_required(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        tool_catalog_store=ToolCatalogStore(tmp_path / "tools"),
        application_store=ApplicationStore(tmp_path / "apps"),
    )
    client = TestClient(app)

    response = client.post("/api/apps", json={"name": "   "})

    assert response.status_code == 400


def test_create_run_from_application_entry(tmp_path, monkeypatch):
    """应用调试接口会自动使用应用绑定的工作流，隐藏内部编排细节。"""
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        tool_catalog_store=ToolCatalogStore(tmp_path / "tools"),
        application_store=ApplicationStore(tmp_path / "apps"),
    )
    client = TestClient(app)

    created = client.post("/api/apps", json={"name": "邮件助手"}).json()
    run = client.post(
        f"/api/apps/{created['id']}/runs",
        json={"input": {"input": "给张三写一封会议提醒邮件"}},
    ).json()
    history = client.get(f"/api/apps/{created['id']}/runs").json()

    assert run["workflow_id"] == created["workflow_id"]
    assert run["metadata"]["application_id"] == created["id"]
    assert history


def test_create_workflow_application_seeds_start_and_end(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        tool_catalog_store=ToolCatalogStore(tmp_path / "tools"),
        application_store=ApplicationStore(tmp_path / "apps"),
    )
    client = TestClient(app)

    created = client.post("/api/apps", json={"name": "审批流", "app_type": "workflow"})
    assert created.status_code == 200
    payload = created.json()
    workflow = client.get(f"/api/workflows/{payload['workflow_id']}").json()
    kinds = [node["config"]["node_kind"] for node in workflow["graph"]["agents"]]

    assert payload["app_type"] == "workflow"
    assert kinds == ["start", "end"]
    assert workflow["graph"]["entry"] == payload["entry_agent_id"]
    assert workflow["graph"]["connections"][0]["target"] == workflow["graph"]["agents"][1]["id"]
    assert len(workflow["metadata"]["editor"]["positions"]) == 2


def test_workflow_application_mounts_and_runs_tool_node(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    tool_store = ToolCatalogStore(tmp_path / "tools")
    calculator = tool_store.create(
        name="calculator",
        display_name="计算器",
        description="执行安全四则运算",
        metadata={"adapter": "calculator", "risk": "low", "source": "builtin"},
    )
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        tool_catalog_store=tool_store,
        application_store=ApplicationStore(tmp_path / "apps"),
    )
    client = TestClient(app)

    created = client.post("/api/apps", json={"name": "工具工作流", "app_type": "workflow", "model": ""}).json()
    workflow = client.get(f"/api/workflows/{created['workflow_id']}").json()
    start, end = workflow["graph"]["agents"]
    tool_node = {
        "id": "node-calculator",
        "name": "计算器节点",
        "description": "计算表达式",
        "model": "",
        "sys_prompt": "",
        "children": [],
        "config": {
            "node_kind": "tool",
            "tool_id": calculator.id,
            "input_field": "input",
            "output_field": "tool_output",
        },
    }

    updated = client.put(f"/api/apps/{created['id']}", json={"model": "", "tool_ids": [calculator.id]})
    assert updated.status_code == 200
    graph = {
        "entry": start["id"],
        "agents": [start, tool_node, end],
        "connections": [
            {"source": start["id"], "target": tool_node["id"], "conditional": False},
            {"source": tool_node["id"], "target": end["id"], "conditional": False},
        ],
    }
    saved = client.put(f"/api/workflows/{created['workflow_id']}", json={"graph": graph})
    assert saved.status_code == 200

    run = client.post(f"/api/apps/{created['id']}/runs", json={"input": {"input": "3+5"}}).json()
    for _ in range(30):
        if run["status"] != "queued":
            break
        time.sleep(0.1)
        run = client.get(f"/api/runs/{run['id']}").json()

    assert run["status"] == "succeeded"
    assert run["state"]["workflow_output"] == 8.0


def test_workflow_application_rejects_unmounted_tool_node(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    tool_store = ToolCatalogStore(tmp_path / "tools")
    calculator = tool_store.create(name="calculator", display_name="计算器", metadata={"adapter": "calculator"})
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        tool_catalog_store=tool_store,
        application_store=ApplicationStore(tmp_path / "apps"),
    )
    client = TestClient(app)
    created = client.post("/api/apps", json={"name": "未挂载工具工作流", "app_type": "workflow", "model": ""}).json()
    workflow = client.get(f"/api/workflows/{created['workflow_id']}").json()
    start, end = workflow["graph"]["agents"]
    graph = {
        "entry": start["id"],
        "agents": [
            start,
            {"id": "node-calculator", "name": "计算器节点", "description": "", "model": "", "sys_prompt": "", "children": [], "config": {"node_kind": "tool", "tool_id": calculator.id}},
            end,
        ],
        "connections": [
            {"source": start["id"], "target": "node-calculator", "conditional": False},
            {"source": "node-calculator", "target": end["id"], "conditional": False},
        ],
    }

    response = client.put(f"/api/workflows/{created['workflow_id']}", json={"graph": graph})

    assert response.status_code == 400
    assert "未挂载" in response.text


def test_application_rejects_unknown_type(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        application_store=ApplicationStore(tmp_path / "apps"),
    )
    response = TestClient(app).post("/api/apps", json={"name": "未知应用", "app_type": "other"})
    assert response.status_code == 400


def test_prompt_variables_are_persisted_and_required_at_run_time(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        application_store=ApplicationStore(tmp_path / "apps"),
    )
    client = TestClient(app)
    created = client.post("/api/apps", json={
        "name": "变量助手",
        "system_prompt": "请为 ${customer} 生成回复",
        "prompt_variables": [{"name": "customer", "type": "string", "required": True, "default": "", "description": "客户名"}],
    }).json()

    missing = client.post(f"/api/apps/{created['id']}/runs", json={"input": {"input": "你好", "variables": {}}})
    accepted = client.post(f"/api/apps/{created['id']}/runs", json={"input": {"input": "你好", "variables": {"customer": "张三"}}})

    assert created["prompt_variables"][0]["name"] == "customer"
    assert missing.status_code == 400
    assert "customer" in missing.text
    assert accepted.status_code == 200


def test_invalid_or_duplicate_prompt_variables_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    client = TestClient(create_app(application_store=ApplicationStore(tmp_path / "apps")))
    response = client.post("/api/apps", json={
        "name": "错误变量",
        "prompt_variables": [{"name": "1bad", "type": "string"}, {"name": "1bad", "type": "string"}],
    })
    assert response.status_code == 400


def test_tested_model_connection_can_be_bound_to_application(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    models = ModelConnectionStore(tmp_path / "models")
    model = models.create({
        "name": "Edge compatible model", "provider": "compatible", "model_id": "edge-chat",
        "base_url": "https://models.example.com/v1", "tier": "edge", "test_status": "succeeded",
    })
    client = TestClient(create_app(application_store=ApplicationStore(tmp_path / "apps"), model_connection_store=models))

    response = client.post("/api/apps", json={"name": "指定模型助手", "model": model.id})

    assert response.status_code == 200
    assert response.json()["model"] == model.id


def test_model_connection_api_accepts_direct_key_without_returning_it(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    models = ModelConnectionStore(tmp_path / "models")
    client = TestClient(create_app(application_store=ApplicationStore(tmp_path / "apps"), model_connection_store=models))

    response = client.post("/api/model-connections", json={
        "name": "浏览器直接配置", "provider": "openai-compatible", "model_id": "browser-model",
        "base_url": "https://models.example.com/v1", "api_key": "browser-entered-secret", "tier": "cloud",
    })

    assert response.status_code == 200
    assert response.json()["has_api_key"] is True
    assert "browser-entered-secret" not in response.text
    assert models.get(response.json()["id"]).api_key == "browser-entered-secret"


def test_explicit_auto_cannot_publish_until_all_tier_defaults_are_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    client = TestClient(create_app(application_store=ApplicationStore(tmp_path / "apps")))
    created = client.post("/api/apps", json={"name":"AUTO 助手", "model":"auto"}).json()

    response = client.post(f"/api/apps/{created['id']}/publish")

    assert response.status_code == 400
    assert "AUTO" in response.text


def test_model_connection_delete_is_blocked_while_application_uses_it(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    models = ModelConnectionStore(tmp_path / "models")
    model = models.create({"name":"used", "model_id":"used-model", "base_url":"https://used.example/v1", "tier":"cloud", "test_status":"succeeded"})
    client = TestClient(create_app(application_store=ApplicationStore(tmp_path / "apps"), model_connection_store=models))
    client.post("/api/apps", json={"name":"绑定模型应用", "model":model.id})

    blocked = client.delete(f"/api/model-connections/{model.id}")

    assert blocked.status_code == 400
    assert models.exists(model.id)
