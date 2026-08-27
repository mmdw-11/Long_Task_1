"""验证后端产品接口使用真实 Agent 运行时。

这些测试不依赖外部模型服务；当没有 DEVICE/EDGE/CLOUD 配置时，项目已有的
LocalEchoExecutor 会作为 fallback 执行，从而保证接口链路可直接调用。
"""

from fastapi.testclient import TestClient

from engine.modules.product_ops import ToolCatalogStore
from engine.modules.workflows import RunStore, WorkflowStore
from engine.server.app import create_app


def test_sync_run_uses_inference_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    monkeypatch.delenv("DEVICE_BASE_URL", raising=False)
    monkeypatch.delenv("DEVICE_MODEL", raising=False)

    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
    )
    client = TestClient(app)

    agent_id = client.post(
        "/api/agents",
        json={
            "name": "runtime_agent",
            "sys_prompt": "你是任务执行 Agent。",
            "config": {"tier_preference": ["device"]},
        },
    ).json()["id"]
    client.post("/api/graph/entry", json={"agent_id": agent_id})

    response = client.post("/api/run", json={"input": {"input": "执行一次真实运行链路"}})
    assert response.status_code == 200
    message = response.json()["state"]["messages"][0]
    assert message["runtime"] == "inference"
    assert message["result"]["executor"] == "LocalEchoExecutor"
    assert "当前未配置可用的推理模型" in message["content"]
    assert "当前 Agent：runtime_agent" not in message["content"]
    assert "Context Injection" not in message["content"]


def test_background_run_uses_inference_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    monkeypatch.delenv("DEVICE_BASE_URL", raising=False)
    monkeypatch.delenv("DEVICE_MODEL", raising=False)

    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
    )
    client = TestClient(app)

    agent_id = client.post("/api/agents", json={"name": "worker"}).json()["id"]
    client.post("/api/graph/entry", json={"agent_id": agent_id})
    created = client.post("/api/runs", json={"input": {"input": "后台执行"}}).json()
    record = client.get(f"/api/runs/{created['id']}").json()

    assert record["status"] == "succeeded"
    assert record["state"]["messages"][0]["runtime"] == "inference"


def test_background_run_streams_plan_todo_and_tool_events(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    monkeypatch.delenv("DEVICE_BASE_URL", raising=False)
    monkeypatch.delenv("DEVICE_MODEL", raising=False)

    tool_store = ToolCatalogStore(tmp_path / "tools")
    time_tool = tool_store.create(
        name="current_time",
        display_name="当前时间",
        description="读取当前时间和日期",
        category="system",
        metadata={"adapter": "current_time", "risk": "low"},
    )
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        tool_catalog_store=tool_store,
    )
    client = TestClient(app)

    agent_id = client.post(
        "/api/agents",
        json={"name": "tool_worker", "config": {"tool_ids": [time_tool.id]}},
    ).json()["id"]
    client.post("/api/graph/entry", json={"agent_id": agent_id})

    created = client.post("/api/runs", json={"input": {"input": "请读取当前时间"}}).json()
    record = client.get(f"/api/runs/{created['id']}").json()
    event_types = [event["type"] for event in record["events"]]

    assert record["metadata"]["todos"][0]["status"] == "completed"
    assert "plan_created" in event_types
    assert "todo_updated" in event_types
    assert "tool_result" in event_types
    tool_event = next(event for event in record["events"] if event["type"] == "tool_result")
    assert tool_event["tool_call"]["name"] == "current_time"
    assert tool_event["tool_call"]["status"] == "succeeded"
    assert "当前时间" in record["state"]["messages"][0]["content"]
    assert "Context Injection" not in record["state"]["messages"][0]["content"]


def test_tool_approval_decision_is_audited(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    monkeypatch.delenv("DEVICE_BASE_URL", raising=False)
    monkeypatch.delenv("DEVICE_MODEL", raising=False)

    tool_store = ToolCatalogStore(tmp_path / "tools")
    risky_tool = tool_store.create(
        name="desktop_control",
        display_name="桌面控制",
        description="需要用户审批的桌面工具",
        metadata={"adapter": "desktop_control", "risk": "high"},
    )
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        tool_catalog_store=tool_store,
    )
    client = TestClient(app)

    agent_id = client.post(
        "/api/agents",
        json={
            "name": "approval_worker",
            "description": "桌面控制",
            "config": {"tool_ids": [risky_tool.id]},
        },
    ).json()["id"]
    client.post("/api/graph/entry", json={"agent_id": agent_id})
    created = client.post("/api/runs", json={"input": {"input": "请进行桌面控制"}}).json()
    record = client.get(f"/api/runs/{created['id']}").json()
    approval = next(event for event in record["events"] if event["type"] == "approval_required")

    decided = client.post(
        f"/api/runs/{created['id']}/approvals/{approval['sequence']}/reject",
        json={"reason": "测试拒绝"},
    )

    assert decided.status_code == 200
    payload = decided.json()
    assert payload["metadata"]["approval_decisions"][str(approval["sequence"])]["approved"] is False
    assert payload["events"][-1]["type"] == "tool_result"
    assert payload["events"][-1]["tool_call"]["status"] == "rejected"


def test_approved_high_risk_tool_executes_after_decision(tmp_path, monkeypatch):
    """高风险工具在批准前不执行，批准后由后端适配器真实写入 tool_result。"""
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    tool_store = ToolCatalogStore(tmp_path / "tools")
    risky_tool = tool_store.create(
        name="send_mail_demo", display_name="邮件发送演示", description="发送邮件",
        metadata={"adapter": "echo", "risk": "high"},
    )
    app = create_app(run_store=RunStore(tmp_path / "runs"), tool_catalog_store=tool_store)
    client = TestClient(app)
    agent_id = client.post(
        "/api/agents", json={"name": "邮件助手", "description": "发送邮件", "config": {"tool_ids": [risky_tool.id]}}
    ).json()["id"]
    client.post("/api/graph/entry", json={"agent_id": agent_id})
    created = client.post("/api/runs", json={"input": {"input": "请发送邮件给客户"}}).json()
    record = client.get(f"/api/runs/{created['id']}").json()
    approval = next(event for event in record["events"] if event["type"] == "approval_required")

    approved = client.post(f"/api/runs/{created['id']}/approvals/{approval['sequence']}/approve", json={}).json()
    result = approved["events"][-1]["tool_call"]

    assert approved["metadata"]["approval_decisions"][str(approval["sequence"])]["approved"] is True
    assert result["status"] == "succeeded"
    assert result["name"] == "send_mail_demo"


def test_script_tool_is_registered_but_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    monkeypatch.delenv("AGENTFORGE_ENABLE_SCRIPT_TOOLS", raising=False)
    monkeypatch.delenv("DEVICE_BASE_URL", raising=False)
    monkeypatch.delenv("DEVICE_MODEL", raising=False)

    tool_store = ToolCatalogStore(tmp_path / "tools")
    script_tool = tool_store.create(
        name="email_formatter",
        display_name="邮件格式化脚本",
        description="使用脚本处理邮件内容",
        metadata={
            "source": "script",
            "adapter": "script",
            "risk": "low",
            "script": "print('ok')",
        },
    )
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        tool_catalog_store=tool_store,
    )
    client = TestClient(app)

    agent_id = client.post(
        "/api/agents",
        json={"name": "script_worker", "description": "脚本处理", "config": {"tool_ids": [script_tool.id]}},
    ).json()["id"]
    client.post("/api/graph/entry", json={"agent_id": agent_id})

    created = client.post("/api/runs", json={"input": {"input": "请使用脚本工具处理邮件"}}).json()
    record = client.get(f"/api/runs/{created['id']}").json()
    tool_event = next(event for event in record["events"] if event["type"] == "tool_result")

    assert tool_event["tool_call"]["name"] == "email_formatter"
    assert tool_event["tool_call"]["status"] == "succeeded"
    assert tool_event["tool_call"]["result"]["enabled"] is False
