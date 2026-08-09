"""验证后端产品接口使用真实 Agent 运行时。

这些测试不依赖外部模型服务；当没有 DEVICE/EDGE/CLOUD 配置时，项目已有的
LocalEchoExecutor 会作为 fallback 执行，从而保证接口链路可直接调用。
"""

from fastapi.testclient import TestClient

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
    assert "当前 Agent：runtime_agent" in message["content"]


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
