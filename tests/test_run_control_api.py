"""验证运行控制接口。

覆盖长任务产品必需的取消、重试和指标统计能力。测试使用开发环境 fallback，
不依赖真实模型服务。
"""

from fastapi.testclient import TestClient

from engine.modules.workflows import RunStore, WorkflowStore
from engine.server.app import create_app


def test_cancel_created_run_and_metrics(tmp_path):
    run_store = RunStore(tmp_path / "runs")
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=run_store,
    )
    client = TestClient(app)
    record = run_store.create(input={"input": "等待执行"}, recursion_limit=10)

    canceled = client.post(
        f"/api/runs/{record.id}/cancel",
        json={"reason": "用户取消"},
    )
    assert canceled.status_code == 200
    assert canceled.json()["status"] == "canceled"
    assert canceled.json()["metadata"]["cancel_reason"] == "用户取消"

    metrics = client.get("/api/system/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["runs"]["status_counts"]["canceled"] == 1


def test_retry_run_creates_linked_execution(tmp_path, monkeypatch):
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
    original = client.post("/api/runs", json={"input": {"input": "第一次执行"}}).json()

    retried = client.post(f"/api/runs/{original['id']}/retry")
    assert retried.status_code == 200
    retry_data = retried.json()
    assert retry_data["parent_run_id"] == original["id"]
    assert retry_data["retry_count"] == 1

    fetched = client.get(f"/api/runs/{retry_data['id']}").json()
    assert fetched["status"] == "succeeded"
    assert fetched["state"]["messages"][0]["agent"] == "worker"
