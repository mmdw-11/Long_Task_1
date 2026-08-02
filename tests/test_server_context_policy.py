import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from engine.modules.context import BUDGET_PAUSED, RUN_STATUS_KEY, ContextPolicy
from engine.server.app import create_app


def test_rest_app_can_enable_context_policy(tmp_path):
    policy = ContextPolicy(max_context_tokens=20, reserved_output_tokens=5)
    app = create_app(context_policy=policy, context_ledger_root=str(tmp_path / "context"))
    client = TestClient(app)

    created = client.post("/api/agents", json={"name": "worker"}).json()
    client.post("/api/graph/entry", json={"agent_id": created["id"]})
    response = client.post(
        "/api/run",
        json={
            "input": {
                "run_id": "rest-policy-run",
                "goal": "x" * 200,
                "input": "x" * 200,
            }
        },
    )

    assert response.status_code == 200
    state = response.json()["state"]
    assert state[RUN_STATUS_KEY] == BUDGET_PAUSED
    assert (tmp_path / "context" / "rest-policy-run" / "ledger.json").exists()


def test_rest_import_preserves_context_policy(tmp_path):
    policy = ContextPolicy(max_context_tokens=1000)
    app = create_app(context_policy=policy, context_ledger_root=str(tmp_path / "context"))
    client = TestClient(app)
    data = {
        "entry": "agent-a",
        "agents": [{"id": "agent-a", "name": "worker", "children": []}],
        "connections": [],
    }

    response = client.post("/api/import", json=data)
    assert response.status_code == 200
    run = client.post(
        "/api/run",
        json={"input": {"run_id": "rest-import-run", "goal": "ok", "input": "hi"}},
    )

    assert run.status_code == 200
    assert "__context_ledger__" in run.json()["state"]
