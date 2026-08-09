"""Tests for persisted backend run lifecycle APIs.

The run center lets a frontend start execution, poll status, and inspect the
event trail without coupling to the in-memory graph object.
"""

from fastapi.testclient import TestClient

from engine.modules.workflows import RunStore, WorkflowStore
from engine.server.app import create_app


def test_background_run_records_events_and_result(tmp_path):
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
    )
    client = TestClient(app)

    first = client.post("/api/agents", json={"name": "first"}).json()["id"]
    client.post(f"/api/agents/{first}/sub-agents", json={"name": "second"})
    client.post("/api/graph/entry", json={"agent_id": first})

    created = client.post("/api/runs", json={"input": {"input": "ping"}})
    assert created.status_code == 200
    run_id = created.json()["id"]

    fetched = client.get(f"/api/runs/{run_id}")
    assert fetched.status_code == 200
    record = fetched.json()
    assert record["status"] == "succeeded"
    assert record["finished_at"]
    assert [item["agent"] for item in record["state"]["messages"]] == [
        "first",
        "second",
    ]
    assert any(event["type"] == "route" for event in record["events"])


def test_background_run_can_execute_saved_workflow(tmp_path):
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
    )
    client = TestClient(app)

    agent_id = client.post("/api/agents", json={"name": "saved_agent"}).json()["id"]
    client.post("/api/graph/entry", json={"agent_id": agent_id})
    workflow = client.post("/api/workflows", json={"name": "saved"}).json()
    client.post("/api/import", json={"entry": None, "agents": [], "connections": []})

    created = client.post(
        "/api/runs",
        json={"workflow_id": workflow["id"], "input": {"input": "from saved"}},
    )
    assert created.status_code == 200

    runs = client.get(f"/api/runs?workflow_id={workflow['id']}").json()
    assert len(runs) == 1
    assert runs[0]["status"] == "succeeded"
    assert runs[0]["state"]["messages"][0]["agent"] == "saved_agent"
