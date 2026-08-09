"""Tests for backend workflow persistence endpoints.

These tests cover the product API path: create an orchestration, save it,
reload it through REST, and execute the loaded graph.
"""

from fastapi.testclient import TestClient

from engine.modules.workflows import WorkflowStore
from engine.server.app import create_app


def test_workflow_save_list_load_and_run(tmp_path):
    app = create_app(workflow_store=WorkflowStore(tmp_path / "workflows"))
    client = TestClient(app)

    parent = client.post("/api/agents", json={"name": "planner"}).json()["id"]
    client.post(
        f"/api/agents/{parent}/sub-agents",
        json={"name": "writer", "auto_connect": True},
    )
    client.post("/api/graph/entry", json={"agent_id": parent})

    saved = client.post(
        "/api/workflows",
        json={"name": "demo workflow", "description": "product smoke test"},
    )
    assert saved.status_code == 200
    workflow = saved.json()
    assert workflow["name"] == "demo workflow"
    assert workflow["graph"]["entry"] == parent

    listed = client.get("/api/workflows")
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == [workflow["id"]]

    client.post("/api/import", json={"entry": None, "agents": [], "connections": []})
    assert client.get("/api/agents").json() == []

    loaded = client.post(f"/api/workflows/{workflow['id']}/load")
    assert loaded.status_code == 200
    assert len(loaded.json()["graph"]["agents"]) == 2

    result = client.post("/api/run", json={"input": {"input": "hello"}})
    assert result.status_code == 200
    assert [item["agent"] for item in result.json()["state"]["messages"]] == [
        "planner",
        "writer",
    ]


def test_workflow_update_validates_payload(tmp_path):
    app = create_app(workflow_store=WorkflowStore(tmp_path / "workflows"))
    client = TestClient(app)

    created = client.post(
        "/api/workflows",
        json={
            "name": "draft",
            "graph": {"entry": None, "agents": [], "connections": []},
        },
    ).json()

    updated = client.put(
        f"/api/workflows/{created['id']}",
        json={"name": "approved", "tags": ["mvp"]},
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "approved"
    assert updated.json()["tags"] == ["mvp"]

    missing = client.get("/api/workflows/not-exists")
    assert missing.status_code == 404
