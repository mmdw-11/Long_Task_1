"""验证工作流版本历史与回滚。

工作流编辑是前端最频繁的操作之一，后端需要保留历史版本，避免误编辑后无法恢复。
"""

from fastapi.testclient import TestClient

from engine.modules.workflows import WorkflowStore
from engine.server.app import create_app


def test_workflow_versions_and_rollback(tmp_path):
    app = create_app(workflow_store=WorkflowStore(tmp_path / "workflows"))
    client = TestClient(app)

    first_graph = {"entry": None, "agents": [], "connections": []}
    created = client.post(
        "/api/workflows",
        json={"name": "demo", "graph": first_graph},
    )
    assert created.status_code == 200
    workflow_id = created.json()["id"]
    assert created.json()["version"] == 1

    second_graph = {
        "entry": "agent-1",
        "agents": [{"id": "agent-1", "name": "planner"}],
        "connections": [],
    }
    updated = client.put(
        f"/api/workflows/{workflow_id}",
        json={"graph": second_graph, "description": "第二版"},
    )
    assert updated.status_code == 200
    assert updated.json()["version"] == 2

    versions = client.get(f"/api/workflows/{workflow_id}/versions")
    assert versions.status_code == 200
    assert [item["version"] for item in versions.json()] == [2, 1]

    version_one = client.get(f"/api/workflows/{workflow_id}/versions/1")
    assert version_one.status_code == 200
    assert version_one.json()["graph"] == first_graph

    rolled_back = client.post(f"/api/workflows/{workflow_id}/rollback/1")
    assert rolled_back.status_code == 200
    assert rolled_back.json()["version"] == 3
    assert rolled_back.json()["graph"] == first_graph
    assert rolled_back.json()["metadata"]["rollback_to_version"] == 1
