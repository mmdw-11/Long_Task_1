from fastapi.testclient import TestClient

from engine.modules.auth import AuthStore
from engine.modules.product_ops import ApplicationStore
from engine.modules.skills import SkillRepository
from engine.modules.workflows import RunStore, WorkflowStore
from engine.server.app import create_app


def _register(client: TestClient, email: str) -> None:
    response = client.post(
        "/api/auth/register",
        json={"email": email, "name": email.split("@")[0], "password": "password-123"},
    )
    assert response.status_code == 200


def test_builtin_and_private_skills_are_immediately_usable_and_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        application_store=ApplicationStore(tmp_path / "apps"),
        skill_repository=SkillRepository(tmp_path / "skills"),
        auth_store=AuthStore(tmp_path / "auth.sqlite3"),
        auth_required=True,
    )
    alice, bob = TestClient(app), TestClient(app)
    _register(alice, "alice@example.com")
    _register(bob, "bob@example.com")

    builtin = alice.get("/api/skills").json()
    assert len(builtin) == 6
    assert all(item["visibility"] == "builtin" and item["status"] == "published" for item in builtin)

    private = alice.post(
        "/api/skills",
        json={"name": "Alice 写作", "content": "# Alice 写作\n\n只输出简洁结论。"},
    )
    assert private.status_code == 200
    private_skill = private.json()
    assert private_skill["visibility"] == "private"
    assert private_skill["status"] == "published"
    assert private_skill["validation_status"] == "passed"
    assert private_skill["id"] not in {item["id"] for item in bob.get("/api/skills").json()}
    assert bob.get(f"/api/skills/{private_skill['id']}").status_code == 404

    alice_app = alice.post(
        "/api/apps", json={"name": "Alice 应用", "skill_ids": [private_skill["id"]]}
    )
    assert alice_app.status_code == 200
    app_id = alice_app.json()["id"]
    assert bob.get(f"/api/apps/{app_id}").status_code == 404
    assert bob.post("/api/apps", json={"name": "越权应用", "skill_ids": [private_skill["id"]]}).status_code == 400

    assert alice.put(
        f"/api/skills/{builtin[0]['id']}",
        json={"name": "覆盖内置", "content": "# 覆盖", "description": "", "tags": [], "metadata": {}},
    ).status_code == 403
    assert alice.delete(f"/api/skills/{builtin[0]['id']}").status_code == 403


def test_empty_skill_selection_is_preserved_on_application_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    client = TestClient(
        create_app(
            workflow_store=WorkflowStore(tmp_path / "workflows"),
            application_store=ApplicationStore(tmp_path / "apps"),
            skill_repository=SkillRepository(tmp_path / "skills"),
        )
    )
    created = client.post("/api/apps", json={"name": "无 Skill 应用", "skill_ids": []}).json()
    workflow = client.get(f"/api/workflows/{created['workflow_id']}").json()
    assert "skill_ids" in workflow["graph"]["agents"][0]["config"]
    assert workflow["graph"]["agents"][0]["config"]["skill_ids"] == []
