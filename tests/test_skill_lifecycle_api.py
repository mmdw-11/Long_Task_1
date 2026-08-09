"""Tests for backend skill lifecycle and runtime injection.

The flow mirrors the product loop: run a workflow, generate a candidate skill,
validate and publish it, then reuse the published skill during execution.
"""

from fastapi.testclient import TestClient

from engine.modules.skills import SkillRepository, SkillTraceStore
from engine.modules.workflows import RunStore, WorkflowStore
from engine.server.app import create_app


def test_skill_candidate_publish_and_runtime_injection(tmp_path):
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        skill_repository=SkillRepository(tmp_path / "skills"),
        skill_trace_store=SkillTraceStore(tmp_path / "skill_traces"),
    )
    client = TestClient(app)

    agent_id = client.post("/api/agents", json={"name": "calendar_agent"}).json()["id"]
    client.post("/api/graph/entry", json={"agent_id": agent_id})
    first_run = client.post(
        "/api/runs",
        json={"input": {"input": "安排一次日历会议并记录校验方式"}},
    ).json()
    first_record = client.get(f"/api/runs/{first_run['id']}").json()
    assert first_record["status"] == "succeeded"

    candidate = client.post(
        "/api/skills/candidates/from-run",
        json={
            "run_id": first_run["id"],
            "name": "日历会议执行技能",
            "tags": ["日历", "会议"],
        },
    )
    assert candidate.status_code == 200
    skill_id = candidate.json()["id"]
    assert candidate.json()["status"] == "candidate"

    validation = client.post(f"/api/skills/{skill_id}/validate")
    assert validation.status_code == 200
    assert validation.json()["passed"] is True

    missing_approver = client.post(f"/api/skills/{skill_id}/publish", json={})
    assert missing_approver.status_code == 400

    published = client.post(
        f"/api/skills/{skill_id}/publish",
        json={"approved_by": "pm"},
    )
    assert published.status_code == 200
    assert published.json()["status"] == "published"

    search = client.post(
        "/api/skills/search",
        json={"query": "需要安排日历会议", "node": "calendar_agent"},
    )
    assert search.status_code == 200
    assert search.json()["matches"][0]["skill"]["id"] == skill_id

    second_run = client.post(
        "/api/runs",
        json={"input": {"input": "请安排日历会议"}},
    ).json()
    second_record = client.get(f"/api/runs/{second_run['id']}").json()
    assert "可复用技能" in second_record["state"]["messages"][0]["content"]

    traces = client.get(f"/api/runs/{second_run['id']}/skill-traces")
    assert traces.status_code == 200
    assert any(item["event"] == "skill_retrieved" for item in traces.json()["events"])


def test_manual_skill_lifecycle_guards(tmp_path):
    app = create_app(skill_repository=SkillRepository(tmp_path / "skills"))
    client = TestClient(app)

    created = client.post(
        "/api/skills",
        json={
            "name": "短技能",
            "content": "too short",
        },
    )
    assert created.status_code == 200

    rejected_publish = client.post(
        f"/api/skills/{created.json()['id']}/publish",
        json={"approved_by": "pm"},
    )
    assert rejected_publish.status_code == 400

    listed = client.get("/api/skills?status=draft")
    assert listed.status_code == 200
    assert listed.json()[0]["name"] == "短技能"
