"""Backend product-loop smoke demo.

Run with:
    D:\SoftWare\Anaconda\envs\Lang_Task\python.exe examples\backend_product_loop_demo.py

The demo uses REST handlers in-process: create workflow, run it, generate and
publish a skill, then run again with published skill injection.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from engine.modules.skills import SkillRepository, SkillTraceStore
from engine.modules.workflows import RunStore, WorkflowStore
from engine.server.app import create_app


def main() -> None:
    root = Path("runs/demo_product_loop")
    app = create_app(
        workflow_store=WorkflowStore(root / "workflows"),
        run_store=RunStore(root / "runs"),
        skill_repository=SkillRepository(root / "skills"),
        skill_trace_store=SkillTraceStore(root / "skill_traces"),
    )
    client = TestClient(app)

    planner = client.post("/api/agents", json={"name": "planner"}).json()["id"]
    client.post(f"/api/agents/{planner}/sub-agents", json={"name": "executor"})
    client.post("/api/graph/entry", json={"agent_id": planner})

    workflow = client.post("/api/workflows", json={"name": "demo workflow"}).json()
    first_run = client.post(
        "/api/runs",
        json={"workflow_id": workflow["id"], "input": {"input": "安排一次会议任务"}},
    ).json()
    first_run = client.get(f"/api/runs/{first_run['id']}").json()

    candidate = client.post(
        "/api/skills/candidates/from-run",
        json={
            "run_id": first_run["id"],
            "name": "会议任务处理技能",
            "tags": ["会议", "任务"],
        },
    ).json()
    client.post(f"/api/skills/{candidate['id']}/validate")
    client.post(f"/api/skills/{candidate['id']}/publish", json={"approved_by": "demo"})

    second_run = client.post(
        "/api/runs",
        json={"workflow_id": workflow["id"], "input": {"input": "继续安排会议任务"}},
    ).json()
    second_run = client.get(f"/api/runs/{second_run['id']}").json()
    traces = client.get(f"/api/runs/{second_run['id']}/skill-traces").json()["events"]

    print("workflow:", workflow["id"])
    print("first_run:", first_run["status"], first_run["id"])
    print("skill:", candidate["id"])
    print("second_run:", second_run["status"], second_run["id"])
    print("skill_trace_events:", len(traces))


if __name__ == "__main__":
    main()
