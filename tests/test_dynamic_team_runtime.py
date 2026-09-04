"""Executable contract for canvas-authored dynamic agent teams."""

from __future__ import annotations

import asyncio
import time

from fastapi.testclient import TestClient
from engine.modules.product_ops import ApplicationStore
from engine.modules.workflows import RunStore, WorkflowStore
from engine.modules.product_ops import ToolCatalogStore
from engine.modules.workflow_runtime import WorkflowNodeRuntimeFactory
from engine.orchestrator import AgentSpec, Orchestrator
from engine.server.app import create_app


def _team_graph(*, failing: bool = False):
    worker = {
        "id": "worker-search",
        "name": "search-worker",
        "description": "search research facts",
        "children": [],
        "config": {"node_kind": "script", "language": "python", "inputs": [{"name": "input", "path": "input"}], "code": "def main(params):\n    return {'answer': 'search done: ' + str(params['input'])}", "output_field": "answer"},
    }
    if failing:
        worker["config"]["code"] = "def main(params):\n    raise RuntimeError('search unavailable')"
    return {
        "entry": "start",
        "agents": [
            {"id": "start", "name": "start", "children": [], "config": {"node_kind": "start"}},
            {"id": "team", "name": "research-team", "children": ["worker-search", "worker-backup"], "config": {"node_kind": "agent_team", "delegation": {"mode": "single", "selection_top_k": 1, "max_parallel": 1}, "aggregation": {"mode": "concat"}, "recovery": {"allow_substitution": True}, "member_profiles": {"worker-search": {"priority": 10, "capabilities": ["search", "research"]}, "worker-backup": {"priority": 0, "capabilities": ["backup", "search"]}}}},
            worker,
            {"id": "worker-backup", "name": "backup-worker", "description": "backup search facts", "children": [], "config": {"node_kind": "script", "language": "python", "inputs": [{"name": "input", "path": "input"}], "code": "def main(params):\n    return {'answer': 'backup done: ' + str(params['input'])}", "output_field": "answer"}},
            {"id": "end", "name": "end", "children": [], "config": {"node_kind": "end", "output_field": "input"}},
        ],
        "connections": [{"source": "start", "target": "team"}, {"source": "team", "target": "end"}],
    }


def _run(graph):
    orch = Orchestrator.from_dict(graph)
    factory = WorkflowNodeRuntimeFactory(ToolCatalogStore(".test-dynamic-team-tools"), graph=graph)
    compiled = orch.build_graph(node_factory=factory)

    async def execute():
        events = []
        async for event in compiled.astream({"input": "search research"}, 30):
            events.append(event)
        return events

    return asyncio.run(execute())


def test_dynamic_team_selects_member_and_aggregates_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    events = _run(_team_graph())
    team_end = next(event for event in events if event.get("type") == "node_end" and event.get("node") == "research-team")
    update = team_end["update"]
    assert "search done: search research" in update["input"]
    assert update["team_results"][0]["member"] == "search-worker"
    assert any(event["type"] == "team_aggregated" for event in update["__runtime_team_events__"])


def test_dynamic_team_substitutes_backup_after_member_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    events = _run(_team_graph(failing=True))
    team_end = next(event for event in events if event.get("type") == "node_end" and event.get("node") == "research-team")
    update = team_end["update"]
    assert "backup done: search research" in update["input"]
    event_types = [event["type"] for event in update["__runtime_team_events__"]]
    assert "team_member_failed" in event_types
    assert "fallback_selected" in event_types


def test_dynamic_team_runs_through_saved_workflow_api(tmp_path, monkeypatch):
    """The persisted canvas graph produces delegation events in a real Run."""
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        tool_catalog_store=ToolCatalogStore(tmp_path / "tools"),
        application_store=ApplicationStore(tmp_path / "apps"),
    )
    client = TestClient(app)
    created = client.post("/api/apps", json={"name": "dynamic api", "app_type": "workflow", "model": ""}).json()
    workflow = client.get(f"/api/workflows/{created['workflow_id']}").json()
    start, end = workflow["graph"]["agents"]
    graph = _team_graph()
    graph["agents"][0] = {**start, "config": {"node_kind": "start"}}
    graph["agents"][-1] = {**end, "config": {"node_kind": "end", "output_field": "input"}}
    graph["entry"] = start["id"]
    graph["connections"] = [
        {"source": start["id"], "target": "team", "conditional": False},
        {"source": "team", "target": end["id"], "conditional": False},
    ]
    saved = client.put(f"/api/workflows/{created['workflow_id']}", json={"graph": graph})
    assert saved.status_code == 200, saved.text
    run = client.post(f"/api/apps/{created['id']}/runs", json={"input": {"input": "search research"}}).json()
    for _ in range(60):
        run = client.get(f"/api/runs/{run['id']}").json()
        if run["status"] not in {"queued", "running"}:
            break
        time.sleep(0.05)
    assert run["status"] == "succeeded", run.get("error")
    kinds = [event["type"] for event in run["events"]]
    assert "delegation_selected" in kinds
    assert "team_aggregated" in kinds
