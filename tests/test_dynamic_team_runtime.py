"""Executable contract for canvas-authored dynamic agent teams."""

from __future__ import annotations

import asyncio
import time

from fastapi.testclient import TestClient
from engine.modules.product_ops import ApplicationStore
from engine.modules.workflows import RunStore, WorkflowStore, WorkflowRecord
from engine.modules.product_ops import ToolCatalogStore
from engine.modules.execution import InferenceResult
from engine.modules.model_connections import ModelConnectionStore
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


def _run(graph, task="search research"):
    orch = Orchestrator.from_dict(graph)
    factory = WorkflowNodeRuntimeFactory(ToolCatalogStore(".test-dynamic-team-tools"), graph=graph)
    compiled = orch.build_graph(node_factory=factory)

    async def execute():
        events = []
        async for event in compiled.astream({"input": task}, 30):
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


def test_dynamic_team_hard_filters_missing_resources_and_audits_score(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    graph = _team_graph()
    team = next(item for item in graph["agents"] if item["id"] == "team")
    team["config"]["member_profiles"]["worker-search"]["requirements"] = {"tool_ids": ["tool-not-installed"]}
    events = _run(graph)
    update = next(event for event in events if event.get("type") == "node_end" and event.get("node") == "research-team")["update"]
    assert update["team_results"][0]["member"] == "backup-worker"
    filtered = next(event for event in update["__runtime_team_events__"] if event["type"] == "candidate_filtered")
    rejected = next(item for item in filtered["candidates"] if item["member_id"] == "worker-search")
    assert rejected["eligible"] is False
    assert "必要工具不可用" in rejected["excluded_reasons"]
    assert "semantic_similarity" in rejected["score_breakdown"]


def test_current_todo_required_tool_hard_filters_team_candidates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tools = ToolCatalogStore(tmp_path / "tools")
    search_tool = tools.create(name="todo_search", display_name="TODO search", metadata={"adapter": "calculator"})
    graph = _team_graph()
    next(item for item in graph["agents"] if item["id"] == "worker-search")["config"]["tool_ids"] = [search_tool.id]
    team_item = next(item for item in graph["agents"] if item["id"] == "team")
    team_item["config"]["member_profiles"]["worker-backup"]["priority"] = 100
    factory = WorkflowNodeRuntimeFactory(tools, graph=graph)
    team = factory(AgentSpec(id="team", name="research-team", children=team_item["children"], config=team_item["config"]))
    update = asyncio.run(team.invoke({"input": "research", "current_todo": {"required_tools": [search_tool.id]}}))
    assert update["team_results"][0]["member_id"] == "worker-search"
    filtered = next(event for event in update["__runtime_team_events__"] if event["type"] == "candidate_filtered")
    backup = next(item for item in filtered["candidates"] if item["member_id"] == "worker-backup")
    assert backup["eligible"] is False
    assert "未挂载当前 TODO 所需工具" in backup["excluded_reasons"]


def test_dynamic_team_supervisor_approves_structured_handoff(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    graph = _team_graph()
    search = next(item for item in graph["agents"] if item["id"] == "worker-search")
    search["config"]["code"] = "def main(params):\n    return {'answer': 'need specialist', 'handoff_request': {'status': 'needs_handoff', 'remaining_task': 'resolve backup search facts', 'required_capabilities': ['backup'], 'reason': 'specialist needed'}}"
    team = next(item for item in graph["agents"] if item["id"] == "team")
    team["config"]["delegation"] = {"mode": "single", "selection_top_k": 1, "max_parallel": 1, "allow_handoff": True, "max_handoffs": 2}
    events = _run(graph)
    update = next(event for event in events if event.get("type") == "node_end" and event.get("node") == "research-team")["update"]
    handoff = next(event for event in update["__runtime_team_events__"] if event["type"] == "handoff_selected")
    assert handoff["source"] == "search-worker"
    assert handoff["target"] == "backup-worker"
    assert any(item["member"] == "backup-worker" for item in update["team_results"])


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


def _sequential_planner_graph():
    return {
        "entry": "start",
        "agents": [
            {"id": "start", "name": "start", "children": [], "config": {"node_kind": "start"}},
            {"id": "planner", "name": "planner", "children": [], "config": {"node_kind": "task_planner", "input_field": "input", "max_subtasks": 5, "max_replans": 1, "route_key": "route", "execute_route": "execute", "done_route": "done", "failed_route": "failed"}},
            {"id": "team", "name": "executor-pool", "children": ["worker"], "config": {"node_kind": "agent_team", "delegation": {"mode": "single", "selection_top_k": 1, "max_parallel": 1}, "member_profiles": {"worker": {"capabilities": ["research", "writing"]}}}},
            {"id": "worker", "name": "worker", "description": "research writing", "children": [], "config": {"node_kind": "script", "inputs": [{"name": "input", "path": "input"}], "code": "def main(params):\n    return {'answer': 'DONE: ' + str(params['input'])}", "output_field": "answer"}},
            {"id": "gate", "name": "quality-gate", "children": [], "config": {"node_kind": "quality_gate", "preset": "general", "route_key": "route", "continue_route": "continue"}},
            {"id": "aggregator", "name": "aggregator", "children": [], "config": {"node_kind": "result_aggregator", "mode": "ordered", "input_field": "plan_results", "output_field": "final"}},
            {"id": "end", "name": "end", "children": [], "config": {"node_kind": "end", "output_field": "input"}},
        ],
        "connections": [
            {"source": "start", "target": "planner"},
            {"source": "planner", "target": "<conditional>", "conditional": True, "condition_key": "route", "path_map": {"execute": "team", "done": "aggregator", "failed": "end"}},
            {"source": "team", "target": "gate"},
            {"source": "gate", "target": "<conditional>", "conditional": True, "condition_key": "route", "path_map": {"continue": "planner"}},
            {"source": "aggregator", "target": "end"},
        ],
    }


def test_planner_team_quality_gate_aggregator_execute_todos_in_order(tmp_path, monkeypatch):
    """The complete dynamic loop advances one TODO at a time and exits only after all pass."""
    monkeypatch.chdir(tmp_path)
    events = _run(_sequential_planner_graph(), "research facts; write report")
    final = next(event["state"] for event in events if event.get("type") == "final")
    assert final["plan_ledger"]["status"] == "completed"
    assert [todo["status"] for todo in final["plan_ledger"]["todos"]] == ["completed", "completed"]
    assert len(final["plan_results"]) == 2
    assert "planner-todo-1" in final["input"]
    assert "planner-todo-2" in final["input"]
    planner_starts = [event for event in events if event.get("type") == "node_start" and event.get("node") == "planner"]
    assert len(planner_starts) == 3


def test_quality_gate_requires_runtime_evidence_for_factual_work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    factory = WorkflowNodeRuntimeFactory(ToolCatalogStore(tmp_path / "tools"))
    gate = factory(AgentSpec(id="gate", name="gate", config={"node_kind": "quality_gate", "preset": "factual", "route_key": "route"}))
    state = {
        "input": "正文声称有来源但没有检索证据",
        "current_todo": {"id": "todo-1", "attempts": 1, "max_attempts": 2, "evidence_requirements": {"citations_required": True, "minimum_sources": 1}},
        "plan_ledger": {"replan_count": 0},
    }
    update = asyncio.run(gate.invoke(state))
    assert update["route"] == "continue"
    assert update["quality_report"]["decision"] == "revise"
    assert update["quality_report"]["checks"]["source_validity"] == 0
    assert any(issue["type"] == "missing_evidence" for issue in update["quality_report"]["issues"])


def test_quality_gate_rejects_use_outside_a_planned_todo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    factory = WorkflowNodeRuntimeFactory(ToolCatalogStore(tmp_path / "tools"))
    gate = factory(AgentSpec(id="gate", name="gate", config={"node_kind": "quality_gate"}))
    try:
        asyncio.run(gate.invoke({"input": "orphan output"}))
    except RuntimeError as exc:
        assert "当前 TODO" in str(exc)
    else:
        raise AssertionError("质量门不应脱离任务规划器运行")


def test_planner_replaces_unfinished_todos_after_quality_replan(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    factory = WorkflowNodeRuntimeFactory(ToolCatalogStore(tmp_path / "tools"))
    responses = iter([
        '{"todos":[{"objective":"collect weak data"},{"objective":"write draft"}]}',
        '{"todos":[{"objective":"collect authoritative evidence","evidence_requirements":{"citations_required":true,"minimum_sources":2}}]}',
    ])
    monkeypatch.setattr(factory.agent_runtime, "_run_pinned_model", lambda *_: InferenceResult(text=next(responses), executor="test", endpoint="local"))
    planner = factory(AgentSpec(id="planner", name="planner", config={"node_kind": "task_planner", "model": "test-model", "route_key": "route", "max_replans": 2}))
    first = asyncio.run(planner.invoke({"input": "produce a sourced report"}))
    second_state = {**first, "input": "unsupported answer", "quality_report": {"decision": "replan", "revision_instruction": "use authoritative sources"}}
    replanned = asyncio.run(planner.invoke(second_state))
    assert replanned["plan_ledger"]["replan_count"] == 1
    assert replanned["current_todo"]["id"] == "planner-replan-1-todo-1"
    assert replanned["current_todo"]["objective"] == "collect authoritative evidence"
    assert replanned["current_todo"]["evidence_requirements"]["minimum_sources"] == 2


def test_planner_emits_failed_after_replan_budget_is_exhausted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    factory = WorkflowNodeRuntimeFactory(ToolCatalogStore(tmp_path / "tools"))
    planner = factory(AgentSpec(id="planner", name="planner", config={"node_kind": "task_planner", "max_replans": 0, "route_key": "route", "failed_route": "failed"}))
    first = asyncio.run(planner.invoke({"input": "write report"}))
    failed = asyncio.run(planner.invoke({**first, "input": "unsupported answer", "quality_report": {"decision": "replan", "revision_instruction": "evidence missing"}}))
    assert failed["route"] == "failed"
    assert failed["plan_ledger"]["status"] == "failed"
    assert failed["plan_ledger"]["todos"][0]["status"] == "failed"


def test_quality_feedback_cycle_reaches_planner_failed_route(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    graph = _sequential_planner_graph()
    next(item for item in graph["agents"] if item["id"] == "planner")["config"].update({"max_retries_per_todo": 1, "max_replans": 0})
    next(item for item in graph["agents"] if item["id"] == "gate")["config"]["preset"] = "factual"
    events = _run(graph, "prepare a sourced answer")
    final = next(event["state"] for event in events if event.get("type") == "final")
    assert final["plan_ledger"]["status"] == "failed"
    assert final["plan_ledger"]["todos"][0]["status"] == "failed"


def test_dynamic_control_graph_can_be_persisted_with_complete_routes(tmp_path, monkeypatch):
    """The exact canvas protocol for planner and quality feedback survives API validation."""
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    models = ModelConnectionStore(tmp_path / "models")
    model = models.create({"name": "planner-model", "provider": "compatible", "model_id": "planner", "base_url": "https://models.example/v1", "tier": "cloud", "test_status": "succeeded"})
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"), run_store=RunStore(tmp_path / "runs"),
        tool_catalog_store=ToolCatalogStore(tmp_path / "tools"), application_store=ApplicationStore(tmp_path / "apps"),
        model_connection_store=models,
    )
    client = TestClient(app)
    created = client.post("/api/apps", json={"name": "dynamic-control", "app_type": "workflow", "model": ""}).json()
    workflow = client.get(f"/api/workflows/{created['workflow_id']}").json()
    persisted_start, persisted_end = workflow["graph"]["agents"]
    graph = _sequential_planner_graph()
    graph["entry"] = persisted_start["id"]
    graph["agents"][0] = {**persisted_start, "config": {"node_kind": "start"}}
    graph["agents"][-1] = {**persisted_end, "config": {"node_kind": "end", "output_field": "input"}}
    next(item for item in graph["agents"] if item["id"] == "planner")["config"]["model"] = model.id
    for edge in graph["connections"]:
        if edge["source"] == "start": edge["source"] = persisted_start["id"]
        if edge.get("target") == "end": edge["target"] = persisted_end["id"]
        if "path_map" in edge:
            edge["path_map"] = {route: persisted_end["id"] if target == "end" else target for route, target in edge["path_map"].items()}
    # Simulate an older browser bundle: route labels were visible in the
    # canvas, but the payload contained only ordinary direct connections.
    legacy_connections = []
    for edge in graph["connections"]:
        if edge.get("conditional") and edge["source"] in {"planner", "gate"}:
            legacy_connections.extend({"source": edge["source"], "target": target, "conditional": False} for target in edge["path_map"].values())
        else:
            legacy_connections.append(edge)
    graph["connections"] = legacy_connections
    saved = client.put(f"/api/workflows/{created['workflow_id']}", json={"graph": graph})
    assert saved.status_code == 200, saved.text
    stored = saved.json()["graph"]
    assert next(edge for edge in stored["connections"] if edge["source"] == "planner")["path_map"].keys() == {"execute", "done", "failed"}
    assert next(edge for edge in stored["connections"] if edge["source"] == "gate")["path_map"].keys() == {"continue"}


def test_application_update_does_not_block_a_later_canvas_graph_repair(tmp_path, monkeypatch):
    """Application bindings are saved before canvas graph data in the UI."""
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    workflow_store = WorkflowStore(tmp_path / "workflows")
    app = create_app(workflow_store=workflow_store, run_store=RunStore(tmp_path / "runs"), application_store=ApplicationStore(tmp_path / "apps"))
    client = TestClient(app)
    created = client.post("/api/apps", json={"name": "repair-order", "app_type": "workflow", "model": ""}).json()
    existing = workflow_store.get(created["workflow_id"])
    start, end = existing.graph["agents"]
    legacy_graph = {
        "entry": start["id"],
        "agents": [start, {"id": "planner", "name": "old-planner", "children": [], "config": {"node_kind": "task_planner"}}, end],
        "connections": [{"source": start["id"], "target": "planner"}, {"source": "planner", "target": end["id"]}],
    }
    workflow_store.save_draft(WorkflowRecord.from_dict({**existing.to_dict(), "graph": legacy_graph}))
    updated = client.put(f"/api/apps/{created['id']}", json={"model": "", "tool_ids": []})
    assert updated.status_code == 200, updated.text
