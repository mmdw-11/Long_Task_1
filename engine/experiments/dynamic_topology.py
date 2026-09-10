"""Runnable, auditable experiment harness for task-local dynamic topology.

The task set intentionally contains no answer keys or target-agent names in
the model prompt.  Its oracle checks only externally observable contracts:
required resources, event type, selected-member capability and an actual
model/tool invocation.  Controlled Handoff/Fallback events are explicitly
marked as injected; they test the supervisor's re-selection path, not an
unsubstantiated claim that an LLM independently chose to hand off.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..modules.model_connections import ModelConnectionStore
from ..modules.product_ops import ToolCatalogStore
from ..modules.workflow_runtime import WorkflowNodeRuntimeFactory
from ..orchestrator import AgentSpec
from .io import read_jsonl, write_json, write_jsonl


FAMILIES = ("resource_match", "handoff", "fallback", "multi_todo")


def build_dynamic_topology_dataset(output: str | Path) -> List[Dict[str, Any]]:
    """Build forty frozen task contracts (ten balanced examples per family).

    Prompts vary entity, wording and arithmetic requirement, but no generated
    record contains a gold natural-language answer.  Live API values are
    deliberately resolved only at run time.
    """
    metadata_pairs = [
        ("默认分支", "公开问题数"), ("主语言", "许可证"), ("最近更新时间", "可见性"),
        ("仓库全名", "默认分支"), ("是否为 fork", "公开问题数"), ("描述", "主语言"),
        ("许可证", "最近推送时间"), ("默认分支", "是否归档"), ("开放问题数", "可见性"),
        ("仓库全名", "主语言"),
    ]
    rows: List[Dict[str, Any]] = []
    for index, fields in enumerate(metadata_pairs, 1):
        base = {
            "source": "dynamic_topology_mechanism_suite_v1",
            "external_source": "https://api.github.com/repos/python/cpython",
            "repository": "python/cpython",
            "requested_fields": list(fields),
            "acceptance": {"require_real_llm": True, "require_successful_tool": True, "require_nonempty_output": True},
        }
        if index % 2:
            s1_task = f"查询 GitHub 公开仓库 python/cpython 的元数据。必须调用已挂载的 GitHub API，并报告“{fields[0]}”和“{fields[1]}”；不要猜测或使用训练记忆。"
            s1_todo = {"required_capabilities": ["web_time_retrieval"], "required_tool_role": "time_api"}
            s1_capability = "web_time_retrieval"
        else:
            left, right = index + 11, index + 7
            s1_task = f"独立核验算式 {left} × {right}。必须调用已挂载的计算器并仅报告计算结果，不要凭心算作答。"
            s1_todo = {"required_capabilities": ["numeric_verification"], "required_tool_role": "calculator"}
            s1_capability = "numeric_verification"
        rows.append({
            **base, "id": f"S1-{index:02d}", "family": "resource_match", "task": s1_task,
            "todo": s1_todo,
            "expected": {"selected_capability": s1_capability, "events": ["candidate_filtered", "delegation_selected"]},
        })
        rows.append({
            **base, "id": f"S2-{index:02d}", "family": "handoff",
            "task": f"先获取 GitHub 公开仓库 python/cpython 的“{fields[0]}”。获得 API 证据后，需要由另一位具备数值核验能力的成员计算 {index + 2} × {index + 3}，并把两项结果简洁汇总。",
            "todo": {"required_capabilities": ["web_time_retrieval"], "required_tool_role": "time_api"},
            "event_plan": {"id": f"handoff-{index:02d}", "kind": "handoff", "source_member_id": "researcher", "required_capabilities": ["numeric_verification"], "remaining_task": f"计算 {index + 2} × {index + 3}，并与已获得的时间证据一起汇总。", "reason": "受控能力缺口：当前阶段需要独立数值核验", "require_successful_tool_call": True},
            "expected": {"selected_capability": "web_time_retrieval", "handoff_target_capability": "numeric_verification", "events": ["handoff_selected"]},
        })
        rows.append({
            **base, "id": f"S3-{index:02d}", "family": "fallback",
            "task": f"使用 GitHub 公开 API 查询 python/cpython 的“{fields[1]}”并形成一条带来源链接的事实记录。若当前执行成员不可继续，团队应由可用替代成员完成同一任务。",
            "todo": {"required_capabilities": ["web_time_retrieval"], "required_tool_role": "time_api"},
            "event_plan": {"id": f"fallback-{index:02d}", "kind": "failure_after_tool", "source_member_id": "researcher", "reason": "受控成员故障：已完成真实工具调用后模拟成员不可继续", "require_successful_tool_call": True},
            "expected": {"selected_capability": "web_time_retrieval", "fallback_target_capability": "web_time_retrieval", "events": ["team_member_failed", "fallback_selected"]},
        })
        rows.append({
            **base, "id": f"S4-{index:02d}", "family": "multi_todo",
            "task": f"阶段一：查询 GitHub 公开仓库 python/cpython 的“{fields[0]}”。阶段二：用独立数值核验计算 {index + 4} × {index + 5}。每阶段只报告由相应工具得到的结果。",
            "todos": [
                {"id": "retrieve", "required_capabilities": ["web_time_retrieval"], "required_tool_role": "time_api"},
                {"id": "verify", "required_capabilities": ["numeric_verification"], "required_tool_role": "calculator"},
            ],
            "expected": {"events": ["topology_reconfigured"], "different_active_members": True},
        })
    rows.sort(key=lambda item: item["id"])
    validate_dynamic_topology_dataset(rows)
    write_jsonl(output, rows)
    return rows


def validate_dynamic_topology_dataset(rows: Iterable[Dict[str, Any]]) -> None:
    items = list(rows)
    if len(items) != 40:
        raise ValueError(f"dynamic topology dataset must contain 40 rows, got {len(items)}")
    ids = [str(item.get("id") or "") for item in items]
    if len(set(ids)) != len(ids) or any(not item for item in ids):
        raise ValueError("dataset ids must be unique and non-empty")
    for family in FAMILIES:
        selected = [item for item in items if item.get("family") == family]
        if len(selected) != 10:
            raise ValueError(f"{family} must contain exactly 10 rows")
    # A target agent / gold answer in the task prompt would leak the outcome.
    banned = ("gold_answer", "expected_answer", "recommended_agent", "selected_member")
    for item in items:
        if any(key in item for key in banned):
            raise ValueError(f"{item['id']} leaks an answer or target member")
        if not str(item.get("task") or "").strip():
            raise ValueError(f"{item['id']} has no task text")
        if item.get("family") in {"handoff", "fallback"} and not isinstance(item.get("event_plan"), dict):
            raise ValueError(f"{item['id']} lacks its controlled event plan")


def dataset_manifest(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    payload = list(rows)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"dataset": "dynamic_topology_mechanism_suite_v1", "count": len(payload), "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(), "families": {name: sum(1 for item in payload if item.get("family") == name) for name in FAMILIES}}


def _ensure_tools(catalog: ToolCatalogStore) -> Dict[str, str]:
    existing = {tool.name: tool.id for tool in catalog.list()}
    def create(name: str, **data: Any) -> str:
        if name in existing:
            return existing[name]
        return catalog.create(name=name, **data).id
    return {
        "time_api": create("topology_github_cpython", display_name="GitHub CPython Metadata API", description="读取 GitHub 公开仓库 python/cpython 的当前元数据。", category="external", tags=["github", "repository", "http", "read"], metadata={"adapter": "openapi_http", "risk": "read", "operation_url": "https://api.github.com/repos/python/cpython", "http_method": "GET", "timeout_seconds": 15}),
        "calculator": create("topology_calculator", display_name="Topology Calculator", description="执行数值核验所需的四则运算。", category="utility", tags=["math", "calculate"], metadata={"adapter": "calculator", "risk": "low"}),
    }


def _graph(model_id: str, tool_ids: Dict[str, str], method: str, event_plan: Dict[str, Any] | None) -> Dict[str, Any]:
    recovery = method == "ours"
    delegation: Dict[str, Any] = {"mode": "hybrid_selector", "selection_top_k": 1, "max_parallel": 1, "allow_handoff": recovery, "max_handoffs": 1}
    if method == "fixed_sparse":
        # A fair fixed-sparse baseline has both ordinary capabilities mounted
        # for every TODO.  It can complete simple tasks, but pays for a fixed
        # two-member topology and cannot reconfigure after a runtime event.
        delegation.update({"mode": "fixed_sparse", "fixed_member_ids": ["researcher", "analyst"], "max_parallel": 2, "allow_handoff": False, "max_handoffs": 0})
    if method == "dynamic_selection_only":
        delegation.update({"allow_handoff": False, "max_handoffs": 0})
    team_config: Dict[str, Any] = {
        "node_kind": "agent_team", "input_field": "input", "output_field": "team_result", "delegation": delegation,
        "topology": {"edge_ttl": 1}, "recovery": {"allow_substitution": recovery},
        "member_profiles": {
            "researcher": {"priority": 20, "capabilities": ["web_time_retrieval", "research"], "requirements": {"tool_ids": [tool_ids["time_api"]]}},
            "analyst": {"priority": 10, "capabilities": ["numeric_verification", "calculation"], "requirements": {"tool_ids": [tool_ids["calculator"]]}},
            "fallback": {"priority": 5, "capabilities": ["web_time_retrieval", "recovery"], "requirements": {"tool_ids": [tool_ids["time_api"]]}},
        },
    }
    if event_plan:
        team_config["experimental_event_plan"] = dict(event_plan)
    members = [
        {"id": "researcher", "name": "researcher", "model": model_id, "description": "GitHub repository metadata researcher. Obtain current public facts only through the attached API.", "sys_prompt": "Use the attached real GitHub API when the task asks for repository metadata. State only tool-supported facts and include the source URL.", "children": [], "config": {"node_kind": "agent", "tool_ids": [tool_ids["time_api"]]}},
        {"id": "analyst", "name": "analyst", "model": model_id, "description": "Independent numeric verification specialist.", "sys_prompt": "Use the calculator for every requested arithmetic expression and report its result.", "children": [], "config": {"node_kind": "agent", "tool_ids": [tool_ids["calculator"]]}},
        {"id": "fallback", "name": "fallback", "model": model_id, "description": "Backup GitHub metadata researcher for recovery.", "sys_prompt": "Use the attached real GitHub API and provide a concise sourced result.", "children": [], "config": {"node_kind": "agent", "tool_ids": [tool_ids["time_api"]]}},
        {"id": "team", "name": "topology-team", "children": ["researcher", "analyst", "fallback"], "config": team_config},
    ]
    return {"agents": members}


def _real_llm_and_tool(results: List[Dict[str, Any]]) -> tuple[bool, bool]:
    model_seen = tool_seen = False
    for item in results:
        update = item.get("update") or {}
        for message in update.get("messages") or []:
            meta = ((message.get("result") or {}).get("metadata") or {}) if isinstance(message, dict) else {}
            if not meta.get("simulated") and meta.get("connection_id"):
                model_seen = True
        tool_seen = tool_seen or any(call.get("status") == "succeeded" for call in update.get("__runtime_tool_calls__") or [] if isinstance(call, dict))
    return model_seen, tool_seen


async def run_dynamic_topology_case(case: Dict[str, Any], *, method: str, model_connection: str, root: str | Path) -> Dict[str, Any]:
    """Execute one case against the real model-connection runtime."""
    if method not in {"fixed_sparse", "dynamic_selection_only", "ours"}:
        raise ValueError("method must be fixed_sparse, dynamic_selection_only, or ours")
    root = Path(root)
    catalog = ToolCatalogStore(root / "tools")
    models = ModelConnectionStore(root / "models")
    if not models.get(model_connection).runnable:
        raise RuntimeError(f"model connection {model_connection!r} is not runnable; refusing LocalEcho fallback")
    tool_ids = _ensure_tools(catalog)
    event = dict(case.get("event_plan") or {}) or None
    graph = _graph(model_connection, tool_ids, method, event)
    factory = WorkflowNodeRuntimeFactory(catalog, models=models, graph=graph)
    team_item = next(item for item in graph["agents"] if item["id"] == "team")
    team = factory(AgentSpec(id="team", name="topology-team", children=list(team_item["children"]), config=dict(team_item["config"])))
    todo_list = list(case.get("todos") or [case.get("todo") or {}])
    root_task = str(case["task"])
    state: Dict[str, Any] = {"input": root_task, "experimental_event_plan": event}
    all_events: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for todo in todo_list:
        current_todo = {**dict(todo), "id": str(todo.get("id") or f"todo-{len(all_results)+1}")}
        role = str(current_todo.pop("required_tool_role", "") or "")
        if role:
            current_todo["required_tools"] = [tool_ids[role]]
        state["current_todo"] = current_todo
        # Team aggregation writes its result back to ``input``.  Before a
        # following TODO, restore a stage-specific task instruction just as
        # the production planner does, instead of asking the next specialist
        # to infer its work from the previous member's prose output.
        state["input"] = f"总任务：{root_task}\n\n当前 TODO（{current_todo['id']}）：{json.dumps(current_todo, ensure_ascii=False)}"
        update = await team.invoke(state)
        state.update(update)
        all_events.extend(update.get("__runtime_team_events__") or [])
        all_results.extend(update.get("team_results") or [])
        # A controlled event is single-shot, so a second TODO cannot repeat it.
        state["experimental_event_plan"] = None
    real_llm, real_tool = _real_llm_and_tool(all_results)
    event_types = [str(item.get("type") or "") for item in all_events]
    active = [tuple(item.get("active_members") or []) for item in all_events if item.get("type") == "topology_reconfigured"]
    expected_events = list((case.get("expected") or {}).get("events") or [])
    event_ok = all(name in event_types for name in expected_events)
    expected = dict(case.get("expected") or {})
    capability_members = {
        "web_time_retrieval": {"researcher", "fallback"},
        "numeric_verification": {"analyst"},
    }
    selected = [str(item.get("target_id") or "") for item in all_events if item.get("type") == "delegation_selected"]
    resource_ok = True
    if expected.get("selected_capability"):
        resource_ok = any(member_id in capability_members[str(expected["selected_capability"])] for member_id in selected)
    handoff_ok = True
    if expected.get("handoff_target_capability"):
        handoff_targets = [str(item.get("target_id") or "") for item in all_events if item.get("type") == "handoff_selected"]
        handoff_ok = bool(handoff_targets) and handoff_targets[-1] in capability_members[str(expected["handoff_target_capability"])]
    fallback_ok = True
    if expected.get("fallback_target_capability"):
        fallback_targets = [str(item.get("target_id") or "") for item in all_events if item.get("type") == "fallback_selected"]
        fallback_ok = bool(fallback_targets) and fallback_targets[-1] in capability_members[str(expected["fallback_target_capability"])]
    if case.get("family") == "multi_todo":
        event_ok = event_ok and len(active) >= 2 and len(set(active)) >= 2
    passed = bool(real_llm and real_tool and event_ok and resource_ok and handoff_ok and fallback_ok and all_results and any(item.get("status") == "succeeded" for item in all_results))
    return {
        "id": case["id"], "family": case["family"], "method": method, "passed": passed,
        "metrics": {"real_llm": int(real_llm), "real_tool": int(real_tool), "event_contract": int(event_ok), "resource_contract": int(resource_ok), "handoff_contract": int(handoff_ok), "fallback_contract": int(fallback_ok), "active_members": sum(len(item) for item in active) / max(1, len(active)), "temporary_edges": sum(1 for item in all_events if item.get("type") in {"delegation_selected", "handoff_selected", "fallback_selected"}), "trace_complete": int(bool(all_events and all_results)), "execution_ms": round((time.perf_counter() - started) * 1000, 2)},
        "events": all_events, "results": all_results, "output": state.get("input"),
    }


def load_dynamic_topology_dataset(path: str | Path) -> List[Dict[str, Any]]:
    rows = list(read_jsonl(path)); validate_dynamic_topology_dataset(rows); return rows


def save_dynamic_topology_run(row: Dict[str, Any], output_dir: str | Path) -> None:
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    write_json(root / f"{row['id']}-{row['method']}.json", row)
