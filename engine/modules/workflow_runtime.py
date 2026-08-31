"""Runtime dispatch for application-owned visual workflow nodes."""

from __future__ import annotations

import json
import asyncio
from typing import Any, Dict

from ..node import Node, NodeType
from ..orchestrator import AgentSpec, Orchestrator
from .agent_runtime import AgentRuntimeFactory
from .product_ops import ToolCatalogStore
from .model_connections import ModelConnectionStore
from .tool_runtime import ToolRuntime
from .knowledge import KnowledgeStore
from .workflow_scripts import run_workflow_script


class WorkflowNodeRuntimeFactory:
    """Dispatch persisted ``config.node_kind`` values to executable nodes."""

    def __init__(
        self,
        tools: ToolCatalogStore,
        models: ModelConnectionStore | None = None,
        graph: Dict[str, Any] | None = None,
        knowledge_store: KnowledgeStore | None = None,
    ) -> None:
        self.tools = tools
        self.tool_runtime = ToolRuntime(tools)
        self.agent_runtime = AgentRuntimeFactory(tool_catalog_store=tools, model_connection_store=models, knowledge_store=knowledge_store)
        self.knowledge_store = knowledge_store
        self.models = models
        self.graph = graph or {}

    def __call__(self, spec: AgentSpec) -> Node:
        kind = str(spec.config.get("node_kind") or "agent")
        if kind in {"agent", "llm"}:
            return self.agent_runtime(spec)

        async def run(state: Dict[str, Any]) -> Dict[str, Any]:
            config = spec.config
            if kind == "start":
                return {"input": state.get("input", state)}
            if kind in {"loop_start", "batch_start"}:
                return {}
            if kind in {"loop_end", "batch_end"}:
                parent = str(config.get("parent_id") or "")
                if not parent: return {}
                result_key = f"__{'loop' if kind == 'loop_end' else 'batch'}_results_{parent}"
                return {result_key: [*(state.get(result_key) or []), state.get("input")]}
            if kind == "end":
                value = _get(state, str(config.get("output_field") or "input"))
                return {"input": value, "workflow_output": value, spec.name: value}
            if kind == "tool":
                tool_id = str(config.get("tool_id") or "")
                tool = self.tools.get(tool_id)
                task = _get(state, str(config.get("input_field") or "input"))
                task_text = task if isinstance(task, str) else json.dumps(task, ensure_ascii=False, default=str)
                result = self.tool_runtime.execute(tool, task_text).to_dict()
                if result["status"] == "failed":
                    raise RuntimeError(result.get("error") or "工具调用失败")
                output_key = str(config.get("output_field") or "tool_output")
                return {output_key: result.get("result"), "input": result.get("result"), "__runtime_tool_calls__": [result]}
            if kind == "knowledge":
                kb_ids = list(config.get("knowledge_base_ids") or [])
                if config.get("knowledge_base_id"):
                    kb_ids.append(str(config["knowledge_base_id"]))
                if not self.knowledge_store or not kb_ids:
                    raise RuntimeError("知识库节点未配置可用知识库")
                query = _get(state, str(config.get("input_field") or "input"))
                owner = str(state.get("__owner_user_id__") or "local-user")
                result=self.knowledge_store.retrieve(kb_ids, str(query or ""), owner=owner, mode=str(config.get("mode") or "hybrid"), top_k=int(config.get("top_k") or 5), threshold=float(config.get("threshold") or .15), labels=list(config.get("labels") or []), workflow_id=str(state.get("__workflow_id__") or ""), run_id=str(state.get("__run_id__") or ""))
                output_key = str(config.get("output_field") or "documents")
                return {
                    output_key: result["documents"], "documents":result["documents"], "context":result["context"], "citations":result["citations"], "retrieval_metadata":result["retrieval_metadata"], "input": result["context"],
                }
            if kind == "condition":
                branches = config.get("branches") or _legacy_condition_branches(config)
                route = str(config.get("default_route") or "default")
                for branch in branches:
                    groups = branch.get("groups") or []
                    if any(all(_match_condition(state, item) for item in group.get("conditions") or []) for group in groups):
                        route = str(branch.get("route") or route); break
                key = str(config.get("route_key") or f"route_{spec.id}")
                return {key: route}
            if kind == "intent":
                categories = config.get("categories") or _legacy_categories(config)
                model = str(config.get("model") or spec.model or "")
                if not model:
                    raise RuntimeError("意图分类节点尚未选择模型")
                text = str(_get(state, str(config.get("input_field") or config.get("field") or "input")) or "")
                names = [str(item.get("name") or item.get("route")) for item in categories]
                prompt = f"输入：{text}\n候选意图：{json.dumps(categories, ensure_ascii=False)}\n{config.get('prompt') or ''}\n只返回 JSON：{{\"categories\":[\"意图名\"]}}"
                result = self.agent_runtime._run_pinned_model(model, prompt, "你是严格的意图分类器。只能从候选意图中选择。")
                if not result.success: raise RuntimeError(result.error or "意图分类失败")
                try: selected = json.loads(result.text).get("categories") or []
                except (json.JSONDecodeError, AttributeError): selected = []
                allowed = [name for name in selected if name in names]
                if str(config.get("mode") or "single") == "single": allowed = allowed[:1]
                routes = [str(next(item.get("route") for item in categories if str(item.get("name") or item.get("route")) == name)) for name in allowed]
                fallback = str(config.get("default_route") or "default")
                value: Any = routes if str(config.get("mode") or "single") == "multiple" else (routes[0] if routes else fallback)
                return {str(config.get("route_key") or f"route_{spec.id}"): value, str(config.get("output_field") or "intent_result"): allowed}
            if kind == "script":
                if config.get("code"):
                    params = {str(item.get("name")): _get(state, str(item.get("path") or item.get("name"))) for item in config.get("inputs") or []}
                    if not params: params = {"input": state.get("input")}
                    attempts, last = max(1, min(5, int(config.get("retry_count") or 0) + 1)), None
                    for attempt in range(attempts):
                        try:
                            result = await asyncio.to_thread(run_workflow_script, str(config.get("language") or "python"), str(config.get("code")), params, float(config.get("timeout_seconds") or 5)); break
                        except Exception as exc:
                            last = exc
                            if attempt + 1 < attempts: await asyncio.sleep(max(0, min(10, float(config.get("retry_interval_ms") or 0) / 1000)))
                    else:
                        if str(config.get("error_strategy") or "raise") == "default": result = config.get("default_output") or {}
                        else: raise RuntimeError(str(last)) from last
                else:
                    rendered = str(config.get("template") or "{{input}}").replace("{{input}}", str(state.get("input", "")))
                    result = {str(config.get("output_field") or "script_output"): rendered}
                return {**result, "input": result, str(config.get("route_key") or f"route_{spec.id}"): str(config.get("route") or "next")}
            if kind == "assign":
                updates: Dict[str, Any] = {}
                for item in config.get("assignments") or []:
                    target, operation = str(item.get("target") or ""), str(item.get("operation") or "set")
                    if not target: continue
                    value = _get(state, str(item.get("source"))) if item.get("source") else item.get("value")
                    current = _get(state, target)
                    updates[target] = _assign(current, value, operation)
                return updates
            if kind == "loop":
                counter_key = f"__loop_{spec.id}"
                count = int(state.get(counter_key) or 0)
                limit = max(1, min(100, int(config.get("max_iterations") or 3)))
                route_key = str(config.get("route_key") or f"route_{spec.id}")
                continue_route = str(config.get("continue_route") or "continue")
                done_route = str(config.get("done_route") or "done")
                termination_field = str(config.get("termination_field") or "").strip()
                terminated = bool(termination_field) and _compare(
                    _get(state, termination_field),
                    config.get("termination_value"),
                    str(config.get("termination_operator") or "equals"),
                )
                updates: Dict[str, Any] = {}
                if str(config.get("loop_type") or "count") == "array":
                    input_arrays = config.get("input_arrays") or [{"path": config.get("items_path") or "input.items", "item_field": config.get("item_field") or "loop_item"}]
                    arrays = [value if isinstance(value := _get(state, str(item.get("path") or "")), list) else [] for item in input_arrays]
                    length = min((len(value) for value in arrays), default=0)
                    should_continue = not terminated and count < min(limit, length)
                    if should_continue:
                        for item, values in zip(input_arrays, arrays): updates[str(item.get("item_field") or "loop_item")] = values[count]
                        updates[str(config.get("index_field") or "loop_index")] = count
                else:
                    should_continue = not terminated and count < limit
                if should_continue:
                    updates[counter_key] = count + 1
                    updates[route_key] = continue_route
                else:
                    updates[counter_key] = count
                    updates[route_key] = done_route
                    updates[str(config.get("output_field") or "loop_output")] = state.get(f"__loop_results_{spec.id}", [])
                return updates
            if kind == "batch":
                input_arrays = config.get("input_arrays") or [{"path": config.get("items_path") or "input.items", "item_field": config.get("item_field") or "batch_item"}]
                arrays = [value if isinstance(value := _get(state, str(item.get("path") or "")), list) else [] for item in input_arrays]
                limit = min([len(value) for value in arrays] + [max(1, min(100, int(config.get("max_items") or 100)))])
                if spec.children and self.graph:
                    semaphore = asyncio.Semaphore(max(1, min(10, int(config.get("concurrency") or 3))))
                    async def execute(index: int) -> Dict[str, Any]:
                        async with semaphore:
                            item_state = dict(state)
                            for offset, item in enumerate(input_arrays): item_state[str(item.get("item_field") or f"item_{offset}")] = arrays[offset][index]
                            item_state[str(config.get("index_field") or "batch_index")] = index
                            try:
                                output, events = await self._run_child_graph(spec, item_state)
                                return {"index": index, "status": "succeeded", "output": output.get("input", output), "events": events}
                            except Exception as exc:
                                return {"index": index, "status": "failed", "error": str(exc), "events": []}
                    results = await asyncio.gather(*(execute(index) for index in range(limit)))
                    results.sort(key=lambda item: item["index"])
                    key = str(config.get("output_field") or "batch_output")
                    return {key: results, "input": results, str(config.get("route_key") or f"route_{spec.id}"): str(config.get("done_route") or "done"), "__runtime_child_events__": [event for item in results for event in item["events"]]}
                counter_key, count = f"__batch_{spec.id}", int(state.get(f"__batch_{spec.id}") or 0)
                route_key, continue_route, done_route = str(config.get("route_key") or f"route_{spec.id}"), str(config.get("continue_route") or "continue"), str(config.get("done_route") or "done")
                key = str(config.get("output_field") or "batch_output")
                if count < limit:
                    values = {str(item.get("item_field") or f"item_{offset}"): arrays[offset][count] for offset, item in enumerate(input_arrays)}
                    return {**values, str(config.get("index_field") or "batch_index"): count, counter_key: count + 1, route_key: continue_route}
                results = state.get(f"__batch_results_{spec.id}", [])
                return {key: results, "input": results, counter_key: count, route_key: done_route}
            raise RuntimeError(f"不支持的工作流节点类型：{kind}")

        return Node(spec.name, run, NodeType.FUNCTION, {"id": spec.id, "node_kind": kind, **spec.config})

    async def _run_child_graph(self, parent: AgentSpec, state: Dict[str, Any]) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
        child_ids = set(parent.children)
        agents = [item for item in self.graph.get("agents", []) if item.get("id") in child_ids]
        connections = [item for item in self.graph.get("connections", []) if item.get("source") in child_ids and (item.get("target") in child_ids or item.get("target") == "END")]
        start = next((item for item in agents if str((item.get("config") or {}).get("node_kind")) == "batch_start"), None)
        if not start: raise RuntimeError("批处理子流程缺少批处理开始节点")
        child_graph = {"entry": start["id"], "agents": agents, "connections": connections}
        orchestrator = Orchestrator.from_dict(child_graph)
        factory = WorkflowNodeRuntimeFactory(self.tools, self.models, graph=child_graph, knowledge_store=self.knowledge_store)
        compiled = orchestrator.build_graph(node_factory=factory, recursion_limit=50)
        events, output = [], state
        async for event in compiled.astream(state, 50):
            enriched = {**event, "parent_node_id": parent.id, "batch_index": state.get(str(parent.config.get("index_field") or "batch_index"))}
            events.append(enriched)
            if event.get("type") == "final": output = event.get("state") or output
        return output, events


def _get(state: Dict[str, Any], path: str) -> Any:
    current: Any = state
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _compare(left: Any, right: Any, operator: str) -> bool:
    if operator == "equals":
        return left == right
    if operator == "not_equals":
        return left != right
    if operator == "exists":
        return left is not None
    if operator == "greater_than":
        try:
            return float(left) > float(right)
        except (TypeError, ValueError):
            return False
    if operator == "less_than":
        try: return float(left) < float(right)
        except (TypeError, ValueError): return False
    if operator == "not_contains": return str(right or "") not in str(left or "")
    if operator == "is_empty": return left is None or left == "" or left == [] or left == {}
    if operator == "not_empty": return not _compare(left, None, "is_empty")
    return str(right or "") in str(left or "")


def _match_condition(state: Dict[str, Any], item: Dict[str, Any]) -> bool:
    return _compare(_get(state, str(item.get("field") or "input")), item.get("value"), str(item.get("operator") or "equals"))


def _legacy_condition_branches(config: Dict[str, Any]) -> list[Dict[str, Any]]:
    return [{"route": str(config.get("true_route") or "true"), "groups": [{"conditions": [{"field": config.get("field") or "input", "operator": config.get("operator") or "contains", "value": config.get("value")}]}]}]


def _legacy_categories(config: Dict[str, Any]) -> list[Dict[str, Any]]:
    return [{"name": str(item.get("name") or item.get("route") or "意图"), "description": "；".join(str(x) for x in item.get("keywords") or []), "route": str(item.get("route") or "matched")} for item in config.get("intents") or []]


def _assign(current: Any, value: Any, operation: str) -> Any:
    if operation == "append": return (current if isinstance(current, list) else []) + [value]
    if operation == "extend": return (current if isinstance(current, list) else []) + (value if isinstance(value, list) else [value])
    if operation == "add": return float(current or 0) + float(value or 0)
    if operation == "merge": return {**(current if isinstance(current, dict) else {}), **(value if isinstance(value, dict) else {})}
    return value
