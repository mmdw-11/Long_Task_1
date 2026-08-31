"""Runtime dispatch for application-owned visual workflow nodes."""

from __future__ import annotations

import json
from typing import Any, Dict

from ..node import Node, NodeType
from ..orchestrator import AgentSpec
from .agent_runtime import AgentRuntimeFactory
from .product_ops import ToolCatalogStore
from .model_connections import ModelConnectionStore
from .tool_runtime import ToolRuntime
from .knowledge import KnowledgeStore


class WorkflowNodeRuntimeFactory:
    """Dispatch persisted ``config.node_kind`` values to executable nodes."""

    def __init__(self, tools: ToolCatalogStore, models: ModelConnectionStore | None = None, knowledge_store: KnowledgeStore | None = None) -> None:
        self.tools = tools
        self.tool_runtime = ToolRuntime(tools)
        self.agent_runtime = AgentRuntimeFactory(tool_catalog_store=tools, model_connection_store=models, knowledge_store=knowledge_store)
        self.knowledge_store = knowledge_store

    def __call__(self, spec: AgentSpec) -> Node:
        kind = str(spec.config.get("node_kind") or "agent")
        if kind in {"agent", "llm"}:
            return self.agent_runtime(spec)

        async def run(state: Dict[str, Any]) -> Dict[str, Any]:
            config = spec.config
            if kind == "start":
                return {"input": state.get("input", state)}
            if kind in {"loop_start", "loop_end"}:
                return {}
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
                left = _get(state, str(config.get("field") or "input"))
                right = config.get("value")
                operator = str(config.get("operator") or "contains")
                matched = _compare(left, right, operator)
                key = str(config.get("route_key") or f"route_{spec.id}")
                return {key: str(config.get("true_route") if matched else config.get("false_route"))}
            if kind == "intent":
                text = str(_get(state, str(config.get("field") or "input")) or "").lower()
                route = str(config.get("default_route") or "default")
                for item in config.get("intents") or []:
                    if any(str(word).lower() in text for word in item.get("keywords") or []):
                        route = str(item.get("route") or route)
                        break
                return {str(config.get("route_key") or f"route_{spec.id}"): route}
            if kind == "script":
                template = str(config.get("template") or "{{input}}")
                rendered = template.replace("{{input}}", str(state.get("input", "")))
                key = str(config.get("output_field") or "script_output")
                return {key: rendered, "input": rendered, str(config.get("route_key") or f"route_{spec.id}"): str(config.get("route") or "next")}
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
                    items = _get(state, str(config.get("items_path") or "input.items"))
                    values = items if isinstance(items, list) else []
                    should_continue = not terminated and count < min(limit, len(values))
                    if should_continue:
                        updates[str(config.get("item_field") or "loop_item")] = values[count]
                        updates[str(config.get("index_field") or "loop_index")] = count
                else:
                    should_continue = not terminated and count < limit
                if should_continue:
                    updates[counter_key] = count + 1
                    updates[route_key] = continue_route
                else:
                    updates[counter_key] = count
                    updates[route_key] = done_route
                    updates[str(config.get("output_field") or "loop_output")] = state.get("input")
                return updates
            if kind == "batch":
                items = _get(state, str(config.get("items_path") or "input.items"))
                values = items if isinstance(items, list) else []
                key = str(config.get("output_field") or "batch_output")
                return {key: values, "input": values, str(config.get("route_key") or f"route_{spec.id}"): str(config.get("done_route") or "done")}
            raise RuntimeError(f"不支持的工作流节点类型：{kind}")

        return Node(spec.name, run, NodeType.FUNCTION, {"id": spec.id, "node_kind": kind, **spec.config})


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
    return str(right or "") in str(left or "")
