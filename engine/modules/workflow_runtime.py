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


class WorkflowNodeRuntimeFactory:
    """Dispatch persisted ``config.node_kind`` values to executable nodes."""

    def __init__(self, tools: ToolCatalogStore, models: ModelConnectionStore | None = None) -> None:
        self.tools = tools
        self.tool_runtime = ToolRuntime(tools)
        self.agent_runtime = AgentRuntimeFactory(tool_catalog_store=tools, model_connection_store=models)

    def __call__(self, spec: AgentSpec) -> Node:
        kind = str(spec.config.get("node_kind") or "agent")
        if kind in {"agent", "llm"}:
            return self.agent_runtime(spec)

        async def run(state: Dict[str, Any]) -> Dict[str, Any]:
            config = spec.config
            if kind == "start":
                return {"input": state.get("input", state)}
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
                # 知识库资源目前只完成工作流挂载与持久化。检索适配器接入前，
                # 节点安全透传上游输入并返回空文档列表，避免占位能力阻断发布
                # 或让已发布工作流在运行时因“不支持节点类型”而失败。
                value = state.get("input", state)
                output_key = str(config.get("output_field") or "documents")
                return {
                    output_key: [],
                    "input": value,
                    "__knowledge_binding__": {
                        "knowledge_base_id": str(config.get("knowledge_base_id") or ""),
                        "retrieval_enabled": False,
                    },
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
                count = int(state.get(counter_key) or 0) + 1
                limit = max(1, min(100, int(config.get("max_iterations") or 3)))
                route = str(config.get("continue_route") if count < limit else config.get("done_route"))
                return {counter_key: count, str(config.get("route_key") or f"route_{spec.id}"): route}
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
