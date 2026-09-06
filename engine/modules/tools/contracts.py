"""Protocol-neutral contracts shared by the agent loop and tool runtime."""

from __future__ import annotations

import json
from typing import Any, Dict


def decode_tool_arguments(raw: Any) -> Dict[str, Any]:
    """Parse a model tool-call payload and fail closed unless it is an object."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise ValueError("工具参数必须是 JSON 对象")
    # Some OpenAI-compatible providers wrap the function payload once more as
    # {"arguments": {...}}. Tool schemas describe the inner object, so unwrap
    # that transport artefact before validation and execution.
    if set(raw) == {"arguments"} and isinstance(raw["arguments"], dict):
        return dict(raw["arguments"])
    return raw


def tool_function_schema(tool: Any) -> Dict[str, Any]:
    """Convert a catalog record into the OpenAI-compatible tools schema."""
    metadata = dict(getattr(tool, "metadata", {}) or {})
    parameters = metadata.get("input_schema") or metadata.get("schema") or {
        "type": "object",
        "properties": {"task": {"type": "string", "description": "用户任务"}},
        "required": ["task"],
    }
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        parameters = {"type": "object", "properties": {"task": {"type": "string"}}}
    return {
        "type": "function",
        "function": {
            "name": str(getattr(tool, "name", "tool")),
            "description": str(getattr(tool, "description", "") or getattr(tool, "display_name", "工具")),
            "parameters": parameters,
        },
    }


def mcp_error_message(payload: Any) -> str:
    """Return MCP business-error text embedded in a successful HTTP response."""
    if not isinstance(payload, dict):
        return ""
    result = payload.get("result") if "result" in payload else payload
    if not isinstance(result, dict) or not result.get("isError"):
        return ""
    content = result.get("content") or []
    messages = [str(item.get("text") or "") for item in content if isinstance(item, dict)]
    return "\n".join(part for part in messages if part).strip() or "MCP 工具返回业务错误"
