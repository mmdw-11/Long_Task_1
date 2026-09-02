"""Tool execution domain: schemas, arguments, protocol errors and runtime bridges.

New tool-facing code should import from this package instead of adding more
top-level modules under :mod:`engine.modules`.
"""

from .contracts import decode_tool_arguments, mcp_error_message, tool_function_schema
from .mcp_remote import MCPAuthorizationRequired, MCPAuthorizationStore, call_tool, discover_tools, inspect_connection, start_authorization, complete_authorization
from .connection_schema import GMAIL_STATIC_OAUTH_SCHEMA, MANUAL_BEARER_SCHEMA, connection_schema, missing_required

__all__ = [
    "decode_tool_arguments", "mcp_error_message", "tool_function_schema",
    "ToolRuntime", "ToolRuntimeResult", "ensure_builtin_tools",
    "MCPAuthorizationRequired", "MCPAuthorizationStore", "call_tool", "discover_tools", "inspect_connection", "start_authorization", "complete_authorization",
    "GMAIL_STATIC_OAUTH_SCHEMA", "MANUAL_BEARER_SCHEMA", "connection_schema", "missing_required",
]


def __getattr__(name: str):
    """Load the legacy runtime lazily to keep the package import acyclic."""
    if name in {"ToolRuntime", "ToolRuntimeResult", "ensure_builtin_tools"}:
        from .runtime import ToolRuntime, ToolRuntimeResult, ensure_builtin_tools
        return {"ToolRuntime": ToolRuntime, "ToolRuntimeResult": ToolRuntimeResult, "ensure_builtin_tools": ensure_builtin_tools}[name]
    raise AttributeError(name)
