"""Tool execution domain: schemas, arguments, protocol errors and runtime bridges.

New tool-facing code should import from this package instead of adding more
top-level modules under :mod:`engine.modules`.
"""

from .contracts import decode_tool_arguments, mcp_error_message, tool_function_schema
from .runtime import ToolRuntime, ToolRuntimeResult, ensure_builtin_tools

__all__ = [
    "decode_tool_arguments", "mcp_error_message", "tool_function_schema",
    "ToolRuntime", "ToolRuntimeResult", "ensure_builtin_tools",
]
