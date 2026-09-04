"""Public tool-runtime entry point.

The legacy ``engine.modules.tool_runtime`` module remains as a compatibility
implementation while callers migrate to this package-local API.
"""

from ..tool_runtime import ToolRuntime, ToolRuntimeResult, ensure_builtin_tools

__all__ = ["ToolRuntime", "ToolRuntimeResult", "ensure_builtin_tools"]
