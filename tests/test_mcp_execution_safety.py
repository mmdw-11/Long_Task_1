"""Regression tests for real MCP business errors and Bird email safety."""

from __future__ import annotations

import pytest

from engine.modules.mcp_integration import _arguments_from_schema
from engine.modules.product_ops import ToolCatalogStore
from engine.modules.tool_runtime import ToolRuntime
from engine.modules.tools import mcp_remote


def test_mcp_embedded_business_error_is_not_reported_as_success(monkeypatch):
    monkeypatch.setattr(mcp_remote, "_post_jsonrpc", lambda *args, **kwargs: {"result": {"isError": True, "content": [{"type": "text", "text": "invalid arguments"}]}})
    with pytest.raises(RuntimeError, match="invalid arguments"):
        mcp_remote.call_tool("https://mcp.example", "read", {}, token="token")


def test_empty_mcp_schema_does_not_invent_task_argument():
    assert _arguments_from_schema({"type": "object", "properties": {}}, "send an email") == {}


def test_bird_send_is_blocked_before_approval_when_no_verified_domain(tmp_path, monkeypatch):
    catalog = ToolCatalogStore(tmp_path / "tools")
    tool = catalog.create(
        name="bird_email_send", display_name="Send email", category="mcp",
        metadata={"adapter":"mcp_http", "remote_tool_name":"email_send", "mcp_url":"https://mcp.bird.com", "credential_env":"BIRD_TOKEN", "risk":"high"},
    )
    monkeypatch.setenv("BIRD_TOKEN", "token")
    calls: list[str] = []
    def fake_call(url, name, arguments, **kwargs):
        calls.append(name)
        return {"content": [{"type": "text", "text": '{"data":[]}'}]}
    monkeypatch.setattr("engine.modules.tool_runtime.call_remote_mcp_tool", fake_call)

    result = ToolRuntime(catalog).execute(tool, "send", arguments={"from":"noreply@example.com"})
    assert result.status == "blocked"
    assert "没有已验证的发件域" in result.error
    assert calls == ["email_domains_list"]
