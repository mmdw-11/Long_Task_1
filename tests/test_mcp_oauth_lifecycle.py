"""Contract tests for the OAuth branch of remote MCP installation."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from engine.modules.tools import mcp_remote
from engine.modules.workflows import RunStore, WorkflowStore
from engine.server.app import create_app


def test_oauth_authorize_then_exchange_stores_token(tmp_path, monkeypatch):
    pytest.importorskip("cryptography")
    store = mcp_remote.MCPAuthorizationStore(tmp_path / "oauth")

    def protected(*args, **kwargs):
        raise mcp_remote.MCPAuthorizationRequired(
            {"resource_metadata": "https://mcp.example/.well-known/oauth-protected-resource"}
        )

    monkeypatch.setattr(mcp_remote, "discover_tools", protected)
    monkeypatch.setattr(
        mcp_remote,
        "_get_json",
        lambda url, timeout: (
            {"authorization_servers": ["https://login.example"]}
            if "protected-resource" in url
            else {
                "authorization_endpoint": "https://login.example/authorize",
                "token_endpoint": "https://login.example/token",
                "registration_endpoint": "https://login.example/register",
            }
        ),
    )
    monkeypatch.setattr(mcp_remote, "_post_json", lambda *args: {"client_id": "client-1"})
    started = mcp_remote.start_authorization(
        endpoint="https://mcp.example/mcp",
        redirect_uri="http://127.0.0.1:8000/api/tool-connections/oauth/callback",
        connection_id="conn-1",
        store=store,
    )
    assert "code_challenge=" in started["authorization_url"]
    resumed = mcp_remote.start_authorization(
        endpoint="https://mcp.example/mcp",
        redirect_uri="http://127.0.0.1:8000/api/tool-connections/oauth/callback",
        connection_id="conn-1",
        store=store,
    )
    assert resumed["resumed"] is True
    assert resumed["authorization_url"] == started["authorization_url"]
    state = started["authorization_url"].split("state=")[1].split("&")[0]
    monkeypatch.setattr(mcp_remote, "_post_form", lambda *args: {"access_token": "secret-token", "expires_in": 3600})

    assert mcp_remote.complete_authorization(state=state, code="code", store=store) == "conn-1"
    assert store.token_for("conn-1") == "secret-token"
    assert "secret-token" not in (tmp_path / "oauth" / "conn-1.json").read_text(encoding="utf-8")


def test_remote_discovery_keeps_tool_shape(monkeypatch):
    monkeypatch.setattr(
        mcp_remote,
        "_post_jsonrpc",
        lambda *args, **kwargs: {"result": {"tools": [{"name": "send_email", "inputSchema": {"type": "object"}}]}},
    )
    tools = mcp_remote.discover_tools("https://mcp.example/mcp")
    assert tools[0]["name"] == "send_email"


def test_oauth_callback_is_public_but_still_requires_valid_state(tmp_path):
    """External providers cannot carry the console's localhost cookie."""
    app = create_app(
        auth_required=True,
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
    )
    response = TestClient(app).get("/api/tool-connections/oauth/callback?code=unused&state=invalid")
    assert response.status_code == 400
    assert "请先登录" not in response.text
