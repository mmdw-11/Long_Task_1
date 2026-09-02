"""End-to-end API coverage for deterministic MCP configuration schemas."""

from __future__ import annotations

from fastapi.testclient import TestClient

import engine.server.app as app_module
from engine.modules.product_ops import ToolCatalogStore, ToolConnectionStore
from engine.modules.workflows import RunStore, WorkflowStore
from engine.server.app import create_app
from engine.modules.tools.connection_schema import MANUAL_BEARER_SCHEMA


def test_probe_and_connect_renders_schema_without_exposing_secret(tmp_path, monkeypatch):
    connection_store = ToolConnectionStore(tmp_path / "connections")
    received: list[str] = []
    monkeypatch.setattr(app_module, "inspect_connection", lambda *args, **kwargs: dict(MANUAL_BEARER_SCHEMA))

    def discovered(url, credential_env="", timeout=8, access_token=""):
        received.append(access_token)
        return [{"name": "lookup", "title": "Lookup", "description": "safe lookup", "inputSchema": {"type": "object"}, "annotations": {"readOnlyHint": True}}]

    monkeypatch.setattr(app_module, "discover_mcp_tools", discovered)
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        tool_catalog_store=ToolCatalogStore(tmp_path / "tools"),
        tool_connection_store=connection_store,
    )
    client = TestClient(app)

    probe = client.post("/api/tool-connections/mcp/probe", json={"url": "https://example.com/mcp"})
    assert probe.status_code == 200
    assert probe.json()["schema"]["auth_type"] == "manual_bearer"
    assert probe.json()["schema"]["fields"][0]["type"] == "secret"

    connected = client.post("/api/tool-connections/mcp", json={
        "url": "https://example.com/mcp", "name": "Example", "configuration": {"bearer_token": "super-secret"},
    })
    assert connected.status_code == 200
    payload = connected.json()
    assert payload["connection"]["tool_count"] == 1
    assert "super-secret" not in str(payload)
    assert received == ["super-secret"]
    assert "super-secret" not in next((tmp_path / "mcp_oauth").glob("*.json")).read_text(encoding="utf-8")


def test_gmail_market_schema_requires_static_oauth_fields(tmp_path):
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=RunStore(tmp_path / "runs"),
        tool_catalog_store=ToolCatalogStore(tmp_path / "tools"),
        tool_connection_store=ToolConnectionStore(tmp_path / "connections"),
    )
    client = TestClient(app)
    market = client.get("/api/marketplace/mcp").json()
    gmail = next(item for item in market if item["slug"] == "gmail")
    assert gmail["requires_configuration"] is True
    assert [field["name"] for field in gmail["connection_schema"]["fields"]] == ["client_id", "client_secret"]
