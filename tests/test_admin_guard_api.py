"""验证最小管理保护与接口审计。

目标是确保后端在默认开发模式下兼容现有调用；一旦配置管理密钥，所有写接口都
需要显式授权，并且会留下可查询的审计记录。
"""

from fastapi.testclient import TestClient

from engine.modules.product_ops import ToolCatalogStore
from engine.modules.security_ops import ApiAuditStore
from engine.server.app import create_app


def test_admin_guard_is_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    app = create_app(
        tool_catalog_store=ToolCatalogStore(tmp_path / "tools"),
        api_audit_store=ApiAuditStore(tmp_path / "audit" / "api.jsonl"),
    )
    client = TestClient(app)

    response = client.post(
        "/api/tools",
        json={"name": "demo_tool", "display_name": "演示工具"},
    )
    assert response.status_code == 200

    status = client.get("/api/system/status")
    assert status.status_code == 200
    assert status.json()["security"]["admin_key_enabled"] is False


def test_admin_guard_blocks_write_requests_and_records_audit(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "secret-admin-key")
    app = create_app(
        tool_catalog_store=ToolCatalogStore(tmp_path / "tools"),
        api_audit_store=ApiAuditStore(tmp_path / "audit" / "api.jsonl"),
    )
    client = TestClient(app)

    blocked = client.post(
        "/api/tools",
        headers={"X-Actor": "tester"},
        json={"name": "demo_tool", "display_name": "演示工具"},
    )
    assert blocked.status_code == 401

    allowed = client.post(
        "/api/tools",
        headers={"X-Actor": "tester", "X-Admin-Key": "secret-admin-key"},
        json={"name": "demo_tool", "display_name": "演示工具"},
    )
    assert allowed.status_code == 200

    audit = client.get("/api/system/audit-logs?limit=10")
    assert audit.status_code == 200
    records = audit.json()["records"]
    assert len(records) >= 2
    assert records[0]["actor"] == "tester"
    assert records[0]["authorized"] is True
    assert any(item["authorized"] is False for item in records)
