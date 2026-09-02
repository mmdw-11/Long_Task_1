"""Deterministic schemas for external tool connections.

Schemas are sourced from trusted marketplace metadata or MCP/OAuth protocol
metadata. They intentionally never involve an LLM and never contain secret
values in responses.
"""

from __future__ import annotations

from typing import Any, Dict, List


def connection_schema(*, auth_type: str, fields: List[Dict[str, Any]] | None = None, help_text: str = "") -> Dict[str, Any]:
    normalized = []
    for item in fields or []:
        name = str(item.get("name") or "").strip()
        if not name or not name.replace("_", "").isalnum():
            continue
        field_type = str(item.get("type") or "text")
        if field_type not in {"text", "secret", "select"}:
            field_type = "text"
        normalized.append({
            "name": name,
            "label": str(item.get("label") or name),
            "description": str(item.get("description") or ""),
            "type": field_type,
            "required": bool(item.get("required", True)),
            "default": str(item.get("default") or "") if field_type != "secret" else "",
            "choices": [str(value) for value in item.get("choices") or []] if field_type == "select" else [],
            "scope": str(item.get("scope") or "connection"),
        })
    return {"auth_type": auth_type, "fields": normalized, "help_text": help_text}


def missing_required(schema: Dict[str, Any], values: Dict[str, Any]) -> List[str]:
    return [
        str(field["label"])
        for field in schema.get("fields") or []
        if field.get("required") and not str(values.get(field.get("name")) or "").strip()
    ]


GMAIL_STATIC_OAUTH_SCHEMA = connection_schema(
    auth_type="oauth_static",
    help_text="Google 官方 Gmail MCP 需要你的 Google Cloud OAuth 应用。普通用户只需登录 Google；只有连接管理员需要填写以下应用配置。",
    fields=[
        {"name":"client_id", "label":"Google OAuth Client ID", "type":"text", "required":True, "scope":"workspace", "description":"在 Google Cloud → Google Auth Platform → Clients 中创建 Web application 后复制。"},
        {"name":"client_secret", "label":"Google OAuth Client Secret", "type":"secret", "required":True, "scope":"workspace", "description":"只提交给后端加密保存，保存后不会再次显示。"},
    ],
)


MANUAL_BEARER_SCHEMA = connection_schema(
    auth_type="manual_bearer",
    help_text="该服务没有提供标准 OAuth 元数据。仅在服务商文档明确要求 API Key 或 Bearer Token 时使用此高级选项。",
    fields=[
        {"name":"bearer_token", "label":"Bearer Token / API Key", "type":"secret", "required":True, "scope":"connection", "description":"直接加密保存；不要把密钥写进模型提示词或环境变量名称输入框。"},
    ],
)
