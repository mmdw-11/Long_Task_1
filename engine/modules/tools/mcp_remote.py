"""Remote HTTP MCP transport and OAuth support.

This module is deliberately the single place that knows how an externally
installed MCP is authenticated.  Tool records only retain a connection id;
OAuth tokens live in a separate encrypted store and are never returned by the
tool catalogue API.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .connection_schema import MANUAL_BEARER_SCHEMA, connection_schema

try:  # ``httpx`` has more reliable TLS/proxy behaviour than urllib on Windows.
    import httpx
except Exception:  # pragma: no cover - dependency guard for partial installs
    httpx = None  # type: ignore[assignment]

try:  # pragma: no cover - installation dependent
    from cryptography.fernet import Fernet
except Exception:  # pragma: no cover
    Fernet = None  # type: ignore[assignment]


class MCPAuthorizationRequired(RuntimeError):
    """The endpoint is reachable but needs an interactive OAuth grant."""

    def __init__(self, details: Dict[str, Any]) -> None:
        self.details = details
        super().__init__("该 MCP 服务需要在浏览器中登录并授权")


def discover_tools(endpoint: str, *, token: str = "", timeout: float = 12) -> List[Dict[str, Any]]:
    payload = _post_jsonrpc(endpoint, "tools/list", {}, token=token, timeout=timeout)
    if payload.get("error"):
        raise ValueError(f"MCP 工具发现失败：{payload['error']}")
    tools = (payload.get("result") or {}).get("tools") or []
    result = [item for item in tools if isinstance(item, dict) and item.get("name")]
    if not result:
        raise ValueError("MCP 服务未返回可用工具")
    return result


def call_tool(endpoint: str, tool_name: str, arguments: Dict[str, Any], *, token: str = "", timeout: float = 15) -> Dict[str, Any]:
    payload = _post_jsonrpc(
        endpoint, "tools/call", {"name": tool_name, "arguments": arguments}, token=token, timeout=timeout
    )
    if payload.get("error"):
        raise RuntimeError(str(payload["error"].get("message") or payload["error"]))
    result = payload.get("result") or {}
    if isinstance(result, dict) and result.get("isError"):
        messages = [str(item.get("text") or "") for item in result.get("content") or [] if isinstance(item, dict)]
        raise RuntimeError("\n".join(item for item in messages if item).strip() or "MCP 工具返回业务错误")
    return result


def inspect_connection(endpoint: str, *, timeout: float = 12) -> Dict[str, Any]:
    """Return a public, deterministic connection schema for a remote MCP."""
    try:
        tools = discover_tools(endpoint, timeout=timeout)
        return {**connection_schema(auth_type="none", help_text="该 MCP 未要求认证，可直接连接。"), "tool_count": len(tools), "protocol_source": "mcp"}
    except MCPAuthorizationRequired as exc:
        protected_resource = str(exc.details.get("resource_metadata") or "")
        if not protected_resource:
            return {**MANUAL_BEARER_SCHEMA, "tool_count": 0, "protocol_source": "http_401"}
        _, metadata = _oauth_metadata(endpoint, protected_resource, timeout)
        if metadata.get("registration_endpoint"):
            return {**connection_schema(auth_type="oauth_dcr", help_text="该 MCP 支持标准 OAuth 自动注册。点击连接后只需在服务商页面登录授权。"), "tool_count": 0, "protocol_source": "oauth_metadata"}
        return {**connection_schema(auth_type="oauth_static", help_text="该 MCP 使用 OAuth，但未开放动态客户端注册。请填写服务商要求的 OAuth Client ID 和 Client Secret。", fields=[
            {"name":"client_id", "label":"OAuth Client ID", "type":"text", "required":True, "scope":"workspace"},
            {"name":"client_secret", "label":"OAuth Client Secret", "type":"secret", "required":True, "scope":"workspace"},
        ]), "tool_count": 0, "protocol_source": "oauth_metadata"}


def start_authorization(
    *, endpoint: str, redirect_uri: str, connection_id: str, store: "MCPAuthorizationStore", timeout: float = 12,
    client_id: str = "", client_secret: str = "",
) -> Dict[str, Any]:
    """Discover OAuth metadata, dynamically register, and create a PKCE grant."""
    cached = store.pending_authorization_url(connection_id)
    if cached:
        return {"connection_id": connection_id, "authorization_url": cached, "resumed": True}
    try:
        discover_tools(endpoint, timeout=timeout)
    except MCPAuthorizationRequired as exc:
        protected_resource = str(exc.details.get("resource_metadata") or "")
    else:
        raise ValueError("该 MCP 不需要 OAuth 授权")
    authorization_server, metadata = _oauth_metadata(endpoint, protected_resource, timeout)
    authorization_endpoint = str(metadata.get("authorization_endpoint") or "")
    token_endpoint = str(metadata.get("token_endpoint") or "")
    if not authorization_endpoint or not token_endpoint:
        raise ValueError("OAuth 授权服务器缺少 authorization_endpoint 或 token_endpoint")
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    registration_endpoint = str(metadata.get("registration_endpoint") or "")
    if not client_id and registration_endpoint:
        registered = _post_json(
            registration_endpoint,
            {
                "client_name": "AgentForge MCP",
                "redirect_uris": [redirect_uri],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
            timeout,
        )
        client_id = str(registered.get("client_id") or "")
        client_secret = str(registered.get("client_secret") or "")
    if not client_id:
        raise ValueError("此 MCP 未开放动态客户端注册，需要填写 OAuth Client ID 和 Client Secret")
    state = secrets.token_urlsafe(32)
    scope = " ".join(str(item) for item in metadata.get("scopes_supported") or [])
    params = {
        "response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri,
        "code_challenge": challenge, "code_challenge_method": "S256", "state": state,
        "resource": endpoint,
    }
    if scope:
        params["scope"] = scope
    authorization_url = authorization_endpoint + "?" + urllib.parse.urlencode(params)
    store.save_pending(connection_id, {
        "state": state, "endpoint": endpoint, "redirect_uri": redirect_uri,
        "authorization_server": authorization_server, "token_endpoint": token_endpoint,
        "client_id": client_id, "client_secret": client_secret, "verifier": verifier,
        "authorization_url": authorization_url,
    })
    return {"connection_id": connection_id, "authorization_url": authorization_url}


def _oauth_metadata(endpoint: str, protected_resource: str, timeout: float) -> tuple[str, Dict[str, Any]]:
    if not protected_resource:
        protected_resource = urllib.parse.urljoin(endpoint.rstrip("/") + "/", ".well-known/oauth-protected-resource")
    resource = _get_json(protected_resource, timeout)
    servers = resource.get("authorization_servers") or []
    if not servers:
        raise ValueError("MCP 服务要求登录，但没有提供 OAuth 授权服务器地址")
    authorization_server = str(servers[0]).rstrip("/")
    metadata_url = authorization_server + "/.well-known/oauth-authorization-server"
    return authorization_server, _get_json(metadata_url, timeout)


def complete_authorization(*, state: str, code: str, store: "MCPAuthorizationStore", timeout: float = 15) -> str:
    connection_id, pending = store.find_pending(state)
    if not connection_id:
        raise ValueError("授权链接已失效或不属于当前工作区")
    token_payload = {
        "grant_type": "authorization_code", "code": code, "redirect_uri": pending["redirect_uri"],
        "client_id": pending["client_id"], "code_verifier": pending["verifier"], "resource": pending["endpoint"],
    }
    if pending.get("client_secret"):
        token_payload["client_secret"] = pending["client_secret"]
    token = _post_form(str(pending["token_endpoint"]), token_payload, timeout)
    if not token.get("access_token"):
        raise ValueError("OAuth 授权没有返回 access_token")
    token["obtained_at"] = time.time()
    token["endpoint"] = pending["endpoint"]
    token["token_endpoint"] = pending["token_endpoint"]
    token["client_id"] = pending["client_id"]
    token["client_secret"] = pending.get("client_secret") or ""
    store.save_tokens(connection_id, token)
    store.delete_pending(connection_id)
    return connection_id


class MCPAuthorizationStore:
    """Encrypted persistence for OAuth grants and short-lived PKCE state."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._cipher = _cipher()

    def save_pending(self, connection_id: str, payload: Dict[str, Any]) -> None:
        existing = self._read(connection_id)
        self._write(connection_id, {"pending": payload, "tokens": existing.get("tokens") or {}, "configuration": existing.get("configuration") or {}})

    def save_configuration(self, connection_id: str, values: Dict[str, Any]) -> None:
        data = self._read(connection_id)
        data["configuration"] = {str(key): str(value) for key, value in values.items() if str(value).strip()}
        self._write(connection_id, data)

    def configuration(self, connection_id: str) -> Dict[str, str]:
        return {str(key): str(value) for key, value in dict(self._read(connection_id).get("configuration") or {}).items()}

    def bearer_token_for(self, connection_id: str) -> str:
        return str(self.configuration(connection_id).get("bearer_token") or "")

    def find_pending(self, state: str) -> tuple[str, Dict[str, Any]]:
        for path in self.root_dir.glob("*.json"):
            data = self._decode(path.read_text(encoding="utf-8"))
            pending = data.get("pending") or {}
            if secrets.compare_digest(str(pending.get("state") or ""), state):
                return path.stem, dict(pending)
        return "", {}

    def pending_authorization_url(self, connection_id: str) -> str:
        """Return the still-valid, encrypted PKCE URL without re-registering."""
        return str((self._read(connection_id).get("pending") or {}).get("authorization_url") or "")

    def delete_pending(self, connection_id: str) -> None:
        data = self._read(connection_id)
        data.pop("pending", None)
        self._write(connection_id, data)

    def save_tokens(self, connection_id: str, token: Dict[str, Any]) -> None:
        data = self._read(connection_id)
        data["tokens"] = token
        self._write(connection_id, data)

    def token_for(self, connection_id: str) -> str:
        token = dict(self._read(connection_id).get("tokens") or {})
        value = str(token.get("access_token") or "")
        if not value:
            return ""
        expires_in = float(token.get("expires_in") or 0)
        obtained = float(token.get("obtained_at") or 0)
        if expires_in and time.time() >= obtained + expires_in - 45:
            value = self._refresh(connection_id, token)
        return value

    def connected(self, connection_id: str) -> bool:
        return bool(self.token_for(connection_id))

    def delete(self, connection_id: str) -> None:
        path = self.root_dir / f"{_safe_id(connection_id)}.json"
        if path.exists():
            path.unlink()

    def _refresh(self, connection_id: str, token: Dict[str, Any]) -> str:
        refresh_token = str(token.get("refresh_token") or "")
        if not refresh_token:
            return str(token.get("access_token") or "")
        # Client information is intentionally kept in the encrypted pending
        # record only while authorising. Servers accepting public PKCE clients
        # do not require it for refresh; otherwise a user reauthorizes.
        endpoint = str(token.get("token_endpoint") or "")
        if not endpoint:
            return str(token.get("access_token") or "")
        payload = {"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": str(token.get("client_id") or "")}
        if token.get("client_secret"):
            payload["client_secret"] = str(token["client_secret"])
        refreshed = _post_form(endpoint, payload, 15)
        token.update(refreshed); token["obtained_at"] = time.time()
        self.save_tokens(connection_id, token)
        return str(token.get("access_token") or "")

    def _path(self, connection_id: str) -> Path:
        return self.root_dir / f"{_safe_id(connection_id)}.json"

    def _read(self, connection_id: str) -> Dict[str, Any]:
        path = self._path(connection_id)
        if not path.exists(): return {}
        return self._decode(path.read_text(encoding="utf-8"))

    def _write(self, connection_id: str, payload: Dict[str, Any]) -> None:
        self._path(connection_id).write_text(self._encode(payload), encoding="utf-8")

    def _encode(self, payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, ensure_ascii=False).encode()
        if self._cipher is None:
            raise RuntimeError("OAuth 凭据存储需要安装 cryptography")
        return "fernet:" + self._cipher.encrypt(raw).decode()

    def _decode(self, raw: str) -> Dict[str, Any]:
        if not raw: return {}
        if not raw.startswith("fernet:") or self._cipher is None:
            return {}
        return dict(json.loads(self._cipher.decrypt(raw.removeprefix("fernet:").encode()).decode()))


def _post_jsonrpc(endpoint: str, method: str, params: Dict[str, Any], *, token: str, timeout: float) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if token: headers["Authorization"] = f"Bearer {token}"
    payload = {"jsonrpc":"2.0", "id":uuid.uuid4().hex, "method":method, "params":params}
    try:
        response = _http_request("POST", endpoint, headers=headers, json_body=payload, timeout=timeout)
        raw = response.content
    except _MCPHttpStatusError as exc:
        if exc.status_code == 401:
            challenge = str(exc.headers.get("WWW-Authenticate") or "")
            resource = _challenge_value(challenge, "resource_metadata")
            raise MCPAuthorizationRequired({"resource_metadata": resource, "challenge": challenge}) from exc
        raise ValueError(f"MCP HTTP 请求失败：{exc.status_code}") from exc
    if len(raw) > 2_000_000: raise ValueError("MCP 响应超过 2MB 限制")
    text = _strip_sse(raw.decode("utf-8", errors="replace"))
    try: return dict(json.loads(text))
    except json.JSONDecodeError as exc: raise ValueError("该地址未返回有效 MCP JSON-RPC 响应") from exc


def _get_json(url: str, timeout: float) -> Dict[str, Any]:
    response = _http_request("GET", url, headers={"Accept":"application/json", "User-Agent":"AgentForge-MCP/1.0"}, timeout=timeout)
    return dict(json.loads(response.content[:300_000].decode()))


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    response = _http_request("POST", url, headers={"Accept":"application/json", "User-Agent":"AgentForge-MCP/1.0"}, json_body=payload, timeout=timeout)
    return dict(json.loads(response.content[:300_000].decode()))


def _post_form(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    try:
        response = _http_request("POST", url, headers={"Accept":"application/json", "User-Agent":"AgentForge-MCP/1.0"}, form_body=payload, timeout=timeout)
        return dict(json.loads(response.content[:300_000].decode()))
    except _MCPHttpStatusError as exc:
        raise ValueError("OAuth 令牌交换失败，请重新连接并完成授权") from exc


def _challenge_value(value: str, name: str) -> str:
    for part in value.split(","):
        key, _, raw = part.strip().partition("=")
        if key.strip().lower().endswith(name): return raw.strip().strip('"')
    marker = name + "=\""; start = value.find(marker)
    if start >= 0:
        end = value.find('"', start + len(marker)); return value[start + len(marker):end] if end >= 0 else ""
    return ""


class _MCPHttpStatusError(RuntimeError):
    def __init__(self, status_code: int, headers: Any) -> None:
        self.status_code, self.headers = status_code, headers
        super().__init__(f"HTTP {status_code}")


def _http_request(
    method: str, url: str, *, headers: Dict[str, str], timeout: float,
    json_body: Optional[Dict[str, Any]] = None, form_body: Optional[Dict[str, Any]] = None,
) -> Any:
    """Make MCP/OAuth requests through httpx, with bounded retries.

    Bird's edge closes some urllib TLS handshakes on Windows. httpx uses a
    different connection stack and also behaves correctly with common local
    proxy settings. The fallback message is actionable rather than exposing a
    low-level SSL exception.
    """
    if httpx is None:
        raise ValueError("MCP OAuth 需要 httpx，请安装 requirements.txt 后重启服务")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=httpx.Timeout(timeout), follow_redirects=False, trust_env=True) as client:
                response = client.request(method, url, headers=headers, json=json_body, data=form_body)
            if response.status_code >= 400:
                raise _MCPHttpStatusError(response.status_code, response.headers)
            return response
        except _MCPHttpStatusError:
            raise
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.35 * (attempt + 1))
    raise ValueError("无法与 MCP OAuth 服务建立安全连接，请检查网络或代理后重试") from last_error


def _strip_sse(raw: str) -> str:
    parts = [line[5:].strip() for line in raw.splitlines() if line.startswith("data:") and line[5:].strip() not in {"", "[DONE]"}]
    return "\n".join(parts) if parts else raw.strip()


def _cipher() -> Any:
    if Fernet is None: return None
    raw = os.environ.get("MCP_SECRET_KEY") or os.environ.get("AGENTFORGE_SECRET_KEY") or "agentforge-dev-mcp-secret"
    if raw.startswith("fernet:"): return Fernet(raw.removeprefix("fernet:").encode())
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(raw.encode()).digest()))


def _safe_id(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum() or ch in {"-", "_"})
