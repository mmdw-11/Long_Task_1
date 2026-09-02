"""Remote MCP configuration, discovery, and runtime calling helpers."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import ipaddress
import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - optional but present in the target Lang_Task env
    from cryptography.fernet import Fernet
except Exception:  # pragma: no cover
    Fernet = None  # type: ignore[assignment]


@dataclass
class MCPToolRecord:
    name: str
    description: str = ""
    title: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "title": self.title,
            "input_schema": dict(self.input_schema),
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPToolRecord":
        name = str(data.get("name") or "").strip()
        if not name:
            raise ValueError("mcp tool name is required")
        return cls(
            name=name,
            description=str(data.get("description") or ""),
            title=str(data.get("title") or data.get("display_name") or ""),
            input_schema=dict(data.get("input_schema") or data.get("inputSchema") or {}),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass
class MCPServerRecord:
    id: str
    user_id: str = ""
    name: str = ""
    transport: str = "streamable_http"
    endpoint: str = ""
    auth_type: str = "none"
    auth_secret: str = ""
    status: str = "active"
    tools: List[MCPToolRecord] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: utc_now())
    updated_at: str = field(default_factory=lambda: utc_now())
    last_synced_at: str = ""

    def to_dict(self, *, include_secret: bool = False) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "transport": self.transport,
            "endpoint": self.endpoint,
            "auth_type": self.auth_type,
            "status": self.status,
            "tools": [tool.to_dict() for tool in self.tools],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_synced_at": self.last_synced_at,
        }
        data["has_auth_secret"] = bool(self.auth_secret)
        if include_secret:
            data["auth_secret"] = self.auth_secret
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPServerRecord":
        name = str(data.get("name") or "").strip()
        endpoint = str(data.get("endpoint") or "").strip()
        if not name:
            raise ValueError("mcp server name is required")
        if not endpoint:
            raise ValueError("mcp endpoint is required")
        return cls(
            id=clean_id(str(data.get("id") or "")) or f"mcp-{uuid.uuid4().hex[:12]}",
            user_id=str(data.get("user_id") or ""),
            name=name,
            transport=str(data.get("transport") or "streamable_http"),
            endpoint=endpoint,
            auth_type=str(data.get("auth_type") or "none"),
            auth_secret=str(data.get("auth_secret") or ""),
            status=str(data.get("status") or "active"),
            tools=[MCPToolRecord.from_dict(item) for item in data.get("tools") or []],
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
            last_synced_at=str(data.get("last_synced_at") or ""),
        )


@dataclass
class AgentMCPBindingRecord:
    id: str
    agent_id: str
    mcp_server_id: str
    enabled: bool = True
    enabled_tools: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: utc_now())
    updated_at: str = field(default_factory=lambda: utc_now())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "mcp_server_id": self.mcp_server_id,
            "enabled": self.enabled,
            "enabled_tools": list(self.enabled_tools),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMCPBindingRecord":
        agent_id = str(data.get("agent_id") or "").strip()
        mcp_server_id = str(data.get("mcp_server_id") or "").strip()
        if not agent_id or not mcp_server_id:
            raise ValueError("agent_id and mcp_server_id are required")
        return cls(
            id=clean_id(str(data.get("id") or "")) or f"agent-mcp-{uuid.uuid4().hex[:12]}",
            agent_id=agent_id,
            mcp_server_id=mcp_server_id,
            enabled=bool(data.get("enabled", True)),
            enabled_tools=[str(item) for item in data.get("enabled_tools") or []],
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
        )


class MCPConfigStore:
    """File-backed store mirroring mcp_servers, mcp_tools, and agent_mcp_servers."""

    def __init__(self, root_dir: str | Path = "runs/mcp") -> None:
        self.root_dir = Path(root_dir)
        self.server_dir = self.root_dir / "servers"
        self.binding_dir = self.root_dir / "agent_bindings"
        self.server_dir.mkdir(parents=True, exist_ok=True)
        self.binding_dir.mkdir(parents=True, exist_ok=True)
        self._cipher = _build_cipher()

    def create_server(
        self,
        *,
        name: str,
        endpoint: str,
        auth_type: str = "none",
        token: str = "",
        tools: Optional[List[MCPToolRecord]] = None,
        enabled_tools: Optional[Iterable[str]] = None,
        user_id: str = "",
    ) -> MCPServerRecord:
        validate_remote_mcp_url(endpoint)
        auth_headers(auth_type, token)
        enabled = {str(item) for item in (enabled_tools or [])}
        discovered = []
        for tool in tools or []:
            discovered.append(
                MCPToolRecord(
                    name=tool.name,
                    title=tool.title,
                    description=tool.description,
                    input_schema=tool.input_schema,
                    enabled=not enabled or tool.name in enabled,
                )
            )
        record = MCPServerRecord.from_dict(
            {
                "name": name,
                "user_id": user_id,
                "transport": "streamable_http",
                "endpoint": endpoint,
                "auth_type": auth_type,
                "auth_secret": self.encrypt_secret(token) if token else "",
                "status": "active",
                "tools": [tool.to_dict() for tool in discovered],
                "last_synced_at": utc_now(),
            }
        )
        return self.save_server(record)

    def save_server(self, record: MCPServerRecord) -> MCPServerRecord:
        now = utc_now()
        if self.exists_server(record.id):
            record.created_at = self.get_server(record.id).created_at
        record.updated_at = now
        self._server_path(record.id).write_text(
            json.dumps(record.to_dict(include_secret=True), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return record

    def get_server(self, server_id: str) -> MCPServerRecord:
        path = self._server_path(server_id)
        if not path.exists():
            raise KeyError(f"mcp server {server_id!r} not found")
        return MCPServerRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_servers(self) -> List[MCPServerRecord]:
        return sorted(
            [self.get_server(path.stem) for path in self.server_dir.glob("*.json")],
            key=lambda item: item.updated_at,
            reverse=True,
        )

    def delete_server(self, server_id: str) -> None:
        path = self._server_path(server_id)
        if not path.exists():
            raise KeyError(f"mcp server {server_id!r} not found")
        path.unlink()
        for binding in self.list_bindings():
            if binding.mcp_server_id == server_id:
                self.delete_binding(binding.id)

    def exists_server(self, server_id: str) -> bool:
        return self._server_path(server_id).exists()

    def bind_agent(
        self,
        *,
        agent_id: str,
        mcp_server_id: str,
        enabled_tools: Iterable[str],
    ) -> AgentMCPBindingRecord:
        for binding in self.list_agent_bindings(agent_id):
            if binding.mcp_server_id == mcp_server_id:
                binding.enabled = True
                binding.enabled_tools = [str(item) for item in enabled_tools]
                return self.save_binding(binding)
        return self.save_binding(
            AgentMCPBindingRecord.from_dict(
                {
                    "agent_id": agent_id,
                    "mcp_server_id": mcp_server_id,
                    "enabled": True,
                    "enabled_tools": [str(item) for item in enabled_tools],
                }
            )
        )

    def save_binding(self, record: AgentMCPBindingRecord) -> AgentMCPBindingRecord:
        now = utc_now()
        if self.exists_binding(record.id):
            record.created_at = self.get_binding(record.id).created_at
        record.updated_at = now
        self._binding_path(record.id).write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return record

    def get_binding(self, binding_id: str) -> AgentMCPBindingRecord:
        path = self._binding_path(binding_id)
        if not path.exists():
            raise KeyError(f"agent mcp binding {binding_id!r} not found")
        return AgentMCPBindingRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_bindings(self) -> List[AgentMCPBindingRecord]:
        return sorted(
            [self.get_binding(path.stem) for path in self.binding_dir.glob("*.json")],
            key=lambda item: item.updated_at,
            reverse=True,
        )

    def list_agent_bindings(self, agent_id: str) -> List[AgentMCPBindingRecord]:
        return [binding for binding in self.list_bindings() if binding.agent_id == agent_id]

    def delete_binding(self, binding_id: str) -> None:
        path = self._binding_path(binding_id)
        if not path.exists():
            raise KeyError(f"agent mcp binding {binding_id!r} not found")
        path.unlink()

    def exists_binding(self, binding_id: str) -> bool:
        return self._binding_path(binding_id).exists()

    def decrypt_secret(self, value: str) -> str:
        if not value:
            return ""
        if not value.startswith("fernet:") or Fernet is None:
            return value
        return self._cipher.decrypt(value.removeprefix("fernet:").encode("utf-8")).decode("utf-8")

    def encrypt_secret(self, value: str) -> str:
        if not value:
            return ""
        if Fernet is None:
            return value
        return "fernet:" + self._cipher.encrypt(value.encode("utf-8")).decode("utf-8")

    def runtime_tools_for_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for binding in self.list_agent_bindings(agent_id):
            if not binding.enabled:
                continue
            try:
                server = self.get_server(binding.mcp_server_id)
            except KeyError:
                continue
            if server.status != "active":
                continue
            enabled = set(binding.enabled_tools)
            for tool in server.tools:
                if not tool.enabled or tool.name not in enabled:
                    continue
                records.append(
                    {
                        "id": f"{server.id}:{tool.name}",
                        "name": tool.name,
                        "display_name": tool.title or tool.name,
                        "description": tool.description,
                        "category": "mcp",
                        "enabled": True,
                        "tags": ["mcp", server.name, tool.name],
                        "metadata": {
                            "source": "agent_mcp",
                            "adapter": "mcp_tool",
                            "risk": "read",
                            "mcp_server_id": server.id,
                            "mcp_name": server.name,
                            "mcp_url": server.endpoint,
                            "auth_type": server.auth_type,
                            "auth_secret": self.decrypt_secret(server.auth_secret),
                            "tool_name": tool.name,
                            "input_schema": tool.input_schema,
                            "timeout_seconds": 15,
                        },
                    }
                )
        return records

    def agent_payload(self, agent_id: str) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for binding in self.list_agent_bindings(agent_id):
            try:
                server = self.get_server(binding.mcp_server_id)
            except KeyError:
                continue
            server_dict = server.to_dict()
            payload.append(
                {
                    **binding.to_dict(),
                    "server": server_dict,
                    "tools": [
                        {
                            **tool.to_dict(),
                            "enabled_for_agent": tool.name in set(binding.enabled_tools),
                        }
                        for tool in server.tools
                    ],
                }
            )
        return payload

    def _server_path(self, server_id: str) -> Path:
        clean = clean_id(server_id)
        if not clean:
            raise ValueError("mcp server id is required")
        return self.server_dir / f"{clean}.json"

    def _binding_path(self, binding_id: str) -> Path:
        clean = clean_id(binding_id)
        if not clean:
            raise ValueError("mcp binding id is required")
        return self.binding_dir / f"{clean}.json"


async def test_mcp_connection(
    *,
    endpoint: str,
    auth_type: str = "none",
    token: str = "",
    timeout: float = 15.0,
) -> Dict[str, Any]:
    validate_remote_mcp_url(endpoint)
    tools = await list_mcp_tools(endpoint=endpoint, auth_type=auth_type, token=token, timeout=timeout)
    return {
        "success": True,
        "server": {"name": urllib.parse.urlparse(endpoint).netloc or "remote-mcp"},
        "tools": [tool.to_dict() for tool in tools],
    }


async def list_mcp_tools(
    *,
    endpoint: str,
    auth_type: str = "none",
    token: str = "",
    timeout: float = 15.0,
) -> List[MCPToolRecord]:
    if _is_builtin_demo_endpoint(endpoint):
        return _builtin_demo_tools()
    headers = auth_headers(auth_type, token)
    try:
        from agentscope.mcp import HttpMCPConfig, MCPClient  # type: ignore

        client = MCPClient(
            name="remote-mcp-discovery",
            is_stateful=False,
            mcp_config=HttpMCPConfig(url=endpoint, headers=headers or None, timeout=timeout),
        )
        await maybe_await(client.connect())
        try:
            raw_tools = await maybe_await(client.list_raw_tools())
            return [_tool_from_any(item) for item in raw_tools]
        finally:
            await maybe_await(client.close())
    except Exception:
        return await _jsonrpc_list_tools(endpoint=endpoint, headers=headers, timeout=timeout)


async def call_mcp_tool(
    metadata: Dict[str, Any], task_text: str, *, arguments: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    endpoint = str(metadata.get("mcp_url") or "").strip()
    tool_name = str(metadata.get("tool_name") or metadata.get("name") or "").strip()
    if not endpoint or not tool_name:
        raise ValueError("mcp_url and tool_name are required")
    validate_remote_mcp_url(endpoint)
    if _is_builtin_demo_endpoint(endpoint):
        args = dict(arguments or _arguments_from_schema(dict(metadata.get("input_schema") or {}), task_text))
        return {
            "tool_name": tool_name,
            "arguments": args,
            "response": {"content": [{"type": "text", "text": f"本地 MCP 已收到：{json.dumps(args, ensure_ascii=False)}"}]},
        }
    auth_type = str(metadata.get("auth_type") or "none")
    token = str(metadata.get("auth_secret") or metadata.get("token") or "")
    headers = auth_headers(auth_type, token)
    args = dict(arguments or _arguments_from_schema(dict(metadata.get("input_schema") or {}), task_text))
    return await _jsonrpc_call_tool(
        endpoint=endpoint,
        headers=headers,
        tool_name=tool_name,
        arguments=args,
        timeout=float(metadata.get("timeout_seconds") or 15),
    )


def call_mcp_tool_sync(
    metadata: Dict[str, Any], task_text: str, *, arguments: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run an MCP tool from sync runtime code, including inside an active loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(call_mcp_tool(metadata, task_text, arguments=arguments))

    result: Dict[str, Any] = {}
    error: list[BaseException] = []

    def _run() -> None:
        try:
            result.update(asyncio.run(call_mcp_tool(metadata, task_text, arguments=arguments)))
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            error.append(exc)

    import threading

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result


def validate_remote_mcp_url(endpoint: str) -> None:
    parsed = urllib.parse.urlparse(endpoint.strip())
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError("当前仅支持 HTTP MCP Endpoint，例如 https://example.com/mcp")
    allow_local = os.environ.get("MCP_ALLOW_LOCALHOST") == "1"
    if parsed.scheme != "https" and not allow_local:
        raise ValueError("当前仅支持 https:// Remote MCP，暂不支持 npx、uvx、Docker 或本地 stdio")
    host = parsed.hostname or ""
    if host.lower() in {"localhost"} and not allow_local:
        raise ValueError("MCP URL 不允许指向 localhost")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, None)}
    except socket.gaierror as exc:
        raise ValueError(f"无法解析 MCP URL 主机：{host}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if allow_local:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            raise ValueError("MCP URL 不允许指向私网、回环或 link-local 地址")


def auth_headers(auth_type: str, token: str) -> Dict[str, str]:
    if auth_type == "bearer":
        if not token:
            raise ValueError("Bearer Token 不能为空")
        return {"Authorization": f"Bearer {token}"}
    if auth_type in {"", "none"}:
        return {}
    raise ValueError("第一版仅支持无认证或 Bearer Token")


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _jsonrpc_list_tools(*, endpoint: str, headers: Dict[str, str], timeout: float) -> List[MCPToolRecord]:
    payload = await asyncio.to_thread(
        _post_jsonrpc,
        endpoint,
        headers,
        {"jsonrpc": "2.0", "id": "tools-list", "method": "tools/list", "params": {}},
        timeout,
    )
    if "error" in payload:
        raise RuntimeError(str(payload["error"].get("message") or payload["error"]))
    tools = (payload.get("result") or {}).get("tools") or []
    return [_tool_from_any(item) for item in tools]


async def _jsonrpc_call_tool(
    *,
    endpoint: str,
    headers: Dict[str, str],
    tool_name: str,
    arguments: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    payload = await asyncio.to_thread(
        _post_jsonrpc,
        endpoint,
        headers,
        {
            "jsonrpc": "2.0",
            "id": f"tools-call-{uuid.uuid4().hex[:8]}",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        },
        timeout,
    )
    if "error" in payload:
        raise RuntimeError(str(payload["error"].get("message") or payload["error"]))
    return {"tool_name": tool_name, "arguments": arguments, "response": payload.get("result")}


def _post_jsonrpc(endpoint: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    request_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        **headers,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    opener = urllib.request.build_opener(_NoRedirectHandler)
    try:
        with opener.open(request, timeout=max(1.0, min(timeout, 30.0))) as response:
            raw = response.read(250_000).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise RuntimeError("认证失败，请检查 Bearer Token") from exc
        raise RuntimeError(f"MCP HTTP request failed: {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法访问 MCP Server：{exc}") from exc
    text = _strip_sse(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("该地址未返回有效的 MCP 服务") from exc


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(req.full_url, code, "redirect blocked", headers, fp)


def _strip_sse(raw: str) -> str:
    text = raw.strip()
    chunks = []
    for line in text.splitlines():
        if line.startswith("data:"):
            value = line[5:].strip()
            if value and value != "[DONE]":
                chunks.append(value)
    return "\n".join(chunks).strip() if chunks else text


def _tool_from_any(item: Any) -> MCPToolRecord:
    if isinstance(item, dict):
        data = item
    else:
        data = {
            "name": getattr(item, "name", ""),
            "title": getattr(item, "title", ""),
            "description": getattr(item, "description", ""),
            "input_schema": getattr(item, "inputSchema", None) or getattr(item, "input_schema", None) or {},
        }
    return MCPToolRecord.from_dict(
        {
            "name": data.get("name"),
            "title": data.get("title") or data.get("displayName") or data.get("name"),
            "description": data.get("description") or "",
            "input_schema": data.get("input_schema") or data.get("inputSchema") or {},
            "enabled": data.get("enabled", True),
        }
    )


def _arguments_from_schema(schema: Dict[str, Any], task_text: str) -> Dict[str, Any]:
    props = schema.get("properties") if isinstance(schema, dict) else {}
    required = [str(item) for item in schema.get("required") or []] if isinstance(schema, dict) else []
    if isinstance(props, dict) and {"libraryId", "query"}.issubset(props):
        match = re.search(r"(/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?)", task_text)
        return {"libraryId": match.group(1) if match else task_text[:1000], "query": task_text[:1000]}
    if isinstance(props, dict) and len(required) == 1:
        return {required[0]: task_text[:1000]}
    if isinstance(props, dict) and "query" in props:
        return {"query": task_text[:1000]}
    if isinstance(props, dict) and "task" in props:
        return {"task": task_text[:1000]}
    return {"task": task_text[:1000]}


def _is_builtin_demo_endpoint(endpoint: str) -> bool:
    parsed = urllib.parse.urlparse(endpoint)
    return (parsed.hostname in {"127.0.0.1", "localhost"} and parsed.path.rstrip("/") == "/mcp/demo")


def _builtin_demo_tools() -> List[MCPToolRecord]:
    return [
        MCPToolRecord(
            name="preview_email",
            title="preview_email",
            description="仅生成邮件预览，不发送真实邮件",
            input_schema={"type": "object", "properties": {"task": {"type": "string"}}},
        ),
        MCPToolRecord(
            name="lookup_demo",
            title="lookup_demo",
            description="返回本地演示检索结果",
            input_schema={"type": "object", "properties": {"task": {"type": "string"}}},
        ),
    ]


def _build_cipher() -> Any:
    if Fernet is None:
        return None
    raw = os.environ.get("MCP_SECRET_KEY") or os.environ.get("AGENTFORGE_SECRET_KEY") or "agentforge-dev-mcp-secret"
    if raw.startswith("fernet:"):
        return Fernet(raw.removeprefix("fernet:").encode("utf-8"))
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def clean_id(raw: str) -> str:
    return "".join(ch for ch in raw.strip() if ch.isalnum() or ch in {"-", "_", ":"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
