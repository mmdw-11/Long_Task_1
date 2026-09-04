"""运行时工具选择和执行模块，负责把工具目录中的配置安全接入 Agent 运行过程。"""

from __future__ import annotations

import ast
import json
import operator
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .mcp_integration import _arguments_from_schema, _strip_sse, call_mcp_tool_sync
from .product_ops import ToolCatalogStore, ToolRecord
from .tools.contracts import mcp_error_message
from .tools.mcp_remote import MCPAuthorizationStore, call_tool as call_remote_mcp_tool
from .workspace_tools import WorkspaceStore, WorkspaceToolExecutor


@dataclass
class ToolRuntimeResult:
    id: str
    name: str
    display_name: str
    status: str
    arguments: Dict[str, Any]
    result: Any = None
    error: str = ""
    approval_required: bool = False
    risk: str = "low"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "status": self.status,
            "arguments": dict(self.arguments),
            "result": self.result,
            "error": self.error,
            "approval_required": self.approval_required,
            "risk": self.risk,
        }


class ToolRuntime:
    """Resolve allowed tools for an AgentSpec and execute safe adapters."""

    def __init__(
        self,
        catalog: ToolCatalogStore,
        mcp_oauth_store: MCPAuthorizationStore | None = None,
        workspace_store: WorkspaceStore | None = None,
    ) -> None:
        self.catalog = catalog
        self.mcp_oauth_store = mcp_oauth_store
        self.workspace_tools = WorkspaceToolExecutor(workspace_store)

    def available_for_agent(self, tool_ids: Iterable[str] | None) -> List[ToolRecord]:
        ids = [str(item) for item in (tool_ids or []) if str(item).strip()]
        if not ids:
            return []
        records: List[ToolRecord] = []
        adapters: set[str] = set()
        for tool_id in ids:
            try:
                record = self.catalog.get(tool_id)
            except KeyError:
                continue
            if record.enabled:
                records.append(record)
                adapters.add(str(record.metadata.get("adapter") or record.name))
        if "workspace_apply_patch" in adapters and "workspace_write_files" not in adapters:
            companion = next(
                (
                    item for item in self.catalog.list()
                    if item.enabled and str(item.metadata.get("adapter") or item.name) == "workspace_write_files"
                ),
                None,
            )
            if companion is not None:
                records.append(companion)
        return records

    def available_from_mcp(self, tools: Iterable[Dict[str, Any]] | None) -> List[ToolRecord]:
        records: List[ToolRecord] = []
        for item in tools or []:
            try:
                records.append(ToolRecord.from_dict(item))
            except ValueError:
                continue
        return records

    def select_for_task(self, tools: List[ToolRecord], text: str) -> List[ToolRecord]:
        if not tools:
            return []
        normalized = _normalize_math_text(text.lower())
        scored: List[tuple[int, ToolRecord]] = []
        for tool in tools:
            blob = " ".join(
                [
                    tool.name,
                    tool.display_name,
                    tool.description,
                    tool.category,
                    " ".join(tool.tags),
                ]
            ).lower()
            terms = _tokens(blob)
            score = sum(1 for token in terms if token and token in normalized)
            # Chinese phrases are not separated by whitespace.  A task such
            # as “请发送邮件给客户” should still match a “发送邮件” tool.
            score += sum(
                1 for token in terms
                if len(token) >= 2 and any("\u4e00" <= char <= "\u9fff" for char in token)
                and (token in normalized or any(token in candidate for candidate in _tokens(normalized)))
            )
            adapter = str(tool.metadata.get("adapter") or tool.name).lower()
            if adapter in {"current_time", "time", "now"} and re.search(r"时间|日期|today|now|time|date", normalized):
                score += 4
            if adapter in {"calculator", "calc"} and re.search(r"\d+\s*[-+*/()]", normalized):
                score += 4
            if adapter in {"mcp_http", "mcp_url", "mcp"} and re.search(r"mcp|tool|工具|服务|接口", normalized):
                score += 3
            if adapter in {"script", "python_script"} and re.search(r"script|脚本|代码|处理|转换|生成|工具", normalized):
                score += 3
            if score > 0:
                scored.append((score, tool))
        return [tool for _, tool in sorted(scored, key=lambda item: item[0], reverse=True)[:3]]

    def select_for_model(self, tools: List[ToolRecord], text: str, *, limit: int = 12) -> List[ToolRecord]:
        """Keep a large MCP catalogue from distracting the model.

        Remote providers can expose hundreds of administration tools. Passing
        every one to a function-calling model leads to setup calls (mailbox
        creation, templates, WhatsApp, etc.) instead of the requested action.
        This is a relevance *allowlist*, not an execution decision.
        """
        if len(tools) <= limit:
            return tools
        normalized = text.lower()
        wants_email = bool(re.search(r"邮件|邮箱|email|mail", normalized))
        wants_send = wants_email and bool(re.search(r"发送|发给|寄|send|deliver", normalized))
        scored: list[tuple[int, ToolRecord]] = []
        for index, tool in enumerate(tools):
            remote = str(tool.metadata.get("remote_tool_name") or tool.name).lower()
            blob = f"{remote} {tool.display_name} {tool.description}".lower()
            score = sum(1 for token in _tokens(normalized) if token in blob)
            if wants_email:
                if "email" in remote or "email" in blob: score += 8
                else: score -= 10
            if wants_send:
                if remote == "email_send" or ("email" in remote and "send" in remote): score += 100
                elif remote == "email_domains_list": score += 80  # required safe preflight
                elif remote in {"email_mailboxes_list", "whoami", "workspace_get"}: score += 15
                elif any(word in remote for word in ("create", "delete", "update", "template", "whatsapp", "sms")): score -= 30
            scored.append((score, tool))
        selected = [tool for score, tool in sorted(scored, key=lambda item: item[0], reverse=True) if score > 0][:limit]
        return selected or tools[:limit]

    def execute(
        self,
        tool: ToolRecord,
        task_text: str,
        *,
        arguments: Optional[Dict[str, Any]] = None,
        bypass_approval: bool = False,
    ) -> ToolRuntimeResult:
        adapter = str(tool.metadata.get("adapter") or tool.name).strip().lower()
        # A third-party MCP tool without a verified risk declaration is never
        # silently auto-approved.  Built-ins retain their explicit low risk.
        default_risk = "high" if adapter in {"mcp_http", "mcp_url", "mcp", "mcp_tool", "agent_mcp_tool"} else "low"
        risk = str(tool.metadata.get("risk") or default_risk).lower()
        preflight_error = _bird_email_preflight_error(tool.metadata, dict(arguments or {}), self.mcp_oauth_store)
        if preflight_error:
            return ToolRuntimeResult(
                id=tool.id, name=tool.name, display_name=tool.display_name,
                status="blocked", arguments=dict(arguments or {}), error=preflight_error, risk=risk,
            )
        approval_required = risk not in {"low", "read"}
        if approval_required and not bypass_approval:
            return ToolRuntimeResult(
                id=tool.id,
                name=tool.name,
                display_name=tool.display_name,
                status="approval_required",
                arguments=dict(arguments or {"task": task_text[:500]}),
                approval_required=True,
                risk=risk,
            )
        try:
            if adapter in {"current_time", "time", "now"}:
                result = datetime.now(timezone.utc).isoformat()
                args: Dict[str, Any] = {"timezone": "UTC"}
            elif adapter in {"calculator", "calc"}:
                expression = _extract_expression(task_text)
                result = _safe_eval(expression)
                args = {"expression": expression}
            elif adapter in {"echo", "note"}:
                result = task_text[:1000]
                args = dict(arguments or {"text": task_text[:1000]})
            elif adapter in {"mcp_http", "mcp_url", "mcp"}:
                args = dict(arguments or _arguments_from_schema(dict(tool.metadata.get("input_schema") or {}), task_text))
                result = _call_mcp_http(tool.metadata, task_text, arguments=args, oauth_store=self.mcp_oauth_store)
            elif adapter == "openapi_http":
                args = {"url": str(tool.metadata.get("operation_url") or "")}
                result = _call_openapi_http(tool.metadata, task_text)
            elif adapter in {"mcp_tool", "agent_mcp_tool"}:
                args = {
                    "server": str(tool.metadata.get("mcp_name") or tool.metadata.get("mcp_server_id") or ""),
                    "tool": str(tool.metadata.get("tool_name") or tool.name),
                }
                result = call_mcp_tool_sync(tool.metadata, task_text, arguments=arguments)
            elif adapter in {"script", "python_script"}:
                args = {"language": str(tool.metadata.get("language") or "python")}
                result = _run_script_tool(tool.metadata, task_text)
            elif adapter.startswith("workspace_"):
                args = dict(arguments or {})
                # A workspace must be explicit.  It is never inferred from a
                # file path or process working directory.
                result = self.workspace_tools.execute(adapter, args)
            else:
                return ToolRuntimeResult(
                    id=tool.id,
                    name=tool.name,
                    display_name=tool.display_name,
                    status="unsupported",
                    arguments={},
                    error=f"tool adapter {adapter!r} is not registered",
                    risk=risk,
                )
            return ToolRuntimeResult(
                id=tool.id,
                name=tool.name,
                display_name=tool.display_name,
                status="succeeded",
                arguments=args,
                result=result,
                risk=risk,
            )
        except Exception as exc:  # noqa: BLE001 - tool failures should be visible, not crash the run
            return ToolRuntimeResult(
                id=tool.id,
                name=tool.name,
                display_name=tool.display_name,
                status="failed",
                arguments={"task": task_text[:500]},
                error=str(exc),
                risk=risk,
            )


def ensure_builtin_tools(catalog: ToolCatalogStore) -> None:
    """Seed and gently refresh safe built-ins."""
    existing_by_name = {tool.name: tool for tool in catalog.list()}
    defaults = [
        {
            "name": "current_time",
            "display_name": "当前时间",
            "description": "读取当前 UTC 时间，用于需要日期、时间戳或运行时间判断的任务。",
            "category": "system",
            "tags": ["time", "date", "read"],
            "metadata": {"source": "builtin", "adapter": "current_time", "risk": "low", "schema": {"timezone": "string"}},
        },
        {
            "name": "workspace_list_files", "display_name": "列出工作区文件", "description": "列出已选本地工作区中的文件和目录。仅读取工作区内路径。",
            "category": "workspace", "tags": ["workspace", "files", "read"],
            "metadata": {"source": "builtin", "adapter": "workspace_list_files", "risk": "read", "input_schema": {"type":"object","properties":{"workspace_id":{"type":"string"},"path":{"type":"string"},"limit":{"type":"integer"}},"required":["workspace_id"]}},
        },
        {
            "name": "workspace_read_file", "display_name": "读取代码文件", "description": "读取已选本地工作区中的一个文本文件。",
            "category": "workspace", "tags": ["workspace", "files", "read", "code"],
            "metadata": {"source": "builtin", "adapter": "workspace_read_file", "risk": "read", "input_schema": {"type":"object","properties":{"workspace_id":{"type":"string"},"path":{"type":"string"}},"required":["workspace_id","path"]}},
        },
        {
            "name": "workspace_search", "display_name": "搜索工作区代码", "description": "在已选本地工作区内搜索文本，返回匹配文件、行号和片段。",
            "category": "workspace", "tags": ["workspace", "search", "read", "code"],
            "metadata": {"source": "builtin", "adapter": "workspace_search", "risk": "read", "input_schema": {"type":"object","properties":{"workspace_id":{"type":"string"},"query":{"type":"string"},"path":{"type":"string"},"limit":{"type":"integer"}},"required":["workspace_id","query"]}},
        },
        {
            "name": "workspace_apply_patch", "display_name": "应用代码修改", "description": "以精确 old_text/new_text 修改或创建工作区内文件。适合小范围修复；初始化多文件项目时优先使用“批量创建项目文件”。首次本地工程审批通过后，本任务内可自动继续。",
            "category": "workspace", "tags": ["workspace", "write", "patch", "code"],
            "metadata": {"source": "builtin", "adapter": "workspace_apply_patch", "risk": "high", "input_schema": {"type":"object","properties":{"workspace_id":{"type":"string"},"path":{"type":"string"},"old_text":{"type":"string"},"new_text":{"type":"string"},"create":{"type":"boolean"}},"required":["workspace_id","path","new_text"]}},
        },
        {
            "name": "workspace_write_files", "display_name": "批量创建项目文件", "description": "一次创建或更新多个工作区文件，适合初始化一个小项目或生成脚手架；优先用于用户要求“写一个项目/创建项目”。首次需要任务级批准，之后同一任务内的受限本地写入无需重复确认。",
            "category": "workspace", "tags": ["workspace", "write", "scaffold", "code"],
            "metadata": {"source":"builtin","adapter":"workspace_write_files","risk":"high","input_schema":{"type":"object","properties":{"workspace_id":{"type":"string"},"files":{"type":"array","items":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"},"overwrite":{"type":"boolean"}},"required":["path","content"]}}},"required":["workspace_id","files"]}},
        },
        {
            "name": "workspace_run_command", "display_name": "运行项目检查", "description": "在已选工作区内运行白名单测试、构建或静态检查命令。首次本地工程审批通过后，本任务内可自动运行已允许的本地检查。",
            "category": "workspace", "tags": ["workspace", "test", "build", "command"],
            "metadata": {"source": "builtin", "adapter": "workspace_run_command", "risk": "high", "input_schema": {"type":"object","properties":{"workspace_id":{"type":"string"},"action":{"type":"string","enum":["pytest","npm_test","npm_run_build","npm_run_lint","python_compile","javac_compile"]},"target":{"type":"string"},"timeout_seconds":{"type":"integer"}},"required":["workspace_id","action"]}},
        },
        {
            "name": "workspace_git_status", "display_name": "查看 Git 状态", "description": "读取工作区 Git 修改状态。",
            "category": "workspace", "tags": ["workspace", "git", "read"],
            "metadata": {"source": "builtin", "adapter": "workspace_git_status", "risk": "read", "input_schema": {"type":"object","properties":{"workspace_id":{"type":"string"}},"required":["workspace_id"]}},
        },
        {
            "name": "workspace_git_diff", "display_name": "查看代码 Diff", "description": "读取工作区 Git diff，供审查和防漂移检查。",
            "category": "workspace", "tags": ["workspace", "git", "diff", "read"],
            "metadata": {"source": "builtin", "adapter": "workspace_git_diff", "risk": "read", "input_schema": {"type":"object","properties":{"workspace_id":{"type":"string"},"path":{"type":"string"}},"required":["workspace_id"]}},
        },
        {
            "name": "workspace_create_restore_point", "display_name": "创建恢复点", "description": "为当前工作区创建可恢复快照。需要用户批准。",
            "category": "workspace", "tags": ["workspace", "backup", "restore"],
            "metadata": {"source":"builtin","adapter":"workspace_create_restore_point","risk":"high","input_schema":{"type":"object","properties":{"workspace_id":{"type":"string"}},"required":["workspace_id"]}},
        },
        {
            "name": "workspace_restore_point", "display_name": "恢复工作区", "description": "从指定恢复点恢复工作区文件。需要用户批准。",
            "category": "workspace", "tags": ["workspace", "restore", "write"],
            "metadata": {"source":"builtin","adapter":"workspace_restore_point","risk":"high","input_schema":{"type":"object","properties":{"workspace_id":{"type":"string"},"restore_point_id":{"type":"string"}},"required":["workspace_id","restore_point_id"]}},
        },
        {
            "name": "calculator",
            "display_name": "计算器",
            "description": "执行简单安全的四则运算表达式。",
            "category": "utility",
            "tags": ["math", "calculate", "read"],
            "metadata": {"source": "builtin", "adapter": "calculator", "risk": "low", "schema": {"expression": "string"}},
        },
        {
            "name": "task_note",
            "display_name": "任务记录",
            "description": "把当前任务片段作为可审计记录回传给模型，不访问外部系统。",
            "category": "utility",
            "tags": ["note", "debug", "read"],
            "metadata": {"source": "builtin", "adapter": "echo", "risk": "low", "schema": {"text": "string"}},
        },
    ]
    for item in defaults:
        existing = existing_by_name.get(str(item["name"]))
        if existing is None:
            catalog.create(**item)
            continue
        if existing.metadata.get("source") != "builtin":
            continue
        existing.display_name = str(item["display_name"])
        existing.description = str(item["description"])
        existing.category = str(item["category"])
        existing.tags = list(item["tags"])
        existing.metadata = dict(item["metadata"])
        catalog.save(existing)


def _tokens(text: str) -> List[str]:
    return [item for item in re.split(r"[^a-z0-9_\u4e00-\u9fff]+", text.lower()) if item]


def _call_mcp_http(
    metadata: Dict[str, Any], task_text: str, *, arguments: Optional[Dict[str, Any]] = None,
    oauth_store: MCPAuthorizationStore | None = None,
) -> Dict[str, Any]:
    url = str(metadata.get("mcp_url") or metadata.get("url") or "").strip()
    if not url:
        raise ValueError("mcp_url is required")
    remote_name = str(metadata.get("remote_tool_name") or "")
    resolved_arguments = dict(arguments or _arguments_from_schema(dict(metadata.get("input_schema") or {}), task_text))
    if not remote_name:
        raise ValueError("MCP 工具缺少 remote_tool_name")
    credential_env = str(metadata.get("credential_env") or "")
    token = os.environ.get(credential_env, "") if credential_env else ""
    if not token and oauth_store is not None:
        connection_id = str(metadata.get("connection_id") or "")
        token = oauth_store.bearer_token_for(connection_id) or oauth_store.token_for(connection_id)
    try:
        response = call_remote_mcp_tool(url, remote_name, resolved_arguments, token=token, timeout=float(metadata.get("timeout_seconds") or 8))
        return {"url": url, "method": "tools/call", "response": response}
    except Exception as exc:
        if not token:
            raise RuntimeError("MCP 尚未授权。请在“已安装”中完成浏览器登录后再试。") from exc
        raise RuntimeError(f"MCP 工具调用失败：{exc}") from exc


def _bird_email_preflight_error(metadata: Dict[str, Any], arguments: Dict[str, Any], oauth_store: MCPAuthorizationStore | None) -> str:
    """Block invalid Bird email sends before asking the user to approve them."""
    if str(metadata.get("remote_tool_name") or "") != "email_send" or "mcp.bird.com" not in str(metadata.get("mcp_url") or ""):
        return ""
    sender = str(arguments.get("from") or "").strip().lower()
    credential_env = str(metadata.get("credential_env") or "")
    token = os.environ.get(credential_env, "") if credential_env else ""
    if not token and oauth_store is not None:
        connection_id = str(metadata.get("connection_id") or "")
        token = oauth_store.bearer_token_for(connection_id) or oauth_store.token_for(connection_id)
    if not token:
        return "Bird 尚未授权，请先完成 MCP 登录授权。"
    try:
        response = call_remote_mcp_tool(str(metadata.get("mcp_url")), "email_domains_list", {}, token=token, timeout=12)
        text = next((str(item.get("text") or "") for item in response.get("content") or [] if isinstance(item, dict)), "")
        data = json.loads(text).get("data") if text else []
    except Exception as exc:
        return f"无法验证 Bird 发件域，因此没有发送邮件：{exc}"
    domains = [str(item.get("domain") or item.get("name") or "") for item in data or [] if isinstance(item, dict)]
    if not domains:
        return "Bird 工作区没有已验证的发件域，因此不能发送邮件。请先在 Bird 控制台添加并验证发信域名（SPF/DKIM），然后重新发起任务。"
    sender_domain = sender.rsplit("@", 1)[-1] if "@" in sender else ""
    if sender.endswith("@example.com") or sender == "noreply@example.com" or sender_domain not in domains:
        return f"发件地址 {sender or '未提供'} 不属于已验证域。请使用以下已验证域的地址：{', '.join(domains[:5])}。"
    return ""


def _call_openapi_http(metadata: Dict[str, Any], task_text: str) -> Dict[str, Any]:
    url = str(metadata.get("operation_url") or "").strip()
    if not url:
        raise ValueError("OpenAPI operation_url is required")
    method = str(metadata.get("http_method") or "POST").upper()
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    credential_env = str(metadata.get("credential_env") or "")
    if credential_env and os.environ.get(credential_env):
        headers["Authorization"] = f"Bearer {os.environ[credential_env]}"
    data = json.dumps({"input": task_text[:4000]}, ensure_ascii=False).encode("utf-8") if method not in {"GET", "HEAD"} else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=float(metadata.get("timeout_seconds") or 10)) as response:
            text = response.read(200_000).decode("utf-8", errors="replace")
            try: payload: Any = json.loads(text)
            except json.JSONDecodeError: payload = text
            return {"url": url, "status": response.status, "response": payload}
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAPI request failed: {exc}") from exc


def _run_script_tool(metadata: Dict[str, Any], task_text: str) -> Dict[str, Any]:
    if os.environ.get("AGENTFORGE_ENABLE_SCRIPT_TOOLS") != "1":
        return {
            "enabled": False,
            "message": "Script execution is registered but disabled. Set AGENTFORGE_ENABLE_SCRIPT_TOOLS=1 to run user scripts.",
        }
    language = str(metadata.get("language") or "python").lower()
    if language not in {"python", "python3"}:
        raise ValueError("only python script tools are supported")
    script = str(metadata.get("script") or "").strip()
    if not script:
        raise ValueError("script is required")
    timeout = max(1.0, min(float(metadata.get("timeout_seconds") or 5), 30.0))
    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as handle:
        handle.write(script)
        script_path = handle.name
    try:
        completed = subprocess.run(
            [sys.executable, script_path],
            input=json.dumps({"task": task_text}, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def _extract_expression(text: str) -> str:
    match = re.search(r"[-+*/().\d\s]{3,}", _normalize_math_text(text))
    if not match:
        raise ValueError("no arithmetic expression found")
    return match.group(0).strip()


def _normalize_math_text(text: str) -> str:
    """把常见全角和中文数学符号转换为安全计算器可识别的半角形式。"""
    return text.translate(str.maketrans({"＋":"+","－":"-","−":"-","×":"*","＊":"*","÷":"/","／":"/","（":"(","）":")","．":".","＝":"=","？":"?"}))


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return float(_OPS[type(node.op)](_eval(node.left), _eval(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return float(_OPS[type(node.op)](_eval(node.operand)))
        raise ValueError("unsupported arithmetic expression")

    return _eval(tree)
