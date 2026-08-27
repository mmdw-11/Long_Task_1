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

from .product_ops import ToolCatalogStore, ToolRecord


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

    def __init__(self, catalog: ToolCatalogStore) -> None:
        self.catalog = catalog

    def available_for_agent(self, tool_ids: Iterable[str] | None) -> List[ToolRecord]:
        ids = [str(item) for item in (tool_ids or []) if str(item).strip()]
        if not ids:
            return []
        records: List[ToolRecord] = []
        for tool_id in ids:
            try:
                record = self.catalog.get(tool_id)
            except KeyError:
                continue
            if record.enabled:
                records.append(record)
        return records

    def select_for_task(self, tools: List[ToolRecord], text: str) -> List[ToolRecord]:
        if not tools:
            return []
        normalized = text.lower()
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
            score = sum(1 for token in _tokens(blob) if token and token in normalized)
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

    def execute(self, tool: ToolRecord, task_text: str, *, bypass_approval: bool = False) -> ToolRuntimeResult:
        adapter = str(tool.metadata.get("adapter") or tool.name).strip().lower()
        risk = str(tool.metadata.get("risk") or "low").lower()
        approval_required = risk not in {"low", "read"}
        if approval_required and not bypass_approval:
            return ToolRuntimeResult(
                id=tool.id,
                name=tool.name,
                display_name=tool.display_name,
                status="approval_required",
                arguments={"task": task_text[:500]},
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
                args = {"text": task_text[:1000]}
            elif adapter in {"mcp_http", "mcp_url", "mcp"}:
                args = {"url": str(tool.metadata.get("mcp_url") or tool.metadata.get("url") or "")}
                result = _call_mcp_http(tool.metadata, task_text)
            elif adapter == "openapi_http":
                args = {"url": str(tool.metadata.get("operation_url") or "")}
                result = _call_openapi_http(tool.metadata, task_text)
            elif adapter in {"script", "python_script"}:
                args = {"language": str(tool.metadata.get("language") or "python")}
                result = _run_script_tool(tool.metadata, task_text)
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
    """Seed safe built-ins once so a fresh product has usable tools."""
    existing_names = {tool.name for tool in catalog.list()}
    defaults = [
        {
            "name": "current_time",
            "display_name": "当前时间",
            "description": "读取当前 UTC 时间，用于需要日期、时间戳或运行时间判断的任务。",
            "category": "system",
            "tags": ["time", "date", "read"],
            "metadata": {"adapter": "current_time", "risk": "low", "schema": {"timezone": "string"}},
        },
        {
            "name": "calculator",
            "display_name": "计算器",
            "description": "执行简单安全的四则运算表达式。",
            "category": "utility",
            "tags": ["math", "calculate", "read"],
            "metadata": {"adapter": "calculator", "risk": "low", "schema": {"expression": "string"}},
        },
        {
            "name": "task_note",
            "display_name": "任务记录",
            "description": "把当前任务片段作为可审计记录回传给模型，不访问外部系统。",
            "category": "utility",
            "tags": ["note", "debug", "read"],
            "metadata": {"adapter": "echo", "risk": "low", "schema": {"text": "string"}},
        },
    ]
    for item in defaults:
        if item["name"] not in existing_names:
            catalog.create(**item)


def _tokens(text: str) -> List[str]:
    return [item for item in re.split(r"[^a-z0-9_\u4e00-\u9fff]+", text.lower()) if item]


def _call_mcp_http(metadata: Dict[str, Any], task_text: str) -> Dict[str, Any]:
    url = str(metadata.get("mcp_url") or metadata.get("url") or "").strip()
    if not url:
        raise ValueError("mcp_url is required")
    method = str(metadata.get("method") or "tools/call")
    remote_name = str(metadata.get("remote_tool_name") or "")
    payload = {
        "jsonrpc": "2.0",
        "id": f"agentforge-{datetime.now(timezone.utc).timestamp()}",
        "method": method,
        "params": metadata.get("params") or ({"name": remote_name, "arguments": {"input": task_text[:1000]}} if remote_name else {"task": task_text[:1000]}),
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    credential_env = str(metadata.get("credential_env") or "")
    if credential_env and os.environ.get(credential_env):
        headers["Authorization"] = f"Bearer {os.environ[credential_env]}"
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(metadata.get("timeout_seconds") or 8)) as response:
            text = response.read(200_000).decode("utf-8", errors="replace")
            try:
                parsed: Any = json.loads(text)
            except json.JSONDecodeError:
                parsed = text
            return {"url": url, "method": method, "response": parsed}
    except urllib.error.URLError as exc:
        raise RuntimeError(f"MCP HTTP request failed: {exc}") from exc


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
    match = re.search(r"[-+*/().\d\s]{3,}", text)
    if not match:
        raise ValueError("no arithmetic expression found")
    return match.group(0).strip()


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
