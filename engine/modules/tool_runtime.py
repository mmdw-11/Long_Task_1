"""Runtime tool selection and execution for agent runs.

Tool records are product metadata. A tool only becomes executable when its
``name`` or ``metadata.adapter`` maps to one of the safe built-in adapters
below. This keeps natural-language descriptions useful for model/tool
selection without turning descriptions into machine permissions.
"""

from __future__ import annotations

import ast
import operator
import re
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
            if score > 0:
                scored.append((score, tool))
        return [tool for _, tool in sorted(scored, key=lambda item: item[0], reverse=True)[:3]]

    def execute(self, tool: ToolRecord, task_text: str) -> ToolRuntimeResult:
        adapter = str(tool.metadata.get("adapter") or tool.name).strip().lower()
        risk = str(tool.metadata.get("risk") or "low").lower()
        approval_required = risk not in {"low", "read"}
        if approval_required:
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
