"""节点抽象。

节点是图的基本执行单元。一个节点包裹一个"可调用体"，它接收当前状态快照
（dict），返回一个部分状态更新（dict 或 None）。

支持三种来源的可调用体：
- 普通函数（同步或异步）：``fn(state: dict) -> dict | None``
- AgentScope agent：通过 ``adapters`` 封装为符合上述签名的异步函数
- 子图（CompiledGraph）：把整张子图作为一个节点执行
"""

from __future__ import annotations

import asyncio
import enum
import inspect
from typing import Any, Awaitable, Callable, Dict, Optional, Union

# 节点可调用体签名：接收状态快照，返回部分更新。
NodeCallable = Callable[[Dict[str, Any]], Union[Optional[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]]


class NodeType(str, enum.Enum):
    """节点类型，主要用于前端可视化区分与序列化。"""

    FUNCTION = "function"
    AGENT = "agent"
    SUBGRAPH = "subgraph"


class Node:
    """图中的一个节点。

    :param name: 节点唯一名称（在同一张图内唯一）。
    :param func: 节点的可调用体，接收 state dict，返回部分更新 dict 或 None。
    :param node_type: 节点类型，用于可视化与序列化。
    :param metadata: 附加元信息（如 agent 的模型、描述等），供前端展示。
    """

    def __init__(
        self,
        name: str,
        func: NodeCallable,
        node_type: NodeType = NodeType.FUNCTION,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not callable(func):
            raise TypeError(f"节点 {name!r} 的 func 必须可调用")
        self.name = name
        self.func = func
        self.node_type = node_type
        self.metadata: Dict[str, Any] = metadata or {}

    async def invoke(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """执行节点。自动兼容同步 / 异步可调用体。"""
        result = self.func(state)
        if inspect.isawaitable(result):
            result = await result
        return result

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可供前端展示的结构（不含不可序列化的 func 本体）。"""
        return {
            "name": self.name,
            "type": self.node_type.value,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"Node(name={self.name!r}, type={self.node_type.value})"
