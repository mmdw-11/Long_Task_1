"""AgentScope 适配器。

把 AgentScope 的 agent（如 ``ReActAgent``）封装成符合图节点签名的异步可调用体：
``async fn(state: dict) -> dict``。

约定的状态字段（可通过参数自定义）：
- ``input``    : 传给 agent 的本轮输入（str 或 AgentScope ``Msg``）。
- ``messages`` : 累积的对话消息列表（使用 append reducer 时会自动追加）。

设计要点：
- 对 ``agentscope`` 采用**延迟导入**，仅在真正构造 agent 节点时才导入相关模块。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .node import Node, NodeType


def _lazy_import_msg():
    """延迟导入 AgentScope 的 Msg / TextBlock 类型。

    兼容 AgentScope 2.x（``Msg.content`` 为块列表）与旧版（content 可为字符串）。
    """
    try:
        from agentscope.message import Msg, TextBlock  # type: ignore
        return Msg, TextBlock
    except Exception as exc:  # pragma: no cover - 取决于运行环境
        raise ImportError(
            "使用 AgentScope agent 节点需要安装 agentscope：pip install agentscope"
        ) from exc


def _build_user_msg(Msg: Any, TextBlock: Any, text: str) -> Any:
    """构造一条用户 Msg。优先适配 2.x 的块列表格式，回退到旧版字符串格式。"""
    try:
        return Msg(name="user", content=[TextBlock(type="text", text=text)], role="user")
    except Exception:  # pragma: no cover - 旧版兼容
        return Msg("user", text, "user")


def _extract_text(reply: Any) -> str:
    """从 agent 回复中提取纯文本。

    优先使用 2.x 的 ``Msg.get_text_content()``；否则兼容字符串 / 块列表。
    """
    # AgentScope 2.x
    getter = getattr(reply, "get_text_content", None)
    if callable(getter):
        try:
            text = getter()
            if text is not None:
                return text
        except Exception:  # pragma: no cover
            pass
    content = getattr(reply, "content", reply)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            # 兼容 dict 与 pydantic Block（如 TextBlock）两种形态。
            btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            if btype == "text":
                btext = block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
                parts.append(btext or "")
        return "".join(parts)
    return str(content)


async def _invoke_agent(agent: Any, input_msg: Any) -> Any:
    """调用 AgentScope agent 并返回回复 Msg。

    AgentScope 2.x 的 ``Agent`` 通过 ``await agent.reply(msg)`` 调用（本身不可直接
    调用）；同时兼容部分可直接 ``await agent(msg)`` 调用的实现。
    """
    reply_method = getattr(agent, "reply", None)
    if callable(reply_method):
        return await reply_method(input_msg)
    if input_msg is not None:
        return await agent(input_msg)
    return await agent()


def make_agent_node_func(
    agent: Any,
    input_key: str = "input",
    output_key: Optional[str] = None,
    messages_key: str = "messages",
) -> Callable[[Dict[str, Any]], Any]:
    """构造一个包裹 AgentScope agent 的节点函数。

    :param agent: AgentScope agent 实例（需可 ``await agent(msg)``）。
    :param input_key: 从状态中读取本轮输入的字段名。
    :param output_key: 将 agent 回复文本写入状态的字段名；默认使用 agent 名字。
    :param messages_key: 累积消息列表的字段名。
    """

    async def _node(state: Dict[str, Any]) -> Dict[str, Any]:
        Msg, TextBlock = _lazy_import_msg()

        raw_input = state.get(input_key)
        # 若无显式 input，则尝试用最近一条消息作为输入。
        if raw_input is None:
            history = state.get(messages_key) or []
            raw_input = history[-1] if history else None

        if raw_input is None:
            input_msg = None
        elif isinstance(raw_input, str):
            input_msg = _build_user_msg(Msg, TextBlock, raw_input)
        else:
            input_msg = raw_input  # 已经是 Msg

        reply = await _invoke_agent(agent, input_msg)

        text = _extract_text(reply)
        key = output_key or getattr(agent, "name", "output")
        return {
            key: text,
            messages_key: [reply],
        }

    return _node


def agent_node(
    name: str,
    agent: Any,
    input_key: str = "input",
    output_key: Optional[str] = None,
    messages_key: str = "messages",
    metadata: Optional[Dict[str, Any]] = None,
) -> Node:
    """把一个 AgentScope agent 直接构造为图 ``Node``。"""
    meta = dict(metadata or {})
    meta.setdefault("agent_name", getattr(agent, "name", name))
    return Node(
        name=name,
        func=make_agent_node_func(agent, input_key, output_key, messages_key),
        node_type=NodeType.AGENT,
        metadata=meta,
    )
