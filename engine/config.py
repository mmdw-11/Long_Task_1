"""OpenAI 模型配置（从 .env 读取）。

统一在此处读取 ``.env`` 中的 OpenAI 相关配置，并提供便捷工厂：

- :func:`load_settings`      —— 读取 .env 中的配置。
- :func:`build_openai_model` —— 构造 AgentScope 2.x 的 ``OpenAIChatModel``。
- :func:`build_openai_agent` —— 构造一个可直接用于图节点的 AgentScope ``Agent``。

.env 支持的字段：
- ``OPENAI_API_KEY``   : OpenAI / 兼容服务的 API Key（必填）
- ``OPENAI_MODEL``     : 模型名，默认 ``gpt-4o-mini``
- ``OPENAI_BASE_URL``  : 兼容端点地址（可选，如自建/中转/vLLM）
- ``OPENAI_ORG``       : 组织 ID（可选）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class OpenAISettings:
    """从环境 / .env 解析出的 OpenAI 配置。"""

    api_key: str
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    organization: Optional[str] = None


def load_settings(dotenv_path: Optional[str] = None) -> OpenAISettings:
    """加载 .env 并返回 OpenAI 配置。

    :param dotenv_path: 可选，指定 .env 路径；默认自动向上查找。
    """
    try:
        from dotenv import load_dotenv  # type: ignore

        # override=True：以 .env 的值为准，避免 shell 中残留的同名环境变量抢占。
        load_dotenv(dotenv_path=dotenv_path, override=True)
    except Exception:
        # 读取 .env 失败时，退回到已存在的环境变量。
        pass

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "缺少 OPENAI_API_KEY，请在项目根目录的 .env 中配置（可参考 .env.example）"
        )
    return OpenAISettings(
        api_key=api_key,
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        organization=os.environ.get("OPENAI_ORG") or None,
    )


def build_openai_model(settings: Optional[OpenAISettings] = None, stream: bool = False) -> Any:
    """构造 AgentScope 2.x 的 ``OpenAIChatModel``。

    使用 ``OpenAICredential`` 承载 api_key / base_url / organization。
    """
    settings = settings or load_settings()
    from agentscope.credential import OpenAICredential
    from agentscope.model import OpenAIChatModel

    credential = OpenAICredential(
        api_key=settings.api_key,
        base_url=settings.base_url,
        organization=settings.organization,
    )
    return OpenAIChatModel(
        credential=credential,
        model=settings.model,
        stream=stream,
    )


def build_openai_agent(
    name: str,
    system_prompt: str,
    settings: Optional[OpenAISettings] = None,
    stream: bool = False,
    toolkit: Optional[Any] = None,
) -> Any:
    """构造一个基于 OpenAI 的 AgentScope 2.x ``Agent``，可直接接入图节点。

    :param toolkit: 可选的 :class:`agentscope.tool.Toolkit`；传入后 Agent 会在
        ReAct 循环中自主调用其中注册的工具。
    """
    from agentscope.agent import Agent

    model = build_openai_model(settings, stream=stream)
    agent = Agent(
        name=name,
        system_prompt=system_prompt,
        model=model,
        toolkit=toolkit,
    )

    # FunctionTool 包装的工具默认是 ASK 权限（需人工确认才执行），
    # 在无人值守的流水线里会卡在“等待外部确认”。挂载工具时切到 BYPASS，
    # 让 ReAct 循环可以自动调用这些工具。
    if toolkit is not None:
        from agentscope.permission import PermissionMode

        agent.state.permission_context.mode = PermissionMode.BYPASS

    return agent
