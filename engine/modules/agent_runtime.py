"""AgentSpec 到可执行节点的产品运行时适配。

该模块把前端/接口层创建的 AgentSpec 转成真正可运行的 Node。它复用项目已有的
调度器、执行器注册表和重试 runner；没有真实模型配置时会自动走本地 echo fallback，
保证后端闭环在开发环境也能直接跑通。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, TYPE_CHECKING

from ..hooks import MEMORY_CONTEXT_TEXT_KEY
from ..modules.context import CONTEXT_INJECTION_TEXT_KEY
from ..modules.execution import ExecutorRegistry, ResilientInferenceRunner
from ..modules.mcp_integration import MCPConfigStore
from ..modules.product_ops import ToolCatalogStore
from ..modules.scheduling import AdaptiveResourceScheduler, ResourceRequest, ResourceScheduler, ResourceTier
from ..modules.tool_runtime import ToolRuntime
from ..modules.skills import SKILL_CONTEXT_TEXT_KEY
from ..node import Node, NodeType

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型提示，避免运行时循环导入
    from ..orchestrator import AgentSpec


class AgentRuntimeFactory:
    """把 AgentSpec 构造成调用推理 runner 的图节点。"""

    def __init__(
        self,
        *,
        scheduler: Optional[ResourceScheduler] = None,
        registry: Optional[ExecutorRegistry] = None,
        tool_catalog_store: Optional[ToolCatalogStore] = None,
        mcp_config_store: Optional[MCPConfigStore] = None,
        max_attempts: int = 3,
    ) -> None:
        self.scheduler = scheduler or AdaptiveResourceScheduler()
        self.registry = registry or ExecutorRegistry.default()
        self.tool_runtime = ToolRuntime(tool_catalog_store) if tool_catalog_store is not None else None
        self.mcp_config_store = mcp_config_store
        self.max_attempts = max(1, max_attempts)

    def __call__(self, spec: "AgentSpec") -> Node:
        runner = ResilientInferenceRunner(
            scheduler=self.scheduler,
            registry=self.registry,
            max_attempts=self.max_attempts,
        )

        async def _run(state: Dict[str, Any]) -> Dict[str, Any]:
            prompt = self._build_prompt(spec, state)
            tool_calls = self._run_tools(spec, prompt)
            if tool_calls:
                prompt = f"{prompt}\n\n工具调用结果：\n{json.dumps(tool_calls, ensure_ascii=False, default=str)}"
            request = ResourceRequest(
                node=spec.name,
                tier_preference=self._tier_preference(spec),
                metadata={
                    "agent_id": spec.id,
                    "agent_name": spec.name,
                    "model": spec.model,
                    **spec.config,
                },
                state=state,
            )
            result = runner.run(
                resource_request=request,
                prompt=prompt,
                system_prompt=spec.sys_prompt,
                metadata={"agent_id": spec.id, "agent_name": spec.name},
            )
            if not result.success:
                raise RuntimeError(result.error or "agent inference failed")
            return {
                "input": result.text,
                spec.name: result.text,
                "__runtime_tool_calls__": tool_calls,
                "messages": [
                    {
                        "agent": spec.name,
                        "content": result.text,
                        "runtime": "inference",
                        "result": {
                            **result.to_dict(),
                            "metadata": {
                                **result.metadata,
                                "tool_calls": tool_calls,
                            },
                        },
                    }
                ],
            }

        return Node(
            name=spec.name,
            func=_run,
            node_type=NodeType.AGENT,
            metadata={
                "id": spec.id,
                "model": spec.model,
                "description": spec.description,
                "sys_prompt": spec.sys_prompt,
                "runtime": "inference",
                "children": list(spec.children),
                **spec.config,
            },
        )

    def _build_prompt(self, spec: "AgentSpec", state: Dict[str, Any]) -> str:
        # 将运行时上下文拼成一个稳定输入，便于本地 fallback 和真实模型共用。
        sections = [
            f"当前 Agent：{spec.name}",
            f"Agent 描述：{spec.description or '无'}",
            f"任务输入：{self._state_input_text(state)}",
        ]
        context_injection = state.get(CONTEXT_INJECTION_TEXT_KEY)
        skill_context = state.get(SKILL_CONTEXT_TEXT_KEY)
        memory_context = state.get(MEMORY_CONTEXT_TEXT_KEY)
        if context_injection:
            sections.append(f"上下文账本：\n{context_injection}")
        if skill_context:
            sections.append(f"可复用技能：\n{skill_context}")
        if memory_context:
            sections.append(f"记忆上下文：\n{memory_context}")
        return "\n\n".join(sections)

    def _run_tools(self, spec: "AgentSpec", task_text: str) -> list[Dict[str, Any]]:
        if self.tool_runtime is None:
            return []
        tool_ids = spec.config.get("tool_ids") or spec.config.get("tools") or []
        available = self.tool_runtime.available_for_agent(tool_ids)
        if self.mcp_config_store is not None:
            available.extend(self.tool_runtime.available_from_mcp(self.mcp_config_store.runtime_tools_for_agent(spec.id)))
        selected = self.tool_runtime.select_for_task(available, task_text)
        return [self.tool_runtime.execute(tool, task_text).to_dict() for tool in selected]

    def _state_input_text(self, state: Dict[str, Any]) -> str:
        value = state.get("input", state)
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)

    def _tier_preference(self, spec: "AgentSpec") -> list[ResourceTier]:
        raw = spec.config.get("tier_preference") or spec.config.get("resource_tier")
        if raw is None:
            return [ResourceTier.DEVICE, ResourceTier.EDGE, ResourceTier.CLOUD]
        items = raw if isinstance(raw, list) else [raw]
        tiers: list[ResourceTier] = []
        for item in items:
            try:
                tiers.append(ResourceTier(str(item)))
            except ValueError:
                continue
        return tiers or [ResourceTier.DEVICE, ResourceTier.EDGE, ResourceTier.CLOUD]
