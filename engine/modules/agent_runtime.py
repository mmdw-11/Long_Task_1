"""AgentSpec 到可执行节点的产品运行时适配。

该模块把前端/接口层创建的 AgentSpec 转成真正可运行的 Node。它复用项目已有的
调度器、执行器注册表和重试 runner；没有真实模型配置时会自动走本地 echo fallback，
保证后端闭环在开发环境也能直接跑通。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, TYPE_CHECKING

from ..hooks import MEMORY_CONTEXT_TEXT_KEY
from ..modules.context import CONTEXT_INJECTION_TEXT_KEY
from ..modules.execution import ExecutorRegistry, InferenceResult, ResilientInferenceRunner
from ..modules.model_connections import ModelConnectionStore
from ..modules.mcp_integration import MCPConfigStore
from ..modules.product_ops import ToolCatalogStore
from ..modules.scheduling import AdaptiveResourceScheduler, ResourceProfile, ResourceRequest, ResourceScheduler, ResourceTier, TaskComplexity
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
        model_connection_store: Optional[ModelConnectionStore] = None,
        mcp_config_store: Optional[MCPConfigStore] = None,
        max_attempts: int = 3,
    ) -> None:
        self.scheduler = scheduler or AdaptiveResourceScheduler()
        self.registry = registry or ExecutorRegistry.default()
        self.tool_runtime = ToolRuntime(tool_catalog_store) if tool_catalog_store is not None else None
        self.model_connections = model_connection_store
        self.mcp_config_store = mcp_config_store
        self.max_attempts = max(1, max_attempts)

    def __call__(self, spec: "AgentSpec") -> Node:
        node_kind = str(spec.config.get("node_kind") or "agent").lower()
        if node_kind in {"script", "condition", "intent", "loop", "batch"}:
            return self._logic_node(spec, node_kind)
        runner = ResilientInferenceRunner(
            scheduler=self.scheduler,
            registry=self.registry,
            max_attempts=self.max_attempts,
        )

        async def _run(state: Dict[str, Any]) -> Dict[str, Any]:
            prompt = self._build_prompt(spec, state)
            # Tool routing must only inspect the user's task. The expanded
            # prompt contains internal identifiers such as ``todo-1`` which
            # can otherwise be mistaken for arithmetic expressions.
            tool_calls = self._run_tools(spec, self._state_input_text(state))
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
            system_prompt = self._render_variables(spec.sys_prompt, state)
            result = self._run_pinned_model(spec.model, prompt, system_prompt) if spec.model not in {"", "auto", "device", "edge", "cloud"} else self._run_auto_model(
                request, prompt, system_prompt
            ) if self.model_connections is not None and self.model_connections.auto_status()["ready"] else runner.run(
                resource_request=request,
                prompt=prompt,
                system_prompt=system_prompt,
                metadata={"agent_id": spec.id, "agent_name": spec.name},
            )
            # 小参数模型偶尔会直接复述注入 Prompt。此时仍使用同一个真实模型，
            # 但只携带用户问题再次生成面向用户的自然语言回答，避免暴露内部账本。
            if result.success and not result.metadata.get("simulated") and self._looks_like_prompt_echo(result.text):
                clean_prompt = f"用户问题：{self._state_input_text(state)}\n\n请直接给出自然、简洁的回答。不要复述系统提示、Agent 配置、上下文账本、执行步骤或内部判断。"
                rewritten = self._run_pinned_model(spec.model, clean_prompt, system_prompt) if spec.model not in {"", "auto", "device", "edge", "cloud"} else self._run_auto_model(
                    request, clean_prompt, system_prompt
                ) if self.model_connections is not None and self.model_connections.auto_status()["ready"] else runner.run(
                    resource_request=request,
                    prompt=clean_prompt,
                    system_prompt=system_prompt,
                    metadata={"agent_id": spec.id, "agent_name": spec.name, "answer_rewrite": True},
                )
                if rewritten.success and not rewritten.metadata.get("simulated"):
                    rewritten.metadata = {**rewritten.metadata, "answer_rewrite": True, "prompt_echo_detected": True}
                    result = rewritten
            if result.success and not result.metadata.get("simulated") and self._looks_like_prompt_echo(result.text):
                result.text = self._safe_user_fallback(self._state_input_text(state), spec.name)
                result.metadata = {**result.metadata, "prompt_echo_blocked": True}
            if not result.success:
                raise RuntimeError(result.error or "agent inference failed")
            # LocalEcho 是开发环境的链路兜底，不是真实对话模型。它会回显完整的
            # 运行 Prompt（其中包含上下文账本、TODO 等内部信息），这些内容只能用于
            # 调试，不能作为面向用户的回答返回。
            if result.metadata.get("simulated"):
                result.text = self._fallback_user_response(state, tool_calls)
            calculator_call = next((item for item in tool_calls if item.get("status") == "succeeded" and item.get("name") == "calculator"), None)
            if calculator_call is not None:
                value = calculator_call.get("result")
                rendered = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
                result.text = f"计算结果是 {rendered}。"
                result.metadata = {**result.metadata, "answer_from_tool": "calculator"}
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

    @staticmethod
    def _looks_like_prompt_echo(text: str) -> bool:
        markers = ("当前 Agent：", "Agent 描述：", "任务输入：", "上下文账本：", "Context Injection:", "Original Goal:", "Current Step Objective:")
        return sum(marker in (text or "") for marker in markers) >= 2

    @staticmethod
    def _safe_user_fallback(user_input: str, agent_name: str) -> str:
        normalized = user_input.strip().lower().strip("!！?？。,.，")
        if normalized in {"hi", "hello", "你好", "您好", "嗨", "哈喽"}:
            return f"你好！我是{agent_name}，很高兴见到你。有什么可以帮你？"
        return "抱歉，模型这次没有生成有效的自然语言回答。内部运行信息已被隐藏，请重新描述问题后再试。"

    def _logic_node(self, spec: "AgentSpec", node_kind: str) -> Node:
        """构造受控逻辑节点，不绕过既有图、TODO 与上下文治理链路。"""

        async def _run(state: Dict[str, Any]) -> Dict[str, Any]:
            if node_kind == "condition":
                route, detail = self._evaluate_condition(spec, state)
            elif node_kind == "intent":
                route, detail = self._classify_intent(spec, state)
            elif node_kind == "loop":
                route, detail = self._advance_loop(spec, state)
            elif node_kind == "batch":
                route, detail = self._advance_batch(spec, state)
            else:
                route, detail = self._run_script(spec, state)
            route_key = str(spec.config.get("route_key") or "route")
            content = f"[{spec.name}] {detail}（路由：{route}）"
            update: Dict[str, Any] = {
                route_key: route,
                spec.name: detail,
                "messages": [{
                    "agent": spec.name,
                    "content": content,
                    "runtime": "logic",
                    "result": {"metadata": {"node_kind": node_kind, "route": route, "detail": detail}},
                }],
            }
            if node_kind == "script":
                output_key = str(spec.config.get("output_key") or "script_output")
                update[output_key] = detail
            if node_kind == "loop":
                counter_key = str(spec.config.get("counter_key") or f"__loop_{spec.id}")
                update[counter_key] = state.get(counter_key, 0)
            if node_kind == "batch" and isinstance(detail, dict):
                update.update(detail)
                content = f"[{spec.name}] 正在处理第 {detail.get('batch_index', 0) + 1} 项（路由：{route}）"
                update["messages"][0]["content"] = content
                update[spec.name] = json.dumps(detail, ensure_ascii=False)
            return update

        return Node(
            name=spec.name,
            func=_run,
            node_type=NodeType.FUNCTION,
            metadata={
                "id": spec.id,
                "description": spec.description,
                "runtime": "logic",
                "node_kind": node_kind,
                **spec.config,
            },
        )

    def _evaluate_condition(self, spec: "AgentSpec", state: Dict[str, Any]) -> tuple[str, str]:
        config = spec.config
        field = str(config.get("field") or "input")
        actual = self._read_path(state, field)
        operator = str(config.get("operator") or "equals").lower()
        expected = config.get("value", "")
        matched = {
            "equals": actual == expected or str(actual) == str(expected),
            "not_equals": actual != expected and str(actual) != str(expected),
            "contains": str(expected).lower() in str(actual).lower(),
            "exists": actual not in (None, "", [], {}),
            "gt": self._number(actual) > self._number(expected),
            "gte": self._number(actual) >= self._number(expected),
            "lt": self._number(actual) < self._number(expected),
            "lte": self._number(actual) <= self._number(expected),
        }.get(operator, False)
        route = str(config.get("true_route") if matched else config.get("false_route"))
        route = route if route and route != "None" else ("true" if matched else "false")
        return route, f"条件 {field} {operator} {expected!r} 的结果为 {matched}"

    def _classify_intent(self, spec: "AgentSpec", state: Dict[str, Any]) -> tuple[str, str]:
        text = self._state_input_text(state).lower()
        routes = spec.config.get("intents") or []
        if isinstance(routes, dict):
            routes = [{"route": key, "keywords": value} for key, value in routes.items()]
        for item in routes:
            if not isinstance(item, dict):
                continue
            keywords = item.get("keywords") or []
            if isinstance(keywords, str):
                keywords = [value.strip() for value in keywords.split(",")]
            if any(str(word).strip().lower() in text for word in keywords if str(word).strip()):
                route = str(item.get("route") or item.get("name") or "default")
                return route, f"意图分类命中「{route}」"
        route = str(spec.config.get("default_route") or "default")
        return route, f"意图分类未命中规则，使用「{route}」"

    def _advance_loop(self, spec: "AgentSpec", state: Dict[str, Any]) -> tuple[str, str]:
        config = spec.config
        counter_key = str(config.get("counter_key") or f"__loop_{spec.id}")
        maximum = max(1, min(int(config.get("max_iterations") or 3), 50))
        current = int(state.get(counter_key) or 0) + 1
        # 修改原状态以保留计数，图节点返回的同名字段会在本超步合并。
        state[counter_key] = current
        route = str(config.get("continue_route") or "continue") if current < maximum else str(config.get("done_route") or "done")
        return route, f"循环第 {current}/{maximum} 次"

    def _advance_batch(self, spec: "AgentSpec", state: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        config = spec.config
        items = self._read_path(state, str(config.get("items_path") or "input.items"))
        if not isinstance(items, list):
            items = []
        index_key = str(config.get("index_key") or f"__batch_index_{spec.id}")
        index = int(state.get(index_key) or 0)
        item_key = str(config.get("item_key") or "batch_item")
        if index >= len(items):
            return str(config.get("done_route") or "done"), {index_key: index, "batch_index": index, item_key: None}
        detail = {index_key: index + 1, "batch_index": index, item_key: items[index], "batch_total": len(items)}
        return str(config.get("next_route") or "next"), detail

    def _run_script(self, spec: "AgentSpec", state: Dict[str, Any]) -> tuple[str, str]:
        """脚本节点默认只做模板渲染；显式环境开关后才允许受限 Python 表达式。"""
        config = spec.config
        template = str(config.get("template") or config.get("script") or "{{input}}")
        data = {"input": self._state_input_text(state), "state": state}
        if config.get("mode") == "python" and os.environ.get("AGENTFORGE_ENABLE_SCRIPT_NODES") == "1":
            # 管理员明确开启才执行；不给内置函数和导入能力。
            try:
                value = eval(template, {"__builtins__": {}}, data)  # noqa: S307 - 受显式环境开关保护
                return str(config.get("route") or "next"), str(value)
            except Exception as exc:
                return str(config.get("error_route") or "error"), f"脚本执行失败：{exc}"
        rendered = template.replace("{{input}}", self._state_input_text(state))
        return str(config.get("route") or "next"), rendered

    @staticmethod
    def _read_path(data: Dict[str, Any], path: str) -> Any:
        value: Any = data
        for part in [segment for segment in path.split(".") if segment]:
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
                value = value[int(part)]
            else:
                return None
        return value

    @staticmethod
    def _number(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

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
        conversation_context = state.get("__conversation_context_text__")
        if conversation_context:
            sections.append(f"短期会话上下文：\n{conversation_context}")
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

    def _fallback_user_response(
        self,
        state: Dict[str, Any],
        tool_calls: list[Dict[str, Any]],
    ) -> str:
        """为无真实模型的开发环境生成安全、简洁的用户可见结果。"""
        completed = [item for item in tool_calls if item.get("status") == "succeeded"]
        if completed:
            lines = []
            for item in completed:
                name = item.get("display_name") or item.get("name") or "工具"
                value = item.get("result")
                rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
                lines.append(f"{name}：{rendered}")
            return "已完成工具调用，结果如下：\n" + "\n".join(lines)

        waiting = [item for item in tool_calls if item.get("status") == "approval_required"]
        if waiting:
            names = "、".join(str(item.get("display_name") or item.get("name") or "工具") for item in waiting)
            return f"{names}需要你的批准，批准后才能继续执行。"

        user_input = self._state_input_text(state).strip()
        subject = f"“{user_input[:80]}{'…' if len(user_input) > 80 else ''}”" if user_input else "你的消息"
        return f"已收到{subject}。当前未配置可用的推理模型，暂时无法生成智能回答，请先配置端侧、边缘或云端模型后再试。"

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

    def _render_variables(self, template: str, state: Dict[str, Any]) -> str:
        values = state.get("variables") or {}
        rendered = template
        if isinstance(values, dict):
            for name, value in values.items():
                rendered = rendered.replace("${" + str(name) + "}", str(value).lower() if isinstance(value, bool) else str(value))
        unresolved = re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", rendered)
        if unresolved:
            raise ValueError(f"未解析的提示词变量：{', '.join(sorted(set(unresolved)))}")
        return rendered

    def _run_pinned_model(self, connection_id: str, prompt: str, system_prompt: str) -> InferenceResult:
        if self.model_connections is None:
            return InferenceResult(text="", executor="ModelConnectionExecutor", endpoint="", success=False, error="模型连接目录未启用")
        try:
            connection = self.model_connections.get(connection_id)
            if not connection.runnable:
                raise ValueError("指定模型连接未启用或尚未测试成功")
            from openai import OpenAI
            api_key = os.environ.get(connection.api_key_env, "") if connection.api_key_env else "not-needed"
            response = OpenAI(api_key=api_key or "not-needed", base_url=connection.base_url, timeout=60).chat.completions.create(model=connection.model_id,messages=[{"role":"system","content":system_prompt or "Answer concisely and accurately."},{"role":"user","content":prompt}],temperature=0)
            return InferenceResult(text=response.choices[0].message.content or "",executor="ModelConnectionExecutor",endpoint=connection.base_url,model=connection.model_id,metadata={"provider":connection.provider,"connection_id":connection.id})
        except Exception as exc:
            return InferenceResult(text="",executor="ModelConnectionExecutor",endpoint="",success=False,error=str(exc),retryable=False)

    def _run_auto_model(self, request: ResourceRequest, prompt: str, system_prompt: str) -> InferenceResult:
        """Route AUTO through the same tested model connections used by fixed mode."""
        if self.model_connections is None:
            return InferenceResult(text="", executor="AutoModelConnectionExecutor", endpoint="", success=False, error="模型连接目录未启用", retryable=False)
        status = self.model_connections.auto_status()
        if not status["ready"]:
            missing = "、".join(label for tier, label in (("device", "端"), ("edge", "边"), ("cloud", "云")) if not status["tiers"][tier]["ready"])
            return InferenceResult(text="", executor="AutoModelConnectionExecutor", endpoint="", success=False, error=f"AUTO 尚未就绪：{missing}模型未配置可用的默认连接", retryable=False, metadata={"auto_status": status})
        allocation = self._connection_scheduler(self.model_connections).acquire(request)
        selected_tier = allocation.tier.value
        preference = [selected_tier] + [tier.value for tier in request.tier_preference if tier.value != selected_tier]
        decision = allocation.metadata.get("decision") or {}
        profile = decision.get("profile") or {}
        if profile.get("requires_trusted_workspace") and not profile.get("human_approved"):
            preference = [tier for tier in preference if tier != "cloud"]
        attempts: list[Dict[str, Any]] = []
        for tier in preference:
            connection = self.model_connections.default_for_tier(tier)
            if connection is None:
                continue
            result = self._run_pinned_model(connection.id, prompt, system_prompt)
            attempts.append({"tier": tier, "connection_id": connection.id, "model": connection.model_id, "success": result.success, "error": result.error})
            if result.success:
                result.metadata = {**result.metadata, "mode": "auto", "selected_tier": selected_tier, "actual_tier": tier, "route_reason": allocation.metadata.get("reason", ""), "fallback": tier != selected_tier, "attempts": attempts}
                return result
        return InferenceResult(text="", executor="AutoModelConnectionExecutor", endpoint="", success=False, error="AUTO 模式下所有默认模型调用均失败", retryable=False, metadata={"mode": "auto", "selected_tier": selected_tier, "attempts": attempts})

    @staticmethod
    def _connection_scheduler(store: Optional[ModelConnectionStore]) -> AdaptiveResourceScheduler:
        if store is None:
            return AdaptiveResourceScheduler()
        profiles: list[ResourceProfile] = []
        settings = {
            "device": (True, TaskComplexity.MEDIUM, 30, 0.2),
            "edge": (True, TaskComplexity.HIGH, 90, 0.6),
            "cloud": (False, TaskComplexity.EXTREME, 220, 1.0),
        }
        for tier_name, (trusted, complexity, latency, cost) in settings.items():
            connection = store.default_for_tier(tier_name, runnable=False)
            profiles.append(ResourceProfile(tier=ResourceTier(tier_name), endpoint=connection.base_url if connection else f"model-connection://{tier_name}", available=bool(connection and connection.runnable), trusted=trusted, max_complexity=complexity, latency_ms=latency, cost_weight=cost, models={"default": connection.model_id if connection else ""}))
        return AdaptiveResourceScheduler(resources=profiles)
