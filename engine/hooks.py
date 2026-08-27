"""执行钩子与钩子管理器。

在图执行引擎（:class:`CompiledGraph`）的关键位置预留扩展点，让记忆、动态路由、
流控、恢复策略、端边云资源调度等模块得以挂载或替换默认行为。

设计要点：
- ``ExecutionHook``：定义所有扩展点，全部方法默认 no-op / passthrough，
  子类按需覆盖即可。
- ``HookManager``：持有各扩展模块实例（默认全部为 NoOp 桩），把引擎回调翻译为
  对应模块的调用；未注册任何模块时行为与框架现状完全一致。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .failure import FAILURES_KEY, FailureRecord, FailureTrace
from .modules.flow import FlowController, FlowDecision, NoOpFlowController
from .modules.evaluation import Evaluator, EvaluationResult, RuleEvaluator
from .modules.context import (
    BUDGET_PAUSED,
    CONTEXT_INJECTION_KEY,
    CONTEXT_INJECTION_TEXT_KEY,
    CONTEXT_LEDGER_KEY,
    DRIFT_RESULT_KEY,
    PAUSE_REASON_KEY,
    RUN_STATUS_KEY,
    ContextBudgetController,
    ContextInjector,
    ContextLedgerStore,
    DriftDetector,
    FailureSummary,
    TodoManager,
)
from .modules.memory import (
    MemoryContext,
    MemoryItem,
    MemoryScope,
    MemoryStore,
    NoOpMemoryStore,
    WakeupLevel,
    wakeup_profile,
)
from .modules.recovery import (
    NoOpRecoveryStrategy,
    RecoveryAction,
    RecoveryStrategy,
    RepairPlan,
)
from .modules.routing import (
    NoOpRouter,
    PassthroughRoutingPolicy,
    RoutingDecision,
    RoutingPolicy,
    Router,
)
from .modules.security import AuditPackBuilder, SensitiveDataRedactor
from .modules.skills import SKILL_CONTEXT_KEY, SKILL_CONTEXT_TEXT_KEY, SkillRetriever, SkillTraceEvent, SkillTraceStore
from .modules.scheduling import (
    NoOpResourceScheduler,
    ResourceAllocation,
    ResourceRequest,
    ResourceScheduler,
)

MEMORY_CONTEXT_KEY = "__memory_context__"
MEMORY_CONTEXT_ITEMS_KEY = "__memory_context_items__"
MEMORY_CONTEXT_TEXT_KEY = "__memory_context_text__"
RESOURCE_ALLOCATION_KEY = "__resource_allocation__"
REDACTION_RESULT_KEY = "__redaction__"
AUDIT_PACK_KEY = "__audit_pack__"
EVALUATION_RESULT_KEY = "__evaluation__"
VALIDATION_FAILED_STATUS = "validation_failed"


@dataclass
class NodeContext:
    """节点执行上下文，贯穿单个节点的各扩展点调用。"""

    node: str
    step: int
    state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionHook:
    """执行钩子基类：所有扩展点默认 no-op / passthrough。"""

    def on_step_start(self, step: int, frontier: List[str], state: Dict[str, Any]) -> None:
        """一个超步开始时调用。"""
        return None

    def on_node_start(self, ctx: NodeContext) -> FlowDecision:
        """节点执行前调用，返回流控决策（默认 EXECUTE）。"""
        return FlowDecision.EXECUTE

    def acquire_resource(self, ctx: NodeContext) -> Optional[ResourceAllocation]:
        """节点执行前申请资源（默认不分配）。"""
        return None

    def release_resource(self, ctx: NodeContext, allocation: Optional[ResourceAllocation]) -> None:
        """节点执行后释放资源（默认 no-op）。"""
        return None

    def on_node_end(self, ctx: NodeContext, update: Optional[Dict[str, Any]]) -> None:
        """节点执行成功后调用（默认 no-op）。"""
        return None

    def handle_validation_failure(
        self, ctx: NodeContext, update: Optional[Dict[str, Any]]
    ) -> Optional[List[str]]:
        """处理后置验证失败（默认不介入，保持兼容的记录语义）。"""
        return None

    def on_node_error(self, ctx: NodeContext, error: BaseException) -> Optional[List[str]]:
        """节点执行失败时调用。

        :return: 修复路由目标列表（引擎将据此续跑）；返回 None 表示不恢复，
            由引擎维持抛错。
        """
        return None

    def resolve_successors(
        self, node: str, candidates: List[str], state: Dict[str, Any]
    ) -> Optional[List[str]]:
        """对后继候选做动态调整（默认返回 None → 使用原候选）。"""
        return None


class HookManager(ExecutionHook):
    """把各扩展模块桥接到引擎扩展点。

    未显式注入的模块使用对应 NoOp 桩，保证与框架现有行为一致。

    :param project_rules: 项目规则，作为动态路由策略的输入上下文。
    :param extra_hooks: 额外的 :class:`ExecutionHook`（在模块行为之外附加，如日志）。
    """

    def __init__(
        self,
        *,
        memory: Optional[MemoryStore] = None,
        router: Optional[Router] = None,
        routing_policy: Optional[RoutingPolicy] = None,
        flow_controller: Optional[FlowController] = None,
        recovery_strategy: Optional[RecoveryStrategy] = None,
        scheduler: Optional[ResourceScheduler] = None,
        context_ledger: Optional[ContextLedgerStore] = None,
        context_budget: Optional[ContextBudgetController] = None,
        context_injector: Optional[ContextInjector] = None,
        drift_detector: Optional[DriftDetector] = None,
        evaluator: Optional[Evaluator] = None,
        redactor: Optional[SensitiveDataRedactor] = None,
        audit_builder: Optional[AuditPackBuilder] = None,
        skill_retriever: Optional[SkillRetriever] = None,
        skill_trace_store: Optional[SkillTraceStore] = None,
        project_rules: Optional[Dict[str, Any]] = None,
        graph_view: Optional[Dict[str, Any]] = None,
        extra_hooks: Optional[List[ExecutionHook]] = None,
        memory_top_k: int = 5,
        wakeup_level: int | str = WakeupLevel.STANDARD,
    ) -> None:
        self.memory = memory or NoOpMemoryStore()
        self.router = router or NoOpRouter()
        self.routing_policy = routing_policy or PassthroughRoutingPolicy()
        self.flow_controller = flow_controller or NoOpFlowController()
        self.recovery_strategy = recovery_strategy or NoOpRecoveryStrategy()
        self.scheduler = scheduler or NoOpResourceScheduler()
        self.context_ledger = context_ledger
        self.context_budget = context_budget
        self.context_injector = context_injector or ContextInjector()
        self.drift_detector = drift_detector or DriftDetector()
        self.validation_control_enabled = context_ledger is not None or evaluator is not None
        self.evaluator = evaluator or RuleEvaluator()
        self.redactor = redactor or SensitiveDataRedactor()
        self.audit_builder = audit_builder or AuditPackBuilder()
        self.skill_retriever = skill_retriever
        self.skill_trace_store = skill_trace_store
        self.project_rules = project_rules or {}
        self.graph_view = graph_view or {}
        self.extra_hooks: List[ExecutionHook] = list(extra_hooks or [])
        self.memory_top_k = memory_top_k
        self.wakeup_profile = wakeup_profile(wakeup_level)

    # ------------------------------------------------------------------ #
    # 扩展点实现（桥接到各模块）
    # ------------------------------------------------------------------ #
    def on_step_start(self, step: int, frontier: List[str], state: Dict[str, Any]) -> None:
        self._update_context_on_step_start(step, frontier, state)
        self._check_context_budget(state)
        for h in self.extra_hooks:
            h.on_step_start(step, frontier, state)

    def on_node_start(self, ctx: NodeContext) -> FlowDecision:
        self._inject_memory_context(ctx)
        self._inject_skill_context(ctx)
        deps = self.flow_controller.resolve_dependencies(ctx.node, self.graph_view)
        deps_satisfied = self._deps_satisfied(deps, ctx.state)
        decision = self.flow_controller.decide(
            ctx.node, state=ctx.state, deps_satisfied=deps_satisfied
        )
        for h in self.extra_hooks:
            h.on_node_start(ctx)
        return decision

    def acquire_resource(self, ctx: NodeContext) -> Optional[ResourceAllocation]:
        request = ResourceRequest(
            node=ctx.node,
            metadata={
                **ctx.metadata,
                **self._node_metadata(ctx.node),
                **self._scheduling_rules(),
            },
            state=ctx.state,
        )
        allocation = self.scheduler.acquire(request)
        self._apply_security_controls(ctx, allocation)
        ctx.state[RESOURCE_ALLOCATION_KEY] = allocation.to_dict()
        return allocation

    def release_resource(self, ctx: NodeContext, allocation: Optional[ResourceAllocation]) -> None:
        if allocation is not None:
            self.scheduler.release(allocation)

    def on_node_end(self, ctx: NodeContext, update: Optional[Dict[str, Any]]) -> None:
        evaluation = self.evaluator.evaluate_node(
            node=ctx.node,
            update=update,
            state=ctx.state,
            metadata={**ctx.metadata, **self._node_metadata(ctx.node)},
        )
        ctx.state[EVALUATION_RESULT_KEY] = evaluation.to_dict()
        self._update_context_on_node_end(ctx, update, evaluation)
        # 将节点产出写入记忆（NoOp 桩会丢弃）。
        if update:
            self.memory.append(
                update,
                MemoryScope.TASK,
                context=self._memory_context(ctx),
                tags=[ctx.node],
                node=ctx.node,
                step=ctx.step,
            )
        self._record_skill_trace(ctx, "node_end", {"update": update or {}})
        for h in self.extra_hooks:
            h.on_node_end(ctx, update)

    def handle_validation_failure(
        self, ctx: NodeContext, update: Optional[Dict[str, Any]]
    ) -> Optional[List[str]]:
        evaluation_data = ctx.state.get(EVALUATION_RESULT_KEY) or {}
        if not self.validation_control_enabled:
            return None
        if evaluation_data.get("passed", True):
            return None
        action = evaluation_data.get("action", "pause")
        if action == "record":
            return None
        if action == "reroute":
            ctx.state.pop(RUN_STATUS_KEY, None)
            ctx.state[PAUSE_REASON_KEY] = "; ".join(evaluation_data.get("findings") or [])
            return [str(item) for item in evaluation_data.get("targets") or []]
        ctx.state[RUN_STATUS_KEY] = VALIDATION_FAILED_STATUS
        ctx.state[PAUSE_REASON_KEY] = "; ".join(evaluation_data.get("findings") or [])
        return []

    def on_node_error(self, ctx: NodeContext, error: BaseException) -> Optional[List[str]]:
        trace = FailureTrace.from_state(ctx.state)
        trace.record(ctx.node, error, step=ctx.step)
        plan: RepairPlan = self.recovery_strategy.plan(
            trace, node=ctx.node, state=ctx.state
        )
        for h in self.extra_hooks:
            h.on_node_error(ctx, error)
        self._record_skill_trace(
            ctx,
            "node_error",
            {"error_type": type(error).__name__, "message": str(error)},
        )
        self._update_context_on_node_error(
            ctx, error, recoverable=not plan.should_abort
        )
        if plan.should_abort:
            return None
        if plan.action == RecoveryAction.RETRY:
            return [ctx.node]
        return list(plan.targets)

    def resolve_successors(
        self, node: str, candidates: List[str], state: Dict[str, Any]
    ) -> Optional[List[str]]:
        decision: RoutingDecision = self.routing_policy.decide(
            node,
            list(candidates),
            state=state,
            context={
                "project_rules": self.project_rules,
                "memory_ctx": self.memory.cascade_read(
                    query=node,
                    context=self._memory_context_from_state(state, node=node, step=0),
                    top_k=self.memory_top_k,
                ),
                "failure_trace": FailureTrace.from_state(state),
            },
        )
        result = decision.targets
        for h in self.extra_hooks:
            overridden = h.resolve_successors(node, result, state)
            if overridden is not None:
                result = overridden
        return result

    # ------------------------------------------------------------------ #
    # 供引擎构造一条失败记录的状态增量（写回 __failures__）。
    # ------------------------------------------------------------------ #
    @staticmethod
    def failure_update(node: str, error: BaseException, step: int) -> Dict[str, Any]:
        rec = FailureRecord(
            node=node, error_type=type(error).__name__, message=str(error), step=step
        )
        return {FAILURES_KEY: [rec.to_dict()]}

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    @staticmethod
    def _deps_satisfied(deps: List[str], state: Dict[str, Any]) -> bool:
        """默认依赖判定：状态中存在同名字段即视为该依赖已产出。"""
        if not deps:
            return True
        return all(d in state for d in deps)

    def _inject_memory_context(self, ctx: NodeContext) -> None:
        memory_ctx = self._memory_context(ctx)
        query = self._memory_query(ctx)
        wake = getattr(self.memory, "wake", None)
        if callable(wake):
            result = wake(query, profile=self.wakeup_profile, context=memory_ctx)
            items = result.items
            context_text = result.context_text
        else:
            items = self.memory.cascade_read(
                query,
                narrowest=MemoryScope.WORKING,
                context=memory_ctx,
                top_k=self.memory_top_k,
            )
            context_text = self._format_context_pack(items)
        ctx.state[MEMORY_CONTEXT_KEY] = {
            "working_id": memory_ctx.working_id,
            "task_id": memory_ctx.task_id,
            "project_id": memory_ctx.project_id,
            "global_id": memory_ctx.global_id,
            "query": query,
            "wakeup_level": self.wakeup_profile.level.value,
            "wakeup_profile": self.wakeup_profile.name,
        }
        ctx.state[MEMORY_CONTEXT_ITEMS_KEY] = [item.to_dict() for item in items]
        ctx.state[MEMORY_CONTEXT_TEXT_KEY] = context_text
        self._inject_context_ledger(ctx)

    def _inject_skill_context(self, ctx: NodeContext) -> None:
        if self.skill_retriever is None:
            return
        self.skill_retriever.inject(ctx.state, node=ctx.node, metadata=ctx.metadata)
        self._record_skill_trace(
            ctx,
            "skill_retrieved",
            {"skills": [{"skill_id":item.get("skill_id"),"version":item.get("version"),"source_type":(item.get("skill") or {}).get("source_type")} for item in ctx.state.get(SKILL_CONTEXT_KEY, [])]},
        )

    def _record_skill_trace(
        self,
        ctx: NodeContext,
        event: str,
        payload: Dict[str, Any],
    ) -> None:
        if self.skill_trace_store is None:
            return
        self.skill_trace_store.append(
            SkillTraceEvent(
                run_id=self._run_id_from_state(ctx.state),
                node=ctx.node,
                step=ctx.step,
                event=event,
                payload=payload,
            )
        )

    def _update_context_on_step_start(
        self, step: int, frontier: List[str], state: Dict[str, Any]
    ) -> None:
        if self.context_ledger is None:
            return
        run_id = self._run_id_from_state(state)
        ledger = self.context_ledger.on_step_start(
            run_id=run_id,
            step=step,
            frontier=list(frontier),
            state=state,
        )
        state[CONTEXT_LEDGER_KEY] = ledger.to_dict()

    def _check_context_budget(self, state: Dict[str, Any]) -> None:
        if self.context_budget is None:
            return
        decision = self.context_budget.check(state)
        if self.context_ledger is not None:
            run_id = self._run_id_from_state(state)
            ledger = self.context_ledger.load_or_create(run_id, state)
            ledger.budget = self.context_budget.budget_from_decision(decision)
            self.context_ledger.save(ledger)
            state[CONTEXT_LEDGER_KEY] = ledger.to_dict()
        if not decision.allowed:
            state[RUN_STATUS_KEY] = BUDGET_PAUSED
            state[PAUSE_REASON_KEY] = decision.pause_reason

    def _update_context_on_node_end(
        self,
        ctx: NodeContext,
        update: Optional[Dict[str, Any]],
        evaluation: EvaluationResult,
    ) -> None:
        if self.context_ledger is None:
            return
        run_id = self._run_id_from_state(ctx.state)
        ledger = self.context_ledger.on_node_end(
            run_id=run_id,
            node=ctx.node,
            step=ctx.step,
            update=update,
            state=ctx.state,
            verified=evaluation.passed,
        )
        if not evaluation.passed:
            message = "; ".join(evaluation.findings) or "post-execution validation failed"
            ledger.failure_summaries.append(
                FailureSummary(
                    node=ctx.node,
                    step=ctx.step,
                    error_type="EvaluationFailed",
                    message=message,
                    recoverable=evaluation.retryable,
                )
            )
            self.context_ledger.save(ledger)
        drift = self.drift_detector.detect(ledger, current_node=ctx.node)
        ctx.state[DRIFT_RESULT_KEY] = drift.to_dict()
        self._record_todo_update_suggestion(ledger, drift)
        if drift.drifted:
            ledger.failure_summaries.append(
                FailureSummary(
                    node=ctx.node,
                    step=ctx.step,
                    error_type="ContextDrift",
                    message="; ".join(drift.reasons),
                    recoverable=True,
                )
            )
            self.context_ledger.save(ledger)
        ctx.state[CONTEXT_LEDGER_KEY] = ledger.to_dict()

    def _record_todo_update_suggestion(self, ledger, drift) -> None:
        judge_data = drift.metadata.get("task_drift_judge_result") or {}
        if judge_data.get("decision") != "todo_update_needed":
            return
        reason = str(judge_data.get("reason") or "todo update suggested")
        updates = judge_data.get("todo_updates") or []
        if getattr(self.drift_detector, "todo_update_mode", "suggest") == "auto":
            applied = self._apply_todo_updates(ledger, updates, reason=reason)
            if applied and self.context_ledger is not None:
                self.context_ledger.save(ledger)
            return
        message = f"TodoUpdateSuggested: {reason}"
        if updates:
            message = f"{message} updates={json.dumps(updates, ensure_ascii=False, default=str)}"
        if message not in ledger.open_questions:
            ledger.open_questions.append(message)
            if self.context_ledger is not None:
                self.context_ledger.save(ledger)

    def _apply_todo_updates(self, ledger, updates, *, reason: str) -> bool:
        manager = TodoManager(ledger)
        applied = False
        for update in updates:
            if not isinstance(update, dict):
                continue
            action = str(update.get("action", "")).lower()
            todo_id = str(update.get("todo_id", ""))
            content = str(update.get("content", ""))
            update_reason = str(update.get("reason") or reason)
            try:
                if action == "insert" and content:
                    manager.insert_todo(content, after_id=todo_id, reason=update_reason)
                    applied = True
                elif action == "replace" and todo_id and content:
                    manager.replace_todo(todo_id, content, reason=update_reason)
                    applied = True
                elif action == "cancel" and todo_id:
                    manager.cancel_todo(todo_id, reason=update_reason)
                    applied = True
                elif action == "complete" and todo_id:
                    manager.update_todo_status(todo_id, "completed", reason=update_reason)
                    applied = True
                elif action == "select" and todo_id:
                    manager.select_active_todo(todo_id, reason=update_reason)
                    applied = True
            except (KeyError, ValueError) as exc:
                ledger.open_questions.append(f"TodoUpdateFailed: {exc}")
        return applied

    def _update_context_on_node_error(
        self, ctx: NodeContext, error: BaseException, *, recoverable: bool
    ) -> None:
        if self.context_ledger is None:
            return
        run_id = self._run_id_from_state(ctx.state)
        ledger = self.context_ledger.on_node_error(
            run_id=run_id,
            node=ctx.node,
            step=ctx.step,
            error=error,
            state=ctx.state,
            recoverable=recoverable,
        )
        ctx.state[CONTEXT_LEDGER_KEY] = ledger.to_dict()

    def _inject_context_ledger(self, ctx: NodeContext) -> None:
        if self.context_ledger is None:
            return
        run_id = self._run_id_from_state(ctx.state)
        ledger = self.context_ledger.load_or_create(run_id, ctx.state)
        ctx.state[CONTEXT_LEDGER_KEY] = ledger.to_dict()
        injection = self.context_injector.build(
            ledger=ledger,
            node=ctx.node,
            metadata={**ctx.metadata, **self._node_metadata(ctx.node)},
        )
        ctx.state[CONTEXT_INJECTION_KEY] = injection.to_dict()
        ctx.state[CONTEXT_INJECTION_TEXT_KEY] = injection.to_text()

    def _memory_context(self, ctx: NodeContext) -> MemoryContext:
        return self._memory_context_from_state(ctx.state, node=ctx.node, step=ctx.step)

    def _memory_context_from_state(
        self, state: Dict[str, Any], *, node: str, step: int
    ) -> MemoryContext:
        existing = state.get(MEMORY_CONTEXT_KEY) or {}
        task_id = (
            state.get("task_id")
            or state.get("run_id")
            or existing.get("task_id")
            or self.project_rules.get("task_id")
            or "default-task"
        )
        project_id = (
            state.get("project_id")
            or existing.get("project_id")
            or self.project_rules.get("project_id")
            or "default-project"
        )
        global_id = (
            state.get("global_id")
            or existing.get("global_id")
            or self.project_rules.get("global_id")
            or "default"
        )
        working_id = f"{task_id}:{step}:{node}" if step else f"{task_id}:{node}"
        return MemoryContext(
            working_id=working_id,
            task_id=str(task_id),
            project_id=str(project_id),
            global_id=str(global_id),
        )

    def _run_id_from_state(self, state: Dict[str, Any]) -> str:
        existing = state.get(CONTEXT_LEDGER_KEY) or {}
        return str(
            state.get("run_id")
            or state.get("task_id")
            or existing.get("run_id")
            or self.project_rules.get("run_id")
            or self.project_rules.get("task_id")
            or "default-run"
        )

    def _memory_query(self, ctx: NodeContext) -> str:
        parts: List[str] = [ctx.node]
        for key in ("input", "task", "goal", "query"):
            value = ctx.state.get(key)
            if value:
                parts.append(self._stringify(value))
        messages = ctx.state.get("messages") or []
        if messages:
            parts.append(self._stringify(messages[-3:]))
        return "\n".join(parts)

    def _format_context_pack(self, items: List[MemoryItem]) -> str:
        if not items:
            return ""
        lines = ["Relevant memory:"]
        for idx, item in enumerate(items, 1):
            summary = item.summary or self._stringify(item.content)
            ref = f" (ref: {item.raw_ref})" if item.raw_ref else ""
            lines.append(f"{idx}. [{item.scope.value}] {summary}{ref}")
        return "\n".join(lines)

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)

    def _node_metadata(self, node: str) -> Dict[str, Any]:
        for item in self.graph_view.get("nodes", []):
            if item.get("name") == node:
                metadata = item.get("metadata") or {}
                return dict(metadata)
        return {}

    def _scheduling_rules(self) -> Dict[str, Any]:
        rules = self.project_rules.get("scheduling") or {}
        return dict(rules) if isinstance(rules, dict) else {}

    def _apply_security_controls(
        self, ctx: NodeContext, allocation: ResourceAllocation
    ) -> None:
        split = allocation.metadata.get("model_split") or []
        needs_redaction = any(step.get("name") == "trusted_redaction" for step in split)
        if not needs_redaction:
            return
        payload = self._security_payload(ctx)
        result = self.redactor.redact(payload)
        audit = self.audit_builder.build(
            node=ctx.node,
            result=result,
            approved=bool(ctx.state.get("cloud_audit_approved") or ctx.state.get("human_approved")),
            reviewer=str(ctx.state.get("audit_reviewer") or ""),
            reason=str(ctx.state.get("audit_reason") or "trusted workspace preflight"),
            metadata={
                "step": ctx.step,
                "allocation_tier": allocation.tier.value,
                "allocation_endpoint": allocation.endpoint,
            },
        )
        ctx.state[REDACTION_RESULT_KEY] = result.to_dict(include_payload=True)
        ctx.state[AUDIT_PACK_KEY] = audit.to_dict()
        allocation.metadata["redaction"] = result.to_dict(include_payload=False)
        allocation.metadata["audit_pack"] = audit.to_dict()

    def _security_payload(self, ctx: NodeContext) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "node": ctx.node,
            "metadata": {
                key: value
                for key, value in ctx.metadata.items()
                if key in {"description", "sys_prompt", "task_type"}
            },
            "state": {},
        }
        for key in ("input", "task", "goal", "query", "messages"):
            if key in ctx.state:
                payload["state"][key] = ctx.state[key]
        return payload
