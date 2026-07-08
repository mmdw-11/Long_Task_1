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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .failure import FAILURES_KEY, FailureRecord, FailureTrace
from .modules.flow import FlowController, FlowDecision, NoOpFlowController
from .modules.memory import MemoryStore, NoOpMemoryStore
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
from .modules.scheduling import (
    NoOpResourceScheduler,
    ResourceAllocation,
    ResourceRequest,
    ResourceScheduler,
)


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
        project_rules: Optional[Dict[str, Any]] = None,
        graph_view: Optional[Dict[str, Any]] = None,
        extra_hooks: Optional[List[ExecutionHook]] = None,
    ) -> None:
        self.memory = memory or NoOpMemoryStore()
        self.router = router or NoOpRouter()
        self.routing_policy = routing_policy or PassthroughRoutingPolicy()
        self.flow_controller = flow_controller or NoOpFlowController()
        self.recovery_strategy = recovery_strategy or NoOpRecoveryStrategy()
        self.scheduler = scheduler or NoOpResourceScheduler()
        self.project_rules = project_rules or {}
        self.graph_view = graph_view or {}
        self.extra_hooks: List[ExecutionHook] = list(extra_hooks or [])

    # ------------------------------------------------------------------ #
    # 扩展点实现（桥接到各模块）
    # ------------------------------------------------------------------ #
    def on_step_start(self, step: int, frontier: List[str], state: Dict[str, Any]) -> None:
        for h in self.extra_hooks:
            h.on_step_start(step, frontier, state)

    def on_node_start(self, ctx: NodeContext) -> FlowDecision:
        deps = self.flow_controller.resolve_dependencies(ctx.node, self.graph_view)
        deps_satisfied = self._deps_satisfied(deps, ctx.state)
        decision = self.flow_controller.decide(
            ctx.node, state=ctx.state, deps_satisfied=deps_satisfied
        )
        for h in self.extra_hooks:
            h.on_node_start(ctx)
        return decision

    def acquire_resource(self, ctx: NodeContext) -> Optional[ResourceAllocation]:
        return self.scheduler.acquire(ResourceRequest(node=ctx.node))

    def release_resource(self, ctx: NodeContext, allocation: Optional[ResourceAllocation]) -> None:
        if allocation is not None:
            self.scheduler.release(allocation)

    def on_node_end(self, ctx: NodeContext, update: Optional[Dict[str, Any]]) -> None:
        # 将节点产出写入记忆（NoOp 桩会丢弃）。
        if update:
            self.memory.append(update, node=ctx.node, step=ctx.step)
        for h in self.extra_hooks:
            h.on_node_end(ctx, update)

    def on_node_error(self, ctx: NodeContext, error: BaseException) -> Optional[List[str]]:
        trace = FailureTrace.from_state(ctx.state)
        trace.record(ctx.node, error, step=ctx.step)
        plan: RepairPlan = self.recovery_strategy.plan(
            trace, node=ctx.node, state=ctx.state
        )
        for h in self.extra_hooks:
            h.on_node_error(ctx, error)
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
                "memory_ctx": self.memory.read(query=node),
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
