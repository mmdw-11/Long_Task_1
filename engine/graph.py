"""核心图引擎（仿 LangGraph）。

提供两个核心类：

- :class:`StateGraph`  —— 构建期。用于声明节点与边（静态边 / 条件边），
  对应 LangGraph 的 ``StateGraph``。
- :class:`CompiledGraph` —— 运行期。由 ``StateGraph.compile()`` 生成，
  负责按超步（superstep）方式执行图。

执行模型（Pregel 风格的 BFS 超步）：

1. 从 ``START`` 出发的边确定入口节点，构成初始"前沿(frontier)"。
2. 每个超步：并发执行当前前沿的所有节点，各自返回部分状态更新。
3. 所有更新按节点顺序归并进共享状态。
4. 依据每个已执行节点的出边（静态 + 条件）计算下一个前沿；
   相同目标会自动去重（天然支持 fan-in 汇聚）。
5. 指向 ``END`` 的路径终止；前沿为空或达到步数上限时结束。

该模型天然支持：顺序、分支（条件边）、并行扇出、循环（带步数上限保护）。
"""

from __future__ import annotations

import asyncio
import copy
import inspect
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from .constants import END, START
from .checkpoint import GraphCheckpoint, GraphCheckpointStore
from .failure import (
    FAILURES_KEY, RECOVERY_TRACE_KEY, SIDE_EFFECT_JOURNAL_KEY, FailureRecord,
)
from .hooks import ExecutionHook, NodeContext
from .modules.communication import COMMUNICATION_STATE_KEY
from .modules.reasoning import COMPLETED_SUBTASKS_KEY, PLAN_IR_KEY, PLAN_REVISIONS_KEY, PLAN_VALIDATION_KEY
from .modules.context import (
    CONTEXT_INJECTION_KEY,
    CONTEXT_INJECTION_TEXT_KEY,
    CONTEXT_LEDGER_KEY,
)
from .modules.context import RUN_STATUS_KEY
from .modules.flow import FlowDecision
from .modules.skills import SKILL_CONTEXT_KEY, SKILL_CONTEXT_TEXT_KEY
from .node import Node, NodeCallable, NodeType
from .state import GraphState, Reducer


class GraphExecutionError(RuntimeError):
    """图执行期错误（如结构非法、超过步数上限、节点抛错）。"""


# 条件路由函数：接收状态快照，返回下一节点名 / 名列表 / END / 供 path_map 映射的 key
ConditionFn = Callable[[Dict[str, Any]], Union[str, List[str], Awaitable[Union[str, List[str]]]]]
LoopConditionFn = Callable[[Dict[str, Any]], Union[bool, Awaitable[bool]]]


class _ConditionalEdge:
    """一条条件边。"""

    def __init__(
        self,
        source: str,
        condition: ConditionFn,
        path_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.source = source
        self.condition = condition
        self.path_map = path_map or {}

    async def resolve(self, state: Dict[str, Any]) -> List[str]:
        """执行路由函数，解析出实际的目标节点名列表。"""
        result = self.condition(state)
        if inspect.isawaitable(result):
            result = await result
        raw = result if isinstance(result, list) else [result]
        targets: List[str] = []
        for item in raw:
            # 若命中 path_map 则映射，否则将返回值本身作为节点名。
            targets.append(self.path_map.get(item, item))
        return targets

    def possible_targets(self) -> List[str]:
        """静态可视化用：返回该条件边可能到达的目标集合。"""
        return list(dict.fromkeys(self.path_map.values())) if self.path_map else []


class StateGraph:
    """图构建器（构建期）。

    :param schema: 字段 -> reducer 映射，定义状态各字段的归并策略。
    """

    def __init__(self, schema: Optional[Dict[str, Reducer]] = None) -> None:
        self.schema: Dict[str, Reducer] = dict(schema or {})
        self.nodes: Dict[str, Node] = {}
        # 静态边：source -> [target, ...]
        self.edges: Dict[str, List[str]] = {}
        # 条件边：source -> _ConditionalEdge
        self.conditional_edges: Dict[str, _ConditionalEdge] = {}

    # ------------------------------------------------------------------ #
    # 节点
    # ------------------------------------------------------------------ #
    def add_node(
        self,
        name: str,
        func: NodeCallable,
        node_type: NodeType = NodeType.FUNCTION,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "StateGraph":
        """新增一个节点。返回 self 以支持链式调用。"""
        if name in (START, END):
            raise ValueError(f"{name!r} 是保留的哨兵节点名，不能用作节点名")
        if name in self.nodes:
            raise ValueError(f"节点 {name!r} 已存在")
        self.nodes[name] = Node(name, func, node_type, metadata)
        return self

    def add_node_object(self, node: Node) -> "StateGraph":
        """直接加入一个已构造好的 Node 对象。"""
        if node.name in self.nodes:
            raise ValueError(f"节点 {node.name!r} 已存在")
        self.nodes[node.name] = node
        return self

    # ------------------------------------------------------------------ #
    # 边
    # ------------------------------------------------------------------ #
    def add_edge(self, start: str, end: str) -> "StateGraph":
        """新增一条静态边 start -> end。

        ``start`` 可为 ``START``；``end`` 可为 ``END``。
        同一 source 可以拥有多条静态边（并行扇出）。
        """
        self._validate_endpoint(start, is_source=True)
        self._validate_endpoint(end, is_source=False)
        self.edges.setdefault(start, [])
        if end not in self.edges[start]:
            self.edges[start].append(end)
        return self

    def add_conditional_edges(
        self,
        source: str,
        condition: ConditionFn,
        path_map: Optional[Dict[str, str]] = None,
    ) -> "StateGraph":
        """新增条件边。

        :param source: 源节点名。
        :param condition: 路由函数，接收状态快照，返回目标节点名 / 名列表 /
            ``END`` / 供 ``path_map`` 映射的 key。
        :param path_map: 可选，将路由函数返回的 key 映射为真实节点名。
        """
        self._validate_endpoint(source, is_source=True)
        if source in self.conditional_edges:
            raise ValueError(f"节点 {source!r} 已存在条件边")
        self.conditional_edges[source] = _ConditionalEdge(source, condition, path_map)
        return self

    def add_loop(
        self,
        source: str,
        condition: LoopConditionFn,
        *,
        loop_target: Optional[str] = None,
        exit_target: str = END,
    ) -> "StateGraph":
        """Add a boolean-controlled loop edge.

        If ``condition(state)`` returns True, execution goes to ``loop_target``
        (defaults to ``source``). Otherwise execution goes to ``exit_target``
        (defaults to ``END``).
        """
        target = loop_target or source
        self._validate_endpoint(source, is_source=True)
        self._validate_endpoint(target, is_source=False)
        self._validate_endpoint(exit_target, is_source=False)

        async def _route(state: Dict[str, Any]) -> str:
            result = condition(state)
            if inspect.isawaitable(result):
                result = await result
            return "loop" if result else "exit"

        return self.add_conditional_edges(
            source,
            _route,
            path_map={"loop": target, "exit": exit_target},
        )

    def set_entry_point(self, name: str) -> "StateGraph":
        """设置入口节点，等价于 ``add_edge(START, name)``。"""
        return self.add_edge(START, name)

    def set_finish_point(self, name: str) -> "StateGraph":
        """设置结束节点，等价于 ``add_edge(name, END)``。"""
        return self.add_edge(name, END)

    # ------------------------------------------------------------------ #
    # 校验与编译
    # ------------------------------------------------------------------ #
    def _validate_endpoint(self, name: str, is_source: bool) -> None:
        if name in (START, END):
            if name == END and is_source:
                raise ValueError("END 不能作为边的起点")
            if name == START and not is_source:
                raise ValueError("START 不能作为边的终点")
            return
        if name not in self.nodes:
            raise ValueError(f"节点 {name!r} 不存在，请先 add_node")

    def compile(self) -> "CompiledGraph":
        """校验结构并生成可执行的 CompiledGraph。"""
        if START not in self.edges or not self.edges[START]:
            raise GraphExecutionError("图缺少入口：请通过 set_entry_point 或 add_edge(START, ...) 指定")
        # 校验条件边可能目标存在
        for cond in self.conditional_edges.values():
            for tgt in cond.possible_targets():
                if tgt not in self.nodes and tgt != END:
                    raise GraphExecutionError(
                        f"条件边 {cond.source!r} 的 path_map 目标 {tgt!r} 不是已知节点"
                    )
        return CompiledGraph(self)

    # ------------------------------------------------------------------ #
    # 序列化（前端可视化）
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """导出图结构（节点 + 边），供前端渲染。"""
        static_edges: List[Dict[str, Any]] = []
        for src, targets in self.edges.items():
            for tgt in targets:
                static_edges.append({"source": str(src), "target": str(tgt), "conditional": False})
        cond_edges: List[Dict[str, Any]] = []
        for src, cond in self.conditional_edges.items():
            for tgt in cond.possible_targets() or ["<dynamic>"]:
                cond_edges.append({"source": str(src), "target": str(tgt), "conditional": True})
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": static_edges + cond_edges,
        }


class CompiledGraph:
    """已编译、可执行的图（运行期）。"""

    def __init__(
        self,
        builder: StateGraph,
        recursion_limit: int = 50,
        hooks: Optional[ExecutionHook] = None,
    ) -> None:
        self._builder = builder
        self.nodes = builder.nodes
        self.edges = builder.edges
        self.conditional_edges = builder.conditional_edges
        self.recursion_limit = recursion_limit
        # 执行钩子：默认基类实例为纯 no-op / passthrough，保证零行为变化。
        self.hooks: ExecutionHook = hooks or ExecutionHook()

    # ------------------------------------------------------------------ #
    # 执行入口
    # ------------------------------------------------------------------ #
    async def ainvoke(
        self,
        input: Optional[Dict[str, Any]] = None,
        recursion_limit: Optional[int] = None,
        *,
        run_id: Optional[str] = None,
        checkpoint_store: Optional[GraphCheckpointStore] = None,
        resume_from: Optional[GraphCheckpoint | str] = None,
    ) -> Dict[str, Any]:
        """异步执行整张图，返回最终状态字典。"""
        final_state: Dict[str, Any] = {}
        async for event in self.astream(
            input,
            recursion_limit,
            run_id=run_id,
            checkpoint_store=checkpoint_store,
            resume_from=resume_from,
        ):
            if event.get("type") == "final":
                final_state = event["state"]
        return final_state

    def invoke(
        self,
        input: Optional[Dict[str, Any]] = None,
        recursion_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """同步执行（内部创建事件循环）。"""
        return asyncio.run(self.ainvoke(input, recursion_limit))

    async def astream(
        self,
        input: Optional[Dict[str, Any]] = None,
        recursion_limit: Optional[int] = None,
        *,
        run_id: Optional[str] = None,
        checkpoint_store: Optional[GraphCheckpointStore] = None,
        resume_from: Optional[GraphCheckpoint | str] = None,
    ):
        """流式执行，逐个产出事件（供前端实时展示执行过程）。

        事件类型：
        - ``{"type": "node_start", "node": name}``
        - ``{"type": "node_end", "node": name, "update": delta}``
        - ``{"type": "final", "state": {...}}``
        """
        limit = recursion_limit or self.recursion_limit
        if resume_from is not None:
            checkpoint = (
                checkpoint_store.load(str(run_id or (input or {}).get("run_id")), str(resume_from))
                if isinstance(resume_from, str) and checkpoint_store is not None
                else resume_from
            )
            if not isinstance(checkpoint, GraphCheckpoint):
                raise GraphExecutionError("resume_from requires GraphCheckpoint or checkpoint id with store")
            state = GraphState(self._builder.schema, checkpoint.state)
            frontier = self._normalize_targets(checkpoint.frontier)
            step = int(checkpoint.step)
            run_id = checkpoint.run_id
        else:
            state = GraphState(self._builder.schema, input or {})
            frontier = self._normalize_targets(self.edges.get(START, []))
            step = 0
            run_id = run_id or str((input or {}).get("run_id") or (input or {}).get("task_id") or "default-run")
        while frontier:
            if checkpoint_store is not None:
                checkpoint_store.save(
                    run_id=str(run_id),
                    step=step,
                    frontier=list(frontier),
                    state=state.snapshot(),
                    status="running",
                    checkpoint_id=f"step_{step:04d}_before",
                    metadata=self._checkpoint_metadata(state.snapshot()),
                )
            step += 1
            if step > limit:
                raise GraphExecutionError(
                    f"超过最大超步数 {limit}，可能存在无终止的循环"
                )

            snapshot = state.snapshot()
            self.hooks.on_step_start(step, list(frontier), snapshot)
            runtime_keys = (
                PLAN_IR_KEY, PLAN_VALIDATION_KEY, PLAN_REVISIONS_KEY,
                COMPLETED_SUBTASKS_KEY,
                RECOVERY_TRACE_KEY, SIDE_EFFECT_JOURNAL_KEY,
                RUN_STATUS_KEY, "__pause_reason__",
            )
            runtime_patch = {key: snapshot[key] for key in runtime_keys if key in snapshot}
            if runtime_patch:
                state.update(runtime_patch)
            for hook_event in self.hooks.drain_events():
                yield hook_event
            if snapshot.get(RUN_STATUS_KEY):
                break

            # 流控：逐节点决定 执行 / 跳过 / 延迟。
            decisions: Dict[str, FlowDecision] = {}
            node_snapshots: Dict[str, Dict[str, Any]] = {}
            for name in frontier:
                ctx = NodeContext(
                    node=name,
                    step=step,
                    state=copy.deepcopy(snapshot),
                    metadata=dict(self.nodes[name].metadata),
                )
                decisions[name] = self.hooks.on_node_start(ctx)
                node_snapshots[name] = ctx.state
            executable = [n for n in frontier if decisions[n] == FlowDecision.EXECUTE]
            skipped = [n for n in frontier if decisions[n] == FlowDecision.SKIP]
            deferred = [n for n in frontier if decisions[n] == FlowDecision.DEFER]

            for name in executable:
                metadata = self.nodes[name].metadata
                yield {"type": "node_start", "node": name, "parent_node_id": metadata.get("parent_id"), "iteration": snapshot.get("loop_index"), "batch_index": snapshot.get("batch_index"), "input": snapshot.get("input")}

            results = await asyncio.gather(
                *[self._run_node(name, step, node_snapshots.get(name, snapshot)) for name in executable],
                return_exceptions=False,
            )

            # 按执行顺序归并更新，处理失败与恢复。
            executed_ok: List[str] = []
            recovery_targets: List[str] = []
            for name, update, error in results:
                if error is not None:
                    err_ctx = NodeContext(
                        node=name,
                        step=step,
                        state=state.snapshot(),
                        metadata=dict(self.nodes[name].metadata),
                    )
                    repair = self.hooks.on_node_error(err_ctx, error)
                    state.update({
                        key: value for key, value in err_ctx.state.items()
                        if key in {
                            RECOVERY_TRACE_KEY, SIDE_EFFECT_JOURNAL_KEY, RUN_STATUS_KEY,
                            "__pause_reason__", "__failure_context__", "__degrade_mode__",
                            "__resource_degradation__", "__retry_policy__",
                            "__fallback_binding__", "__compensation_pending__",
                            "__checkpoint_resume_request__",
                        }
                    })
                    for hook_event in self.hooks.drain_events():
                        yield hook_event
                    if repair is None:
                        raise GraphExecutionError(
                            f"节点 {name!r} 执行失败：{error}"
                        ) from error
                    # 恢复路径：记录失败轨迹并将修复目标并入下一前沿。
                    rec = FailureRecord(
                        node=name,
                        error_type=type(error).__name__,
                        message=str(error),
                        step=step,
                        kind=str((err_ctx.state.get("__failure_context__") or {}).get("kind") or "agent"),
                        severity=str((err_ctx.state.get("__failure_context__") or {}).get("severity") or "critical"),
                        attempt=int((err_ctx.state.get("__failure_context__") or {}).get("attempt") or 1),
                    )
                    state.update({FAILURES_KEY: [rec.to_dict()]})
                    for tgt in repair:
                        if tgt != END and tgt not in recovery_targets:
                            recovery_targets.append(tgt)
                    yield {"type": "node_end", "node": name, "update": {FAILURES_KEY: [rec.to_dict()]}}
                    continue
                update = _sanitize_node_update(update)
                pre_update_state = state.snapshot()
                end_state = copy.deepcopy(pre_update_state)
                for key in (
                    CONTEXT_LEDGER_KEY,
                    CONTEXT_INJECTION_KEY,
                    CONTEXT_INJECTION_TEXT_KEY,
                    SKILL_CONTEXT_KEY,
                    SKILL_CONTEXT_TEXT_KEY,
                    "__context_drift__",
                    "__evaluation__",
                    COMMUNICATION_STATE_KEY,
                    RECOVERY_TRACE_KEY,
                    SIDE_EFFECT_JOURNAL_KEY,
                    COMPLETED_SUBTASKS_KEY,
                ):
                    if key in node_snapshots.get(name, {}):
                        end_state[key] = node_snapshots[name][key]
                end_ctx = NodeContext(
                    node=name,
                    step=step,
                    state=end_state,
                    metadata=dict(self.nodes[name].metadata),
                )
                self.hooks.on_node_end(end_ctx, update)
                hook_events = self.hooks.drain_events()
                validation_targets = self.hooks.handle_validation_failure(end_ctx, update)
                runtime_context_update = {}
                for key in (
                    CONTEXT_LEDGER_KEY,
                    CONTEXT_INJECTION_KEY,
                    CONTEXT_INJECTION_TEXT_KEY,
                    SKILL_CONTEXT_KEY,
                    SKILL_CONTEXT_TEXT_KEY,
                    "__context_drift__",
                    "__evaluation__",
                    COMMUNICATION_STATE_KEY,
                    RECOVERY_TRACE_KEY,
                    SIDE_EFFECT_JOURNAL_KEY,
                    COMPLETED_SUBTASKS_KEY,
                ):
                    if key in end_ctx.state:
                        runtime_context_update[key] = end_ctx.state[key]
                if runtime_context_update:
                    state.update(runtime_context_update)
                if validation_targets is not None:
                    state.update(
                        {
                            key: value
                            for key, value in end_ctx.state.items()
                            if key.startswith("__")
                        }
                    )
                    if validation_targets:
                        for tgt in validation_targets:
                            if tgt != END and tgt not in recovery_targets:
                                recovery_targets.append(tgt)
                        yield {"type": "node_end", "node": name, "update": runtime_context_update}
                        continue
                    else:
                        frontier = []
                        next_frontier = []
                        break
                state.update(update)
                executed_ok.append(name)
                event_state = state.snapshot()
                yield {"type": "node_end", "node": name, "update": update or {}, "parent_node_id": self.nodes[name].metadata.get("parent_id"), "iteration": event_state.get("loop_index"), "batch_index": event_state.get("batch_index"), "output": update or {}}
                for hook_event in hook_events:
                    yield hook_event

            # 计算下一个前沿。
            next_frontier: List[str] = []
            current_snapshot = state.snapshot()
            # 已执行成功与被跳过的节点：正常计算后继（并允许路由策略覆盖）。
            for name in executed_ok + skipped:
                base_succ = await self._successors(name, current_snapshot)
                overridden = self.hooks.resolve_successors(name, base_succ, current_snapshot)
                succ_list = overridden if overridden is not None else base_succ
                yield {
                    "type": "route",
                    "node": name,
                    "candidates": list(base_succ),
                    "targets": list(succ_list),
                }
                for hook_event in self.hooks.drain_events():
                    yield hook_event
                for succ in succ_list:
                    if succ == END:
                        continue  # 该路径结束
                    if succ not in next_frontier:
                        next_frontier.append(succ)
            # 被延迟的节点：留待下一超步重新评估。
            for name in deferred:
                if name not in next_frontier:
                    next_frontier.append(name)
            # 恢复目标：并入下一前沿。
            for tgt in recovery_targets:
                if tgt not in next_frontier:
                    next_frontier.append(tgt)
            frontier = next_frontier

        if checkpoint_store is not None:
            checkpoint_store.save(
                run_id=str(run_id),
                step=step,
                frontier=[],
                state=state.snapshot(),
                status="completed",
                checkpoint_id="final",
                metadata=self._checkpoint_metadata(state.snapshot()),
            )
        yield {"type": "final", "state": state.to_dict()}

    async def _run_node(
        self, name: str, step: int, snapshot: Dict[str, Any]
    ):
        """执行单个节点（含资源申请/释放），返回 (name, update, error)。"""
        ctx = NodeContext(
            node=name,
            step=step,
            state=snapshot,
            metadata=dict(self.nodes[name].metadata),
        )
        allocation = self.hooks.acquire_resource(ctx)
        try:
            await self.hooks.before_node_invoke(ctx)
            invocation = self.nodes[name].invoke(ctx.state)
            timeout = float(ctx.metadata.get("timeout_seconds") or 0)
            update = await asyncio.wait_for(invocation, timeout=timeout) if timeout > 0 else await invocation
            return name, update, None
        except Exception as exc:  # noqa: BLE001 - 交由恢复策略处理
            return name, None, exc
        finally:
            self.hooks.release_resource(ctx, allocation)

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    async def _successors(self, name: str, state: Dict[str, Any]) -> List[str]:
        """返回节点 name 的后继（合并静态边与条件边）。"""
        successors: List[str] = list(self.edges.get(name, []))
        cond = self.conditional_edges.get(name)
        if cond is not None:
            for tgt in await cond.resolve(state):
                if tgt != END and tgt not in self.nodes:
                    raise GraphExecutionError(
                        f"节点 {name!r} 的条件边解析到未知目标 {tgt!r}"
                    )
                if tgt not in successors:
                    successors.append(tgt)
        return successors

    def _checkpoint_metadata(self, state: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(self.hooks.checkpoint_metadata())
        plan = state.get(PLAN_IR_KEY) or {}
        validation = state.get(PLAN_VALIDATION_KEY) or {}
        metadata.update({
            "verified_plan_revision": plan.get("revision") if validation.get("passed") else None,
            "completed_subtasks": list(state.get(COMPLETED_SUBTASKS_KEY) or []),
            "recovery_episode_count": len(state.get(RECOVERY_TRACE_KEY) or []),
            "side_effect_journal_size": len(state.get(SIDE_EFFECT_JOURNAL_KEY) or []),
        })
        return metadata

    @staticmethod
    def _normalize_targets(targets: List[str]) -> List[str]:
        # 去重且保持顺序，剔除 END（入口不应直接是 END）。
        seen: List[str] = []
        for t in targets:
            if t != END and t not in seen:
                seen.append(t)
        return seen

    def to_dict(self) -> Dict[str, Any]:
        return self._builder.to_dict()


def _sanitize_node_update(update: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Drop deprecated self-attested validation markers from node output."""
    if not isinstance(update, dict):
        return update
    if "__parent_validated__" not in update:
        return update
    sanitized = dict(update)
    sanitized.pop("__parent_validated__", None)
    return sanitized
