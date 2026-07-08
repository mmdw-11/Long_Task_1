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
import inspect
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from .constants import END, START
from .failure import FAILURES_KEY, FailureRecord
from .hooks import ExecutionHook, NodeContext
from .modules.flow import FlowDecision
from .node import Node, NodeCallable, NodeType
from .state import GraphState, Reducer


class GraphExecutionError(RuntimeError):
    """图执行期错误（如结构非法、超过步数上限、节点抛错）。"""


# 条件路由函数：接收状态快照，返回下一节点名 / 名列表 / END / 供 path_map 映射的 key
ConditionFn = Callable[[Dict[str, Any]], Union[str, List[str], Awaitable[Union[str, List[str]]]]]


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
    ) -> Dict[str, Any]:
        """异步执行整张图，返回最终状态字典。"""
        final_state: Dict[str, Any] = {}
        async for event in self.astream(input, recursion_limit):
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
    ):
        """流式执行，逐个产出事件（供前端实时展示执行过程）。

        事件类型：
        - ``{"type": "node_start", "node": name}``
        - ``{"type": "node_end", "node": name, "update": delta}``
        - ``{"type": "final", "state": {...}}``
        """
        limit = recursion_limit or self.recursion_limit
        state = GraphState(self._builder.schema, input or {})

        # 初始前沿：START 的所有静态目标。
        frontier = self._normalize_targets(self.edges.get(START, []))
        step = 0
        while frontier:
            step += 1
            if step > limit:
                raise GraphExecutionError(
                    f"超过最大超步数 {limit}，可能存在无终止的循环"
                )

            snapshot = state.snapshot()
            self.hooks.on_step_start(step, list(frontier), snapshot)

            # 流控：逐节点决定 执行 / 跳过 / 延迟。
            decisions: Dict[str, FlowDecision] = {}
            for name in frontier:
                ctx = NodeContext(node=name, step=step, state=snapshot)
                decisions[name] = self.hooks.on_node_start(ctx)
            executable = [n for n in frontier if decisions[n] == FlowDecision.EXECUTE]
            skipped = [n for n in frontier if decisions[n] == FlowDecision.SKIP]
            deferred = [n for n in frontier if decisions[n] == FlowDecision.DEFER]

            for name in executable:
                yield {"type": "node_start", "node": name}

            results = await asyncio.gather(
                *[self._run_node(name, step, snapshot) for name in executable],
                return_exceptions=False,
            )

            # 按执行顺序归并更新，处理失败与恢复。
            executed_ok: List[str] = []
            recovery_targets: List[str] = []
            for name, update, error in results:
                if error is not None:
                    err_ctx = NodeContext(node=name, step=step, state=state.snapshot())
                    repair = self.hooks.on_node_error(err_ctx, error)
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
                    )
                    state.update({FAILURES_KEY: [rec.to_dict()]})
                    for tgt in repair:
                        if tgt != END and tgt not in recovery_targets:
                            recovery_targets.append(tgt)
                    yield {"type": "node_end", "node": name, "update": {FAILURES_KEY: [rec.to_dict()]}}
                    continue
                state.update(update)
                end_ctx = NodeContext(node=name, step=step, state=state.snapshot())
                self.hooks.on_node_end(end_ctx, update)
                executed_ok.append(name)
                yield {"type": "node_end", "node": name, "update": update or {}}

            # 计算下一个前沿。
            next_frontier: List[str] = []
            current_snapshot = state.snapshot()
            # 已执行成功与被跳过的节点：正常计算后继（并允许路由策略覆盖）。
            for name in executed_ok + skipped:
                base_succ = await self._successors(name, current_snapshot)
                overridden = self.hooks.resolve_successors(name, base_succ, current_snapshot)
                succ_list = overridden if overridden is not None else base_succ
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

        yield {"type": "final", "state": state.to_dict()}

    async def _run_node(
        self, name: str, step: int, snapshot: Dict[str, Any]
    ):
        """执行单个节点（含资源申请/释放），返回 (name, update, error)。"""
        ctx = NodeContext(node=name, step=step, state=snapshot)
        allocation = self.hooks.acquire_resource(ctx)
        try:
            update = await self.nodes[name].invoke(snapshot)
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
                if tgt not in successors:
                    successors.append(tgt)
        return successors

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
