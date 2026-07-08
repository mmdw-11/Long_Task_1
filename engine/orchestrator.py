"""面向前端可视化的编排管理器。

这是暴露给前端的高层接口层。它用**声明式**的方式管理一批 agent 及其连接关系，
并可随时：

- 序列化为 JSON 供前端渲染（节点 + 层级 + 连线）。
- 编译为可执行的 :class:`CompiledGraph` 并运行。

核心概念：
- ``AgentSpec`` : 一个 agent 的声明（名称、系统提示、模型配置、子 agent 等）。
- ``create_agent``   : 创建一个 agent，返回其 id。
- ``add_sub_agent``  : 给某个 agent 添加子 agent（可多次调用添加多个）。
- ``connect``        : 连接两个 agent（等价于在图中加一条边）。
- ``set_entry``      : 指定入口 agent。
- ``build_graph``    : 依据当前声明构建可执行图。

``build_graph`` 允许注入 ``node_factory`` 来自定义每个 agent 对应的节点实现；
默认使用一个“回显(echo)”工厂输出结构化的流转信息。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .constants import END, START
from .failure import FAILURES_KEY
from .graph import CompiledGraph, StateGraph
from .hooks import HookManager
from .modules.flow import FlowController
from .modules.memory import MemoryStore
from .modules.recovery import RecoveryStrategy
from .modules.routing import Router, RoutingPolicy
from .modules.scheduling import ResourceScheduler
from .node import Node, NodeType
from .state import Reducer, append_reducer


@dataclass
class AgentSpec:
    """一个 agent 的声明式定义。"""

    id: str
    name: str
    sys_prompt: str = ""
    model: str = ""
    description: str = ""
    # 子 agent id 列表（层级关系）。
    children: List[str] = field(default_factory=list)
    # 供业务扩展的附加配置。
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "sys_prompt": self.sys_prompt,
            "model": self.model,
            "description": self.description,
            "children": list(self.children),
            "config": self.config,
        }


@dataclass
class Connection:
    """两个 agent 之间的连线（图中的一条边）。"""

    source: str
    target: str  # 可为另一个 agent id，或 END
    conditional: bool = False
    # 条件边可选：从状态字段取值来路由，或提供命名条件。
    condition_key: Optional[str] = None
    path_map: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "conditional": self.conditional,
            "condition_key": self.condition_key,
            "path_map": self.path_map,
        }


# 节点工厂签名：依据 AgentSpec 生成图节点可调用体。
NodeFactory = Callable[[AgentSpec], Node]


def echo_node_factory(spec: AgentSpec) -> Node:
    """默认节点工厂：生成一个“回显”节点。

    它把上游输入原样记录到状态，方便前端观察数据在图中的流转。
    """

    async def _echo(state: Dict[str, Any]) -> Dict[str, Any]:
        incoming = state.get("input")
        text = f"[{spec.name}] 收到: {incoming}"
        return {
            "input": text,          # 传递给下游
            spec.name: text,        # 记录本节点输出
            "messages": [{"agent": spec.name, "content": text}],
        }

    return Node(
        name=spec.name,
        func=_echo,
        node_type=NodeType.AGENT,
        metadata={"id": spec.id, "model": spec.model, "description": spec.description},
    )


class Orchestrator:
    """Agent 编排管理器（前端友好的高层 API）。"""

    def __init__(self) -> None:
        self._agents: Dict[str, AgentSpec] = {}
        self._connections: List[Connection] = []
        self._entry: Optional[str] = None
        # 状态字段的归并策略：messages 与失败轨迹均使用追加。
        self._schema: Dict[str, Reducer] = {
            "messages": append_reducer,
            FAILURES_KEY: append_reducer,
        }
        # 可插拔扩展模块（默认 None → build_graph 时回退为 NoOp 桩）。
        self._memory: Optional[MemoryStore] = None
        self._router: Optional[Router] = None
        self._routing_policy: Optional[RoutingPolicy] = None
        self._flow_controller: Optional[FlowController] = None
        self._recovery_strategy: Optional[RecoveryStrategy] = None
        self._scheduler: Optional[ResourceScheduler] = None
        self._project_rules: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # 注册可插拔模块（接入或替换默认策略）
    # ------------------------------------------------------------------ #
    def set_memory(self, store: MemoryStore) -> None:
        """注入记忆模块。"""
        self._memory = store

    def set_router(self, router: Router) -> None:
        """注入基于任务类型的静态路由表。"""
        self._router = router

    def set_routing_policy(self, policy: RoutingPolicy) -> None:
        """注入运行时动态路由策略。"""
        self._routing_policy = policy

    def set_flow_controller(self, controller: FlowController) -> None:
        """注入流控模块。"""
        self._flow_controller = controller

    def set_recovery_strategy(self, strategy: RecoveryStrategy) -> None:
        """注入恢复策略模块。"""
        self._recovery_strategy = strategy

    def set_scheduler(self, scheduler: ResourceScheduler) -> None:
        """注入端边云资源调度模块。"""
        self._scheduler = scheduler

    def set_project_rules(self, rules: Dict[str, Any]) -> None:
        """设置项目规则（供动态路由策略读取）。"""
        self._project_rules = dict(rules)

    # ------------------------------------------------------------------ #
    # 创建 / 删除 agent
    # ------------------------------------------------------------------ #
    def create_agent(
        self,
        name: str,
        sys_prompt: str = "",
        model: str = "",
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> str:
        """创建一个 agent，返回其 id。"""
        aid = agent_id or f"agent-{uuid.uuid4().hex[:8]}"
        if aid in self._agents:
            raise ValueError(f"agent id {aid!r} 已存在")
        if any(a.name == name for a in self._agents.values()):
            raise ValueError(f"agent 名称 {name!r} 已存在（名称需唯一，用作图节点名）")
        self._agents[aid] = AgentSpec(
            id=aid,
            name=name,
            sys_prompt=sys_prompt,
            model=model,
            description=description,
            config=config or {},
        )
        return aid

    def remove_agent(self, agent_id: str) -> None:
        """删除 agent，并清理相关连线与层级引用。"""
        self._require(agent_id)
        del self._agents[agent_id]
        self._connections = [
            c for c in self._connections if c.source != agent_id and c.target != agent_id
        ]
        for spec in self._agents.values():
            if agent_id in spec.children:
                spec.children.remove(agent_id)
        if self._entry == agent_id:
            self._entry = None

    def get_agent(self, agent_id: str) -> AgentSpec:
        self._require(agent_id)
        return self._agents[agent_id]

    def list_agents(self) -> List[AgentSpec]:
        return list(self._agents.values())

    # ------------------------------------------------------------------ #
    # 子 agent（层级）
    # ------------------------------------------------------------------ #
    def add_sub_agent(
        self,
        parent_id: str,
        child_id: Optional[str] = None,
        *,
        name: Optional[str] = None,
        sys_prompt: str = "",
        model: str = "",
        description: str = "",
        auto_connect: bool = True,
    ) -> str:
        """给 ``parent_id`` 添加一个子 agent（可多次调用添加多个）。

        两种用法：
        1. 传入已存在的 ``child_id``，把它挂到父节点下。
        2. 不传 ``child_id`` 而传 ``name`` 等参数，则**新建**一个子 agent 并挂载。

        :param auto_connect: 为 True 时自动建立 parent -> child 的边（父委派子）。
        :return: 子 agent 的 id。
        """
        self._require(parent_id)
        if child_id is None:
            if not name:
                raise ValueError("新建子 agent 时必须提供 name")
            child_id = self.create_agent(
                name=name, sys_prompt=sys_prompt, model=model, description=description
            )
        else:
            self._require(child_id)
            if child_id == parent_id:
                raise ValueError("agent 不能作为自己的子 agent")

        parent = self._agents[parent_id]
        if child_id not in parent.children:
            parent.children.append(child_id)

        if auto_connect:
            self.connect(parent_id, child_id)
        return child_id

    # ------------------------------------------------------------------ #
    # 连接 / 入口
    # ------------------------------------------------------------------ #
    def connect(self, source_id: str, target_id: str) -> None:
        """连接两个 agent：source -> target（图中的一条静态边）。

        ``target_id`` 可以是另一个 agent 的 id，也可以是 ``END`` 表示结束。
        """
        self._require(source_id)
        if target_id != END:
            self._require(target_id)
        if any(
            c.source == source_id and c.target == target_id and not c.conditional
            for c in self._connections
        ):
            return  # 已存在相同连线
        self._connections.append(Connection(source_id, target_id))

    def connect_conditional(
        self,
        source_id: str,
        condition_key: str,
        path_map: Dict[str, str],
    ) -> None:
        """建立条件连线：依据状态中 ``condition_key`` 的值路由到不同目标。

        :param path_map: 值 -> 目标 agent id（或 END）的映射。
        """
        self._require(source_id)
        for tgt in path_map.values():
            if tgt != END:
                self._require(tgt)
        self._connections.append(
            Connection(
                source=source_id,
                target="<conditional>",
                conditional=True,
                condition_key=condition_key,
                path_map=dict(path_map),
            )
        )

    def disconnect(self, source_id: str, target_id: str) -> None:
        """删除 source -> target 的静态连线。"""
        self._connections = [
            c
            for c in self._connections
            if not (c.source == source_id and c.target == target_id and not c.conditional)
        ]

    def set_entry(self, agent_id: str) -> None:
        """指定入口 agent。"""
        self._require(agent_id)
        self._entry = agent_id

    # ------------------------------------------------------------------ #
    # 构建可执行图
    # ------------------------------------------------------------------ #
    def build_graph(
        self,
        node_factory: Optional[NodeFactory] = None,
        recursion_limit: int = 50,
    ) -> CompiledGraph:
        """依据当前声明构建并编译为可执行图。

        :param node_factory: 依据 AgentSpec 生成节点的工厂；默认使用 echo 工厂。
        """
        if not self._agents:
            raise ValueError("没有任何 agent，无法构建图")
        entry = self._entry or self._infer_entry()
        if entry is None:
            raise ValueError("无法确定入口 agent，请调用 set_entry 指定")

        factory = node_factory or echo_node_factory
        graph = StateGraph(schema=self._schema)

        # 以 agent 名字作为图节点名（名称唯一由 create_agent 保证）。
        id_to_name = {aid: spec.name for aid, spec in self._agents.items()}
        for spec in self._agents.values():
            graph.add_node_object(factory(spec))

        graph.set_entry_point(id_to_name[entry])

        # 静态边与条件边。
        for conn in self._connections:
            if conn.conditional:
                src_name = id_to_name[conn.source]
                path_map = {
                    key: (END if tgt == END else id_to_name[tgt])
                    for key, tgt in conn.path_map.items()
                }
                ckey = conn.condition_key

                def _make_router(k: str):
                    return lambda state: state.get(k)

                graph.add_conditional_edges(src_name, _make_router(ckey), path_map)
            else:
                src_name = id_to_name[conn.source]
                tgt_name = END if conn.target == END else id_to_name[conn.target]
                graph.add_edge(src_name, tgt_name)

        # 对没有任何出边的节点，自动连到 END，保证路径可终止。
        self._auto_finish(graph, id_to_name)

        compiled = graph.compile()
        compiled.recursion_limit = recursion_limit
        # 组装钩子管理器：将已注册的模块桥接到执行引擎的扩展点。
        compiled.hooks = self._build_hook_manager(graph)
        return compiled

    def _auto_finish(self, graph: StateGraph, id_to_name: Dict[str, str]) -> None:
        has_out = set(graph.edges.keys()) | set(graph.conditional_edges.keys())
        for name in list(graph.nodes.keys()):
            if name not in has_out:
                graph.add_edge(name, END)

    def _infer_entry(self) -> Optional[str]:
        """在未显式指定入口时，推断一个没有任何入边的 agent 作为入口。"""
        targets = {c.target for c in self._connections if not c.conditional}
        for conn in self._connections:
            if conn.conditional:
                targets.update(conn.path_map.values())
        candidates = [aid for aid in self._agents if aid not in targets]
        if len(candidates) == 1:
            return candidates[0]
        # 退而求其次：返回第一个创建的 agent。
        return next(iter(self._agents), None)

    # ------------------------------------------------------------------ #
    # 序列化（前端可视化）
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """导出完整编排结构，供前端渲染。"""
        return {
            "entry": self._entry,
            "agents": [a.to_dict() for a in self._agents.values()],
            "connections": [c.to_dict() for c in self._connections],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Orchestrator":
        """从序列化结构恢复编排（前端保存/加载用）。"""
        orch = cls()
        for a in data.get("agents", []):
            orch._agents[a["id"]] = AgentSpec(
                id=a["id"],
                name=a["name"],
                sys_prompt=a.get("sys_prompt", ""),
                model=a.get("model", ""),
                description=a.get("description", ""),
                children=list(a.get("children", [])),
                config=a.get("config", {}),
            )
        for c in data.get("connections", []):
            orch._connections.append(
                Connection(
                    source=c["source"],
                    target=c["target"],
                    conditional=c.get("conditional", False),
                    condition_key=c.get("condition_key"),
                    path_map=c.get("path_map", {}),
                )
            )
        orch._entry = data.get("entry")
        return orch

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _build_hook_manager(self, graph: StateGraph) -> HookManager:
        """依据已注册模块组装 HookManager（未注册的使用 NoOp 桩）。"""
        kwargs: Dict[str, Any] = {
            "project_rules": self._project_rules,
            "graph_view": graph.to_dict(),
        }
        if self._memory is not None:
            kwargs["memory"] = self._memory
        if self._router is not None:
            kwargs["router"] = self._router
        if self._routing_policy is not None:
            kwargs["routing_policy"] = self._routing_policy
        if self._flow_controller is not None:
            kwargs["flow_controller"] = self._flow_controller
        if self._recovery_strategy is not None:
            kwargs["recovery_strategy"] = self._recovery_strategy
        if self._scheduler is not None:
            kwargs["scheduler"] = self._scheduler
        return HookManager(**kwargs)

    def _require(self, agent_id: str) -> None:
        if agent_id not in self._agents:
            raise KeyError(f"agent id {agent_id!r} 不存在")
