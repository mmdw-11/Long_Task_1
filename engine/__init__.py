"""engine

在 AgentScope 之上扩展的、类 LangGraph 的 Agent 图编排框架。

模块划分：
- ``constants``   : START / END 等哨兵节点。
- ``state``       : 图状态（GraphState）与状态归并器（reducer）。
- ``node``        : 节点抽象，可包裹 AgentScope agent、普通函数或子图。
- ``graph``       : 核心图引擎（StateGraph / CompiledGraph），仿照 LangGraph 语义。
- ``adapters``    : 将 AgentScope 的 ReActAgent 等封装为图节点。
- ``orchestrator``: 面向前端可视化的高层管理接口。
- ``server``      : 基于 FastAPI 的 REST 服务。
- ``failure``     : 失败轨迹数据结构（共享基础）。
- ``hooks``       : 执行钩子与钩子管理器（扩展点）。
- ``modules``     : 可插拔扩展模块（记忆 / 动态路由 / 流控 / 恢复策略 / 资源调度）。
"""

from .constants import START, END
from .state import GraphState, add_messages, append_reducer, replace_reducer
from .node import Node, NodeType
from .graph import StateGraph, CompiledGraph, GraphExecutionError
from .orchestrator import Orchestrator, AgentSpec
from .failure import FailureRecord, FailureTrace, FAILURES_KEY
from .hooks import ExecutionHook, HookManager, NodeContext

__all__ = [
    "START",
    "END",
    "GraphState",
    "add_messages",
    "append_reducer",
    "replace_reducer",
    "Node",
    "NodeType",
    "StateGraph",
    "CompiledGraph",
    "GraphExecutionError",
    "Orchestrator",
    "AgentSpec",
    # 失败轨迹
    "FailureRecord",
    "FailureTrace",
    "FAILURES_KEY",
    # 钩子
    "ExecutionHook",
    "HookManager",
    "NodeContext",
]

__version__ = "0.1.0"
