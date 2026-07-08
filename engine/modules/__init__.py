"""可插拔扩展模块集合。

每个子模块提供接口（ABC）+ 空实现桩（NoOp / passthrough），便于独立开发：

- ``memory``     : 记忆模块（四级层级记忆读写与级联检索）。
- ``routing``    : 动态路由模块（静态 Router + 运行时 RoutingPolicy）。
- ``flow``       : 流控模块（依赖解析 + 条件激活，决定执行/跳过/延迟）。
- ``recovery``   : 恢复策略模块（依据失败轨迹决定修复路径）。
- ``scheduling`` : 端边云资源调度模块。
"""

from .memory import MemoryScope, MemoryContext, MemoryItem, MemoryStore, NoOpMemoryStore
from .routing import (
    RoutingDecision,
    Router,
    NoOpRouter,
    RoutingPolicy,
    PassthroughRoutingPolicy,
)
from .flow import FlowDecision, ActivationCondition, FlowController, NoOpFlowController
from .recovery import (
    RecoveryAction,
    RepairPlan,
    RecoveryStrategy,
    NoOpRecoveryStrategy,
)
from .scheduling import (
    ResourceTier,
    ResourceRequest,
    ResourceAllocation,
    ResourceScheduler,
    NoOpResourceScheduler,
)

__all__ = [
    # memory
    "MemoryScope",
    "MemoryContext",
    "MemoryItem",
    "MemoryStore",
    "NoOpMemoryStore",
    # routing
    "RoutingDecision",
    "Router",
    "NoOpRouter",
    "RoutingPolicy",
    "PassthroughRoutingPolicy",
    # flow
    "FlowDecision",
    "ActivationCondition",
    "FlowController",
    "NoOpFlowController",
    # recovery
    "RecoveryAction",
    "RepairPlan",
    "RecoveryStrategy",
    "NoOpRecoveryStrategy",
    # scheduling
    "ResourceTier",
    "ResourceRequest",
    "ResourceAllocation",
    "ResourceScheduler",
    "NoOpResourceScheduler",
]
