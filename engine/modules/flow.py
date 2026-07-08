"""流控模块：依赖解析 + 条件激活。

``FlowController`` 在节点执行前介入，综合依赖是否满足与命名激活条件（如
``on_test_failure``），决定该子任务是**执行 / 跳过 / 延迟**。可用于替换或补充
框架默认「前沿即执行」的调度行为。

本文件仅提供接口与空实现桩（``NoOpFlowController`` 恒返回 EXECUTE）。
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class FlowDecision(str, enum.Enum):
    """流控决策：节点在本超步应如何处理。"""

    EXECUTE = "execute"   # 正常执行
    SKIP = "skip"         # 跳过（不执行，视作已完成）
    DEFER = "defer"       # 延迟（本超步不执行，后续再评估）


# 条件判定函数：接收状态快照，返回该条件是否命中。
Predicate = Callable[[Dict[str, Any]], bool]


@dataclass
class ActivationCondition:
    """命名的激活条件（如 ``on_test_failure``）。"""

    name: str
    predicate: Predicate
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, state: Dict[str, Any]) -> bool:
        return bool(self.predicate(state))


class FlowController(ABC):
    """流控器接口：依赖解析 + 条件激活。"""

    @abstractmethod
    def resolve_dependencies(
        self, node: str, graph_view: Dict[str, Any]
    ) -> List[str]:
        """解析 ``node`` 的前置依赖节点列表。

        :param graph_view: 图结构视图（节点/边信息），供实现解析拓扑依赖。
        """
        raise NotImplementedError

    @abstractmethod
    def decide(
        self,
        node: str,
        *,
        state: Dict[str, Any],
        deps_satisfied: bool,
    ) -> FlowDecision:
        """决定节点在本超步的处理方式（执行 / 跳过 / 延迟）。"""
        raise NotImplementedError


class NoOpFlowController(FlowController):
    """空实现桩：不做依赖解析，恒判定为执行（等价于框架现有行为）。"""

    def resolve_dependencies(
        self, node: str, graph_view: Dict[str, Any]
    ) -> List[str]:
        return []

    def decide(
        self,
        node: str,
        *,
        state: Dict[str, Any],
        deps_satisfied: bool,
    ) -> FlowDecision:
        return FlowDecision.EXECUTE
