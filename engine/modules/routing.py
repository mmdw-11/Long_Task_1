"""动态路由模块。

提供两层抽象：

- ``Router``        : 静态路由接口（根据 key 查目标节点列表）。
- ``RoutingPolicy`` : 运行时动态路由策略接口（综合上下文对候选后继给出决策）。

本文件仅提供接口与空实现桩（``NoOpRouter`` 返回空列表，
``PassthroughRoutingPolicy`` 原样返回候选）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RoutingDecision:
    """一次路由决策。"""

    targets: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class Router(ABC):
    """静态路由接口：根据 key 查目标节点列表。"""

    @abstractmethod
    def route(self, key: str) -> List[str]:
        """根据路由 key 返回目标节点名列表（未命中返回空列表）。"""
        raise NotImplementedError


class NoOpRouter(Router):
    """空实现桩：恒返回空列表。"""

    def route(self, key: str) -> List[str]:
        return []


class RoutingPolicy(ABC):
    """运行时动态路由策略接口。"""

    @abstractmethod
    def decide(
        self,
        source: str,
        candidates: List[str],
        *,
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """综合上下文对候选后继给出路由决策。

        :param source:     当前节点名。
        :param candidates: 引擎计算出的候选后继节点。
        :param state:      当前状态快照。
        :param context:    可选的附加上下文（项目规则、记忆、失败轨迹等）。
        :return: 路由决策。
        """
        raise NotImplementedError


class PassthroughRoutingPolicy(RoutingPolicy):
    """空实现桩：原样返回候选（不改变框架默认路由）。"""

    def decide(
        self,
        source: str,
        candidates: List[str],
        *,
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        return RoutingDecision(targets=list(candidates))
