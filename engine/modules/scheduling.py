"""端边云资源调度模块。

``ResourceScheduler`` 抽象了「端（device）/ 边（edge）/ 云（cloud）」三级资源的
申请与释放。节点执行前向调度器申请资源、执行后释放，从而将计算就近或按需分配到
不同层级。可用于替换/补充框架默认的本地直跑行为。

本文件仅提供接口与空实现桩（``NoOpResourceScheduler`` 本地直跑、无副作用）。
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ResourceTier(str, enum.Enum):
    """资源层级。"""

    DEVICE = "device"  # 端：终端设备本地算力
    EDGE = "edge"      # 边：边缘节点
    CLOUD = "cloud"    # 云：云端集群


@dataclass
class ResourceRequest:
    """一次资源申请。"""

    node: str
    tier_preference: List[ResourceTier] = field(
        default_factory=lambda: [ResourceTier.DEVICE, ResourceTier.EDGE, ResourceTier.CLOUD]
    )
    cpu: float = 0.0
    mem_mb: float = 0.0
    gpu: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceAllocation:
    """一次资源分配结果。"""

    tier: ResourceTier
    endpoint: str = "local"
    handle: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResourceScheduler(ABC):
    """端边云资源调度器接口。"""

    @abstractmethod
    def acquire(self, request: ResourceRequest) -> ResourceAllocation:
        """按请求申请资源，返回分配结果。"""
        raise NotImplementedError

    @abstractmethod
    def release(self, allocation: ResourceAllocation) -> None:
        """释放已分配的资源。"""
        raise NotImplementedError

    @abstractmethod
    def available(self, tier: ResourceTier) -> bool:
        """查询某层级当前是否有可用资源。"""
        raise NotImplementedError


class NoOpResourceScheduler(ResourceScheduler):
    """空实现桩：一律本地直跑（DEVICE / local），申请释放无副作用。"""

    def acquire(self, request: ResourceRequest) -> ResourceAllocation:
        return ResourceAllocation(tier=ResourceTier.DEVICE, endpoint="local")

    def release(self, allocation: ResourceAllocation) -> None:
        return None

    def available(self, tier: ResourceTier) -> bool:
        return True
