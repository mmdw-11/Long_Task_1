"""资源调度器抽象接口与空实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ._types import ResourceAllocation, ResourceRequest, ResourceTier


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
