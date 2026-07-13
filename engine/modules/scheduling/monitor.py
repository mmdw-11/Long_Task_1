"""资源状态探测。"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional

from ._types import ResourceProfile, ResourceStatus, ResourceTier


class ResourceMonitor(ABC):
    """端-边-云资源状态探测接口。"""

    @abstractmethod
    def snapshot(self, resources: Dict[ResourceTier, ResourceProfile]) -> Dict[ResourceTier, ResourceStatus]:
        """返回每个资源层级的实时状态。"""
        raise NotImplementedError


class NoOpResourceMonitor(ResourceMonitor):
    """空探测器：认为所有静态资源状态就是可用状态。"""

    def snapshot(self, resources: Dict[ResourceTier, ResourceProfile]) -> Dict[ResourceTier, ResourceStatus]:
        return {
            tier: ResourceStatus(
                tier=tier,
                available=profile.available,
                latency_ms=profile.latency_ms,
            )
            for tier, profile in resources.items()
        }


class StaticResourceMonitor(ResourceMonitor):
    """测试/演示用静态状态探测器。"""

    def __init__(self, statuses: Iterable[ResourceStatus]) -> None:
        self.statuses = {status.tier: status for status in statuses}

    def snapshot(self, resources: Dict[ResourceTier, ResourceProfile]) -> Dict[ResourceTier, ResourceStatus]:
        result = NoOpResourceMonitor().snapshot(resources)
        result.update(self.statuses)
        return result


class MutableResourceMonitor(ResourceMonitor):
    """Runtime monitor that can absorb execution feedback."""

    def __init__(self, statuses: Optional[Iterable[ResourceStatus]] = None) -> None:
        self.statuses = {status.tier: status for status in statuses or []}

    def snapshot(self, resources: Dict[ResourceTier, ResourceProfile]) -> Dict[ResourceTier, ResourceStatus]:
        result = NoOpResourceMonitor().snapshot(resources)
        result.update(self.statuses)
        return result

    def set_status(self, status: ResourceStatus) -> None:
        self.statuses[status.tier] = status

    def record_failure(self, tier: ResourceTier, reason: str) -> None:
        status = self.statuses.get(tier) or ResourceStatus(tier=tier)
        reason_lower = reason.lower()
        rate_limited = "429" in reason_lower or "rate" in reason_lower or "limit" in reason_lower
        status.rate_limited = status.rate_limited or rate_limited
        status.available = False if not rate_limited else status.available
        status.error_rate = 1.0
        status.metadata = {**status.metadata, "last_error": reason}
        self.statuses[tier] = status


class SystemResourceMonitor(ResourceMonitor):
    """轻量级本机与 HTTP endpoint 探测器。

    - DEVICE: 使用标准库估算本机 load/memory 可用性。
    - EDGE/CLOUD: 对 http(s) endpoint 做 HEAD 探测；非 HTTP 逻辑 endpoint 只沿用
      静态资源状态。
    """

    def __init__(self, *, timeout_seconds: float = 2.0) -> None:
        self.timeout_seconds = timeout_seconds

    def snapshot(self, resources: Dict[ResourceTier, ResourceProfile]) -> Dict[ResourceTier, ResourceStatus]:
        statuses = NoOpResourceMonitor().snapshot(resources)
        for tier, profile in resources.items():
            if tier == ResourceTier.DEVICE:
                statuses[tier] = self._device_status(profile)
            elif profile.endpoint.startswith(("http://", "https://")):
                statuses[tier] = self._http_status(profile)
        return statuses

    def _device_status(self, profile: ResourceProfile) -> ResourceStatus:
        load = None
        try:
            import os

            if hasattr(os, "getloadavg"):
                load = os.getloadavg()[0]
        except Exception:
            load = None
        return ResourceStatus(
            tier=profile.tier,
            available=profile.available,
            latency_ms=profile.latency_ms,
            load=load,
            metadata={"probe": "device"},
        )

    def _http_status(self, profile: ResourceProfile) -> ResourceStatus:
        start = time.perf_counter()
        try:
            import requests

            resp = requests.head(profile.endpoint, timeout=self.timeout_seconds)
            latency = int((time.perf_counter() - start) * 1000)
            return ResourceStatus(
                tier=profile.tier,
                available=profile.available and resp.status_code < 500,
                latency_ms=latency,
                rate_limited=resp.status_code == 429,
                error_rate=0.0 if resp.status_code < 500 else 1.0,
                metadata={"status_code": resp.status_code, "probe": "http_head"},
            )
        except Exception as exc:  # noqa: BLE001 - monitoring should not crash scheduling
            latency = int((time.perf_counter() - start) * 1000)
            return ResourceStatus(
                tier=profile.tier,
                available=False,
                latency_ms=latency,
                error_rate=1.0,
                metadata={"probe": "http_head", "error": str(exc)},
            )
