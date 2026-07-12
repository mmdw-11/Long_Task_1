"""推理执行器注册表。"""

from __future__ import annotations

from typing import Dict, Optional

from engine.modules.scheduling import ResourceTier

from ._types import InferenceRequest, InferenceResult
from .base import InferenceExecutor
from .executors import EdgeHttpExecutor, LocalEchoExecutor, OpenAICompatibleCloudExecutor


class ExecutorRegistry:
    """按调度 tier 分发到对应推理执行器。"""

    def __init__(self, executors: Optional[Dict[ResourceTier, InferenceExecutor]] = None) -> None:
        self.executors: Dict[ResourceTier, InferenceExecutor] = executors or {
            ResourceTier.DEVICE: LocalEchoExecutor(),
            ResourceTier.EDGE: EdgeHttpExecutor(),
            ResourceTier.CLOUD: OpenAICompatibleCloudExecutor(),
        }

    @classmethod
    def default(cls) -> "ExecutorRegistry":
        return cls()

    def run(self, request: InferenceRequest) -> InferenceResult:
        tier = ResourceTier(str(request.allocation.get("tier", ResourceTier.DEVICE.value)))
        executor = self.executors.get(tier)
        if executor is None:
            executor = self.executors[ResourceTier.DEVICE]
        return executor.run(request)
