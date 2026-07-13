"""推理执行器接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ._types import InferenceRequest, InferenceResult


class InferenceExecutor(ABC):
    """根据调度结果执行推理的接口。"""

    @abstractmethod
    def run(self, request: InferenceRequest) -> InferenceResult:
        raise NotImplementedError
