"""推理执行模块：根据端-边-云调度结果调用对应后端。"""

from ._types import InferenceRequest, InferenceResult
from .base import InferenceExecutor
from .executors import EdgeHttpExecutor, LocalEchoExecutor, OpenAICompatibleCloudExecutor
from .registry import ExecutorRegistry
from .runner import ResilientInferenceRunner

__all__ = [
    "EdgeHttpExecutor",
    "ExecutorRegistry",
    "InferenceExecutor",
    "InferenceRequest",
    "InferenceResult",
    "LocalEchoExecutor",
    "OpenAICompatibleCloudExecutor",
    "ResilientInferenceRunner",
]
