"""推理执行数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class InferenceRequest:
    """一次推理执行请求。"""

    prompt: str
    allocation: Dict[str, Any]
    system_prompt: str = ""
    redacted_payload: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """一次推理执行结果。"""

    text: str
    executor: str
    endpoint: str
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    retryable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "executor": self.executor,
            "endpoint": self.endpoint,
            "model": self.model,
            "metadata": self.metadata,
            "success": self.success,
            "error": self.error,
            "retryable": self.retryable,
        }
