"""端-边-云调度类型定义。"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ResourceTier(str, enum.Enum):
    """资源层级。"""

    DEVICE = "device"
    EDGE = "edge"
    CLOUD = "cloud"


class RealtimeRequirement(str, enum.Enum):
    """子任务实时性要求。"""

    HARD = "hard"
    INTERACTIVE = "interactive"
    NORMAL = "normal"
    BATCH = "batch"


class SensitivityLevel(str, enum.Enum):
    """数据敏感等级。"""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


class TaskComplexity(str, enum.Enum):
    """任务复杂度档位。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


COMPLEXITY_SCORE = {
    TaskComplexity.LOW: 1,
    TaskComplexity.MEDIUM: 2,
    TaskComplexity.HIGH: 3,
    TaskComplexity.EXTREME: 4,
}

SENSITIVITY_SCORE = {
    SensitivityLevel.PUBLIC: 0,
    SensitivityLevel.INTERNAL: 1,
    SensitivityLevel.CONFIDENTIAL: 2,
    SensitivityLevel.SECRET: 3,
}


@dataclass
class TaskProfile:
    """门控单元给出的任务画像。"""

    realtime: RealtimeRequirement = RealtimeRequirement.NORMAL
    sensitivity: SensitivityLevel = SensitivityLevel.INTERNAL
    complexity: TaskComplexity = TaskComplexity.MEDIUM
    task_type: str = "general"
    requires_trusted_workspace: bool = False
    human_approved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "realtime": self.realtime.value,
            "sensitivity": self.sensitivity.value,
            "complexity": self.complexity.value,
            "task_type": self.task_type,
            "requires_trusted_workspace": self.requires_trusted_workspace,
            "human_approved": self.human_approved,
            "metadata": self.metadata,
        }


@dataclass
class ModelSplitStep:
    """模型切分计划中的一步。"""

    name: str
    tier: ResourceTier
    model_hint: str
    purpose: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier.value,
            "model_hint": self.model_hint,
            "purpose": self.purpose,
        }


@dataclass
class ResourceProfile:
    """某一层级的资源画像。"""

    tier: ResourceTier
    endpoint: str
    available: bool = True
    trusted: bool = False
    max_complexity: TaskComplexity = TaskComplexity.MEDIUM
    latency_ms: int = 100
    cost_weight: float = 1.0
    models: Dict[str, str] = field(default_factory=dict)

    def supports(self, complexity: TaskComplexity) -> bool:
        return COMPLEXITY_SCORE[self.max_complexity] >= COMPLEXITY_SCORE[complexity]


@dataclass
class ResourceStatus:
    """某一资源层级的实时状态快照。"""

    tier: ResourceTier
    available: bool = True
    latency_ms: Optional[int] = None
    load: Optional[float] = None
    queue_depth: Optional[int] = None
    rate_limited: bool = False
    error_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "available": self.available,
            "latency_ms": self.latency_ms,
            "load": self.load,
            "queue_depth": self.queue_depth,
            "rate_limited": self.rate_limited,
            "error_rate": self.error_rate,
            "metadata": self.metadata,
        }


@dataclass
class SchedulingDecision:
    """一次调度决策。"""

    tier: ResourceTier
    endpoint: str
    profile: TaskProfile
    model_split: List[ModelSplitStep]
    requires_human_review: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "endpoint": self.endpoint,
            "profile": self.profile.to_dict(),
            "model_split": [step.to_dict() for step in self.model_split],
            "requires_human_review": self.requires_human_review,
            "reason": self.reason,
        }


@dataclass
class SchedulingTrace:
    """一次调度的可回放轨迹。"""

    node: str
    task_text: str
    heuristic_profile: TaskProfile
    gate_profile: TaskProfile
    final_profile: TaskProfile
    decision: SchedulingDecision
    policy_hits: List[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "task_text": self.task_text,
            "heuristic_profile": self.heuristic_profile.to_dict(),
            "gate_profile": self.gate_profile.to_dict(),
            "final_profile": self.final_profile.to_dict(),
            "decision": self.decision.to_dict(),
            "policy_hits": list(self.policy_hits),
            "ts": self.ts,
            "metadata": self.metadata,
        }


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
    state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceAllocation:
    """一次资源分配结果。"""

    tier: ResourceTier
    endpoint: str = "local"
    handle: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "endpoint": self.endpoint,
            "metadata": self.metadata,
        }
