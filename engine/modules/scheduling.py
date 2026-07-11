"""端-边-云异构资源自适应调度模块。

该模块把资源调度拆成三步：

1. **任务门控评估**：依据节点元数据、运行状态和任务文本，判断实时性、
   数据敏感等级、复杂度与是否需要可信工作区。
2. **可信工作区约束**：敏感数据默认只能在可信层级运行；如果必须外传到
   非可信层级，需要显式人工审计批准。
3. **端边云选择 + 模型切分**：优先满足约束，再按实时性和复杂度选择 DEVICE /
   EDGE / CLOUD，并给出本地预处理、边缘中间推理、云端重推理等切分计划。

这里的调度器不直接执行远程推理；它产出 :class:`ResourceAllocation`，由图引擎
注入节点状态。真实系统可以根据该分配结果把节点内部推理转发到本地模型、边缘服务
或云端服务。
"""

from __future__ import annotations

import enum
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ResourceTier(str, enum.Enum):
    """资源层级。"""

    DEVICE = "device"  # 端：终端设备本地算力
    EDGE = "edge"      # 边：边缘节点
    CLOUD = "cloud"    # 云：云端集群


class RealtimeRequirement(str, enum.Enum):
    """子任务实时性要求。"""

    HARD = "hard"          # 强实时：优先本地/边缘，避免远程排队。
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


_COMPLEXITY_SCORE = {
    TaskComplexity.LOW: 1,
    TaskComplexity.MEDIUM: 2,
    TaskComplexity.HIGH: 3,
    TaskComplexity.EXTREME: 4,
}

_SENSITIVITY_SCORE = {
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
        return _COMPLEXITY_SCORE[self.max_complexity] >= _COMPLEXITY_SCORE[complexity]


@dataclass
class TrustedWorkspacePolicy:
    """可信工作区策略。

    ``trusted_tiers`` 表示被视为可信工作区的推理位置。敏感等级达到
    ``sensitive_threshold`` 后，调度器会优先限制在可信层级内；如果必须选择
    非可信层级，则需要 ``human_approved`` 或在节点/状态中显式审计通过。
    """

    trusted_tiers: List[ResourceTier] = field(
        default_factory=lambda: [ResourceTier.DEVICE, ResourceTier.EDGE]
    )
    sensitive_threshold: SensitivityLevel = SensitivityLevel.CONFIDENTIAL
    allow_untrusted_with_review: bool = True

    def is_sensitive(self, level: SensitivityLevel) -> bool:
        return _SENSITIVITY_SCORE[level] >= _SENSITIVITY_SCORE[self.sensitive_threshold]

    def is_trusted(self, tier: ResourceTier) -> bool:
        return tier in self.trusted_tiers


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
            "profile": {
                "realtime": self.profile.realtime.value,
                "sensitivity": self.profile.sensitivity.value,
                "complexity": self.profile.complexity.value,
                "task_type": self.profile.task_type,
                "requires_trusted_workspace": self.profile.requires_trusted_workspace,
                "human_approved": self.profile.human_approved,
                "metadata": self.profile.metadata,
            },
            "model_split": [step.to_dict() for step in self.model_split],
            "requires_human_review": self.requires_human_review,
            "reason": self.reason,
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


class TaskGate(ABC):
    """任务门控单元接口。"""

    @abstractmethod
    def evaluate(self, request: ResourceRequest) -> TaskProfile:
        """评估任务画像。"""
        raise NotImplementedError


class HeuristicTaskGate(TaskGate):
    """无需外部模型的本地门控单元。

    生产环境可替换成“本地 LLM skill”或更复杂的分类器；这里提供稳定可测的
    默认实现。显式元数据优先，其次从文本关键词和规模粗略估计。
    """

    _secret_patterns = [
        r"api[_ -]?key",
        r"password",
        r"token",
        r"secret",
        r"身份证",
        r"银行卡",
        r"密钥",
        r"私钥",
    ]
    _urgent_words = ["urgent", "实时", "立即", "马上", "紧急", "低延迟"]
    _cloud_words = ["大规模", "复杂", "长文", "图像", "视频", "全量分析", "深度推理"]

    def evaluate(self, request: ResourceRequest) -> TaskProfile:
        meta = dict(request.metadata or {})
        text = self._collect_text(request)

        sensitivity = self._enum_from_meta(
            meta,
            "sensitivity",
            SensitivityLevel,
            self._infer_sensitivity(text),
        )
        realtime = self._enum_from_meta(
            meta,
            "realtime",
            RealtimeRequirement,
            self._infer_realtime(text),
        )
        complexity = self._enum_from_meta(
            meta,
            "complexity",
            TaskComplexity,
            self._infer_complexity(text),
        )
        task_type = str(meta.get("task_type") or self._infer_task_type(text))
        requires_trusted = bool(
            meta.get("requires_trusted_workspace")
            or sensitivity in (SensitivityLevel.CONFIDENTIAL, SensitivityLevel.SECRET)
        )
        human_approved = bool(
            meta.get("human_approved")
            or request.state.get("human_approved")
            or request.state.get("cloud_audit_approved")
        )
        return TaskProfile(
            realtime=realtime,
            sensitivity=sensitivity,
            complexity=complexity,
            task_type=task_type,
            requires_trusted_workspace=requires_trusted,
            human_approved=human_approved,
            metadata={
                "node": request.node,
                "text_length": len(text),
                "explicit": {
                    key: value
                    for key, value in meta.items()
                    if key in {"sensitivity", "realtime", "complexity", "task_type"}
                },
            },
        )

    def _collect_text(self, request: ResourceRequest) -> str:
        parts = [request.node]
        for key in ("input", "task", "goal", "query", "messages"):
            value = request.state.get(key)
            if value:
                parts.append(str(value))
        for key in ("description", "sys_prompt"):
            value = request.metadata.get(key)
            if value:
                parts.append(str(value))
        return "\n".join(parts)

    def _infer_sensitivity(self, text: str) -> SensitivityLevel:
        lowered = text.lower()
        if any(re.search(pattern, lowered) for pattern in self._secret_patterns):
            return SensitivityLevel.SECRET
        if any(word in text for word in ["客户", "隐私", "个人", "内部", "邮件"]):
            return SensitivityLevel.CONFIDENTIAL
        return SensitivityLevel.INTERNAL

    def _infer_realtime(self, text: str) -> RealtimeRequirement:
        if any(word in text for word in self._urgent_words):
            return RealtimeRequirement.HARD
        if len(text) < 600:
            return RealtimeRequirement.INTERACTIVE
        return RealtimeRequirement.NORMAL

    def _infer_complexity(self, text: str) -> TaskComplexity:
        if len(text) > 6000 or any(word in text for word in self._cloud_words):
            return TaskComplexity.HIGH
        if len(text) > 1800:
            return TaskComplexity.MEDIUM
        return TaskComplexity.LOW

    def _infer_task_type(self, text: str) -> str:
        if any(word in text for word in ["日程", "calendar", "会议"]):
            return "calendar"
        if any(word in text for word in ["邮件", "email", "回复"]):
            return "email"
        if any(word in text for word in ["代码", "测试", "bug"]):
            return "code"
        return "general"

    def _enum_from_meta(self, meta: Dict[str, Any], key: str, enum_cls: Any, default: Any) -> Any:
        value = meta.get(key)
        if value is None:
            return default
        if isinstance(value, enum_cls):
            return value
        return enum_cls(str(value).lower())


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


class AdaptiveResourceScheduler(ResourceScheduler):
    """端-边-云异构资源自适应调度器。

    选择策略：
    - 敏感任务优先留在可信工作区（默认 DEVICE/EDGE）。
    - 强实时任务优先 DEVICE/EDGE。
    - 高复杂度任务优先 CLOUD；若敏感且无审计，则退回可信层级。
    - CLOUD 被选中时，为高复杂度任务生成“端侧脱敏/压缩 + 云端重推理”的切分计划。
    """

    def __init__(
        self,
        *,
        resources: Optional[List[ResourceProfile]] = None,
        gate: Optional[TaskGate] = None,
        trusted_policy: Optional[TrustedWorkspacePolicy] = None,
    ) -> None:
        self.gate = gate or HeuristicTaskGate()
        self.trusted_policy = trusted_policy or TrustedWorkspacePolicy()
        profiles = resources or self.default_resources()
        self.resources: Dict[ResourceTier, ResourceProfile] = {
            profile.tier: profile for profile in profiles
        }
        self.allocations: List[ResourceAllocation] = []

    @staticmethod
    def default_resources() -> List[ResourceProfile]:
        return [
            ResourceProfile(
                tier=ResourceTier.DEVICE,
                endpoint="local",
                available=True,
                trusted=True,
                max_complexity=TaskComplexity.MEDIUM,
                latency_ms=30,
                cost_weight=0.2,
                models={"small": "local-small-llm", "embedding": "local-embedding"},
            ),
            ResourceProfile(
                tier=ResourceTier.EDGE,
                endpoint="edge://default",
                available=True,
                trusted=True,
                max_complexity=TaskComplexity.HIGH,
                latency_ms=90,
                cost_weight=0.6,
                models={"medium": "edge-medium-llm", "embedding": "edge-embedding"},
            ),
            ResourceProfile(
                tier=ResourceTier.CLOUD,
                endpoint="cloud://default",
                available=True,
                trusted=False,
                max_complexity=TaskComplexity.EXTREME,
                latency_ms=220,
                cost_weight=1.0,
                models={"large": "cloud-large-llm", "vision": "cloud-vision-llm"},
            ),
        ]

    def acquire(self, request: ResourceRequest) -> ResourceAllocation:
        profile = self.gate.evaluate(request)
        decision = self.decide(request, profile)
        allocation = ResourceAllocation(
            tier=decision.tier,
            endpoint=decision.endpoint,
            metadata={
                "scheduler": "adaptive",
                "decision": decision.to_dict(),
                "model_split": [step.to_dict() for step in decision.model_split],
                "requires_human_review": decision.requires_human_review,
                "reason": decision.reason,
            },
        )
        self.allocations.append(allocation)
        return allocation

    def release(self, allocation: ResourceAllocation) -> None:
        return None

    def available(self, tier: ResourceTier) -> bool:
        profile = self.resources.get(tier)
        return bool(profile and profile.available)

    def decide(self, request: ResourceRequest, profile: TaskProfile) -> SchedulingDecision:
        candidates = self._ordered_candidates(request, profile)
        selected = self._select_candidate(candidates, profile)
        requires_review = self._requires_review(selected.tier, profile)
        if requires_review and not profile.human_approved:
            trusted = self._best_trusted_candidate(candidates, profile)
            if trusted is not None:
                selected = trusted
                reason = "sensitive_task_kept_in_trusted_workspace"
                requires_review = False
            else:
                reason = "external_sensitive_transfer_requires_human_review"
        else:
            reason = self._reason_for(selected.tier, profile)

        split = self._model_split(selected.tier, profile)
        return SchedulingDecision(
            tier=selected.tier,
            endpoint=selected.endpoint,
            profile=profile,
            model_split=split,
            requires_human_review=requires_review and not profile.human_approved,
            reason=reason,
        )

    def _ordered_candidates(
        self, request: ResourceRequest, profile: TaskProfile
    ) -> List[ResourceProfile]:
        preferred = request.tier_preference or [
            ResourceTier.DEVICE,
            ResourceTier.EDGE,
            ResourceTier.CLOUD,
        ]
        if profile.complexity in (TaskComplexity.HIGH, TaskComplexity.EXTREME):
            preferred = [ResourceTier.CLOUD, ResourceTier.EDGE, ResourceTier.DEVICE]
        if profile.realtime == RealtimeRequirement.HARD:
            preferred = [ResourceTier.DEVICE, ResourceTier.EDGE, ResourceTier.CLOUD]
        result: List[ResourceProfile] = []
        for tier in preferred:
            resource = self.resources.get(tier)
            if resource and resource.available:
                result.append(resource)
        for tier in (ResourceTier.DEVICE, ResourceTier.EDGE, ResourceTier.CLOUD):
            resource = self.resources.get(tier)
            if resource and resource.available and resource not in result:
                result.append(resource)
        return result

    def _select_candidate(
        self, candidates: List[ResourceProfile], profile: TaskProfile
    ) -> ResourceProfile:
        for candidate in candidates:
            if candidate.supports(profile.complexity):
                return candidate
        if candidates:
            return candidates[-1]
        raise RuntimeError("no resource tier is available")

    def _best_trusted_candidate(
        self, candidates: List[ResourceProfile], profile: TaskProfile
    ) -> Optional[ResourceProfile]:
        trusted = [
            candidate for candidate in candidates
            if self.trusted_policy.is_trusted(candidate.tier) and candidate.supports(profile.complexity)
        ]
        if trusted:
            return trusted[0]
        fallback = [
            candidate for candidate in candidates
            if self.trusted_policy.is_trusted(candidate.tier)
        ]
        return fallback[0] if fallback else None

    def _requires_review(self, tier: ResourceTier, profile: TaskProfile) -> bool:
        if not self.trusted_policy.allow_untrusted_with_review:
            return False
        if not self.trusted_policy.is_sensitive(profile.sensitivity):
            return False
        return not self.trusted_policy.is_trusted(tier)

    def _model_split(
        self, tier: ResourceTier, profile: TaskProfile
    ) -> List[ModelSplitStep]:
        steps = [
            ModelSplitStep(
                name="local_gate",
                tier=ResourceTier.DEVICE,
                model_hint="local-small-llm",
                purpose="评估任务类型、难度与敏感等级",
            )
        ]
        if self.trusted_policy.is_sensitive(profile.sensitivity):
            steps.append(
                ModelSplitStep(
                    name="trusted_redaction",
                    tier=ResourceTier.DEVICE,
                    model_hint="local-redactor",
                    purpose="在可信工作区内做脱敏、摘要或人工审计包生成",
                )
            )
        if tier == ResourceTier.DEVICE:
            steps.append(
                ModelSplitStep("local_inference", tier, "local-small-llm", "端侧完成推理")
            )
        elif tier == ResourceTier.EDGE:
            steps.append(
                ModelSplitStep("edge_inference", tier, "edge-medium-llm", "边缘侧完成中等复杂度推理")
            )
        else:
            steps.append(
                ModelSplitStep("cloud_inference", tier, "cloud-large-llm", "云端处理高复杂度子任务")
            )
        return steps

    def _reason_for(self, tier: ResourceTier, profile: TaskProfile) -> str:
        if tier == ResourceTier.CLOUD:
            return "cloud_selected_for_high_complexity"
        if tier == ResourceTier.EDGE:
            return "edge_selected_for_latency_and_capacity_balance"
        return "device_selected_for_low_latency_or_privacy"
