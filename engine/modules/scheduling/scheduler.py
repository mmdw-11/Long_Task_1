"""端-边-云自适应资源调度器。"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from ._types import (
    ModelSplitStep,
    RealtimeRequirement,
    ResourceAllocation,
    ResourceProfile,
    ResourceRequest,
    ResourceStatus,
    ResourceTier,
    SchedulingDecision,
    SchedulingTrace,
    TaskComplexity,
    TaskProfile,
)
from .base import ResourceScheduler
from .gate import HeuristicTaskGate, TaskGate
from .monitor import NoOpResourceMonitor, ResourceMonitor
from .policy import TrustedWorkspacePolicy
from .production import load_production_gate


class AdaptiveResourceScheduler(ResourceScheduler):
    """端-边-云异构资源自适应调度器。"""

    def __init__(
        self,
        *,
        resources: Optional[List[ResourceProfile]] = None,
        gate: Optional[TaskGate] = None,
        trace_gate: Optional[TaskGate] = None,
        monitor: Optional[ResourceMonitor] = None,
        trusted_policy: Optional[TrustedWorkspacePolicy] = None,
        enable_trace: bool = True,
        router_path: Optional[str] = None,
        learned_threshold: float = 0.35,
        use_production_router: bool = False,
    ) -> None:
        if gate is not None:
            self.gate = gate
        elif use_production_router or router_path or _env_use_production_router():
            self.gate = load_production_gate(
                router_path=router_path,
                threshold=learned_threshold,
                fallback=HeuristicTaskGate(),
            )
        else:
            self.gate = HeuristicTaskGate()
        self.trace_gate = trace_gate or HeuristicTaskGate()
        self.monitor = monitor or NoOpResourceMonitor()
        self.trusted_policy = trusted_policy or TrustedWorkspacePolicy()
        self.enable_trace = enable_trace
        profiles = resources or self.default_resources()
        self.resources: Dict[ResourceTier, ResourceProfile] = {
            profile.tier: profile for profile in profiles
        }
        self.allocations: List[ResourceAllocation] = []
        self.traces: List[SchedulingTrace] = []
        self.last_statuses: Dict[ResourceTier, ResourceStatus] = {}

    @staticmethod
    def default_resources() -> List[ResourceProfile]:
        _load_dotenv()
        device_model = os.environ.get("DEVICE_MODEL", "qwen2.5:0.5b")
        edge_model = os.environ.get("EDGE_MODEL", "qwen2.5:3b")
        cloud_model = os.environ.get("OPENAI_MODEL", "deepseek-v4-flash")
        return [
            ResourceProfile(
                tier=ResourceTier.DEVICE,
                endpoint=os.environ.get("DEVICE_ENDPOINT", "http://127.0.0.1:11434/v1"),
                available=True,
                trusted=True,
                max_complexity=TaskComplexity.MEDIUM,
                latency_ms=30,
                cost_weight=0.2,
                models={"small": device_model, "embedding": "local-embedding"},
            ),
            ResourceProfile(
                tier=ResourceTier.EDGE,
                endpoint=os.environ.get("EDGE_ENDPOINT", "http://127.0.0.1:8001/infer"),
                available=True,
                trusted=True,
                max_complexity=TaskComplexity.HIGH,
                latency_ms=90,
                cost_weight=0.6,
                models={"medium": edge_model, "embedding": "edge-embedding"},
            ),
            ResourceProfile(
                tier=ResourceTier.CLOUD,
                endpoint=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
                available=True,
                trusted=False,
                max_complexity=TaskComplexity.EXTREME,
                latency_ms=220,
                cost_weight=1.0,
                models={"large": cloud_model, "vision": "cloud-vision-llm"},
            ),
        ]

    def acquire(self, request: ResourceRequest) -> ResourceAllocation:
        self.last_statuses = self.monitor.snapshot(self.resources)
        heuristic_profile = self.trace_gate.evaluate(request)
        profile = self.gate.evaluate(request)
        decision = self.decide(request, profile)
        trace = self._build_trace(
            request=request,
            heuristic_profile=heuristic_profile,
            gate_profile=profile,
            decision=decision,
        )
        allocation = ResourceAllocation(
            tier=decision.tier,
            endpoint=decision.endpoint,
            metadata={
                "scheduler": "adaptive",
                "decision": decision.to_dict(),
                "model_split": [step.to_dict() for step in decision.model_split],
                "requires_human_review": decision.requires_human_review,
                "reason": decision.reason,
                "resource_status": {
                    tier.value: status.to_dict()
                    for tier, status in self.last_statuses.items()
                },
            },
        )
        if self.enable_trace:
            allocation.metadata["trace"] = trace.to_dict()
            self.traces.append(trace)
        self.allocations.append(allocation)
        return allocation

    def release(self, allocation: ResourceAllocation) -> None:
        return None

    def available(self, tier: ResourceTier) -> bool:
        profile = self.resources.get(tier)
        status = self.last_statuses.get(tier)
        if status is not None:
            return bool(profile and profile.available and status.available and not status.rate_limited)
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
        return SchedulingDecision(
            tier=selected.tier,
            endpoint=selected.endpoint,
            profile=profile,
            model_split=self._model_split(selected.tier, profile),
            requires_human_review=requires_review and not profile.human_approved,
            reason=reason,
        )

    def _ordered_candidates(
        self, request: ResourceRequest, profile: TaskProfile
    ) -> List[ResourceProfile]:
        route_tier = _coerce_route_tier(profile.metadata.get("route_tier"))
        preferred = [route_tier] if route_tier is not None else request.tier_preference or [
            ResourceTier.DEVICE,
            ResourceTier.EDGE,
            ResourceTier.CLOUD,
        ]
        if route_tier is None and profile.complexity in (TaskComplexity.HIGH, TaskComplexity.EXTREME):
            preferred = [ResourceTier.CLOUD, ResourceTier.EDGE, ResourceTier.DEVICE]
        if route_tier is None and profile.realtime == RealtimeRequirement.HARD:
            preferred = [ResourceTier.DEVICE, ResourceTier.EDGE, ResourceTier.CLOUD]
        result: List[ResourceProfile] = []
        for tier in preferred:
            resource = self.resources.get(tier)
            if resource and self._resource_available(resource):
                result.append(resource)
        for tier in (ResourceTier.DEVICE, ResourceTier.EDGE, ResourceTier.CLOUD):
            resource = self.resources.get(tier)
            if resource and self._resource_available(resource) and resource not in result:
                result.append(resource)
        order = {tier: idx for idx, tier in enumerate(preferred)}
        return sorted(result, key=lambda resource: (order.get(resource.tier, 99), self._runtime_rank(resource)))

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
            candidate
            for candidate in candidates
            if self.trusted_policy.is_trusted(candidate.tier) and candidate.supports(profile.complexity)
        ]
        if trusted:
            return trusted[0]
        fallback = [candidate for candidate in candidates if self.trusted_policy.is_trusted(candidate.tier)]
        return fallback[0] if fallback else None

    def _requires_review(self, tier: ResourceTier, profile: TaskProfile) -> bool:
        if not self.trusted_policy.allow_untrusted_with_review:
            return False
        if not self.trusted_policy.is_sensitive(profile.sensitivity):
            return False
        return not self.trusted_policy.is_trusted(tier)

    def _model_split(self, tier: ResourceTier, profile: TaskProfile) -> List[ModelSplitStep]:
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
            steps.append(ModelSplitStep("local_inference", tier, os.environ.get("DEVICE_MODEL", "qwen2.5:0.5b"), "端侧完成推理"))
        elif tier == ResourceTier.EDGE:
            steps.append(ModelSplitStep("edge_inference", tier, os.environ.get("EDGE_MODEL", "qwen2.5:3b"), "边缘侧完成中等复杂度推理"))
        else:
            steps.append(ModelSplitStep("cloud_inference", tier, os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"), "云端处理高复杂度子任务"))
        return steps

    def _reason_for(self, tier: ResourceTier, profile: TaskProfile) -> str:
        if tier == ResourceTier.CLOUD:
            return "cloud_selected_for_high_complexity"
        if tier == ResourceTier.EDGE:
            return "edge_selected_for_latency_and_capacity_balance"
        return "device_selected_for_low_latency_or_privacy"

    def _build_trace(
        self,
        *,
        request: ResourceRequest,
        heuristic_profile: TaskProfile,
        gate_profile: TaskProfile,
        decision: SchedulingDecision,
    ) -> SchedulingTrace:
        return SchedulingTrace(
            node=request.node,
            task_text=self._task_text(request),
            heuristic_profile=heuristic_profile,
            gate_profile=gate_profile,
            final_profile=decision.profile,
            decision=decision,
            policy_hits=self._policy_hits(decision, heuristic_profile, gate_profile),
            metadata={
                "gate": type(self.gate).__name__,
                "trace_gate": type(self.trace_gate).__name__,
                "resource_status": {
                    tier.value: status.to_dict()
                    for tier, status in self.last_statuses.items()
                },
            },
        )

    def _policy_hits(
        self,
        decision: SchedulingDecision,
        heuristic_profile: TaskProfile,
        gate_profile: TaskProfile,
    ) -> List[str]:
        hits: List[str] = []
        profile = decision.profile
        if self.trusted_policy.is_sensitive(profile.sensitivity):
            hits.append("sensitive_data")
        if profile.requires_trusted_workspace:
            hits.append("trusted_workspace_required")
        if decision.requires_human_review:
            hits.append("human_review_required")
        if decision.reason == "sensitive_task_kept_in_trusted_workspace":
            hits.append("kept_in_trusted_workspace")
        if decision.tier == ResourceTier.CLOUD:
            hits.append("cloud_for_high_complexity")
        for tier, status in self.last_statuses.items():
            if not status.available:
                hits.append(f"{tier.value}_unavailable")
            if status.rate_limited:
                hits.append(f"{tier.value}_rate_limited")
        if profile.realtime == RealtimeRequirement.HARD:
            hits.append("hard_realtime")
        if heuristic_profile.sensitivity != gate_profile.sensitivity:
            hits.append("gate_sensitivity_diff")
        if heuristic_profile.complexity != gate_profile.complexity:
            hits.append("gate_complexity_diff")
        if heuristic_profile.realtime != gate_profile.realtime:
            hits.append("gate_realtime_diff")
        if heuristic_profile.task_type != gate_profile.task_type:
            hits.append("gate_task_type_diff")
        return hits

    def _task_text(self, request: ResourceRequest) -> str:
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

    def _resource_available(self, resource: ResourceProfile) -> bool:
        status = self.last_statuses.get(resource.tier)
        if status is None:
            return resource.available
        return resource.available and status.available and not status.rate_limited

    def _runtime_rank(self, resource: ResourceProfile) -> tuple:
        status = self.last_statuses.get(resource.tier)
        latency = status.latency_ms if status and status.latency_ms is not None else resource.latency_ms
        load = status.load if status and status.load is not None else 0.0
        queue = status.queue_depth if status and status.queue_depth is not None else 0
        error_rate = status.error_rate if status else 0.0
        return (error_rate, load, queue, latency, resource.cost_weight)


def _env_use_production_router() -> bool:
    value = str(os.environ.get("USE_PRODUCTION_ROUTER", "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _coerce_route_tier(value: object) -> Optional[ResourceTier]:
    if value is None:
        return None
    try:
        return ResourceTier(str(value))
    except ValueError:
        return None


def _load_dotenv() -> None:
    if os.environ.get("AGENT_GRAPH_LOAD_DOTENV") == "0":
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
        return
    except Exception:
        pass
    env_path = _find_dotenv()
    if env_path is None:
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _find_dotenv():
    from pathlib import Path

    current = Path.cwd()
    for path in [current, *current.parents]:
        candidate = path / ".env"
        if candidate.exists():
            return candidate
    return None
