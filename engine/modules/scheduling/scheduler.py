"""端-边-云自适应资源调度器。"""

from __future__ import annotations

import os
import time
import uuid
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
from .constrained import (
    BUDGET_KEY, TRAJECTORY_KEY, COSTS, ConstrainedPlacementPolicy,
    PlacementInfeasible, action_features, budget_for, finite, recalculate_spent,
)


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
        strict_router: bool = False,
        placement_policy: Optional[ConstrainedPlacementPolicy] = None,
        placement_mode: Optional[str] = None,
        placement_policy_path: Optional[str] = None,
    ) -> None:
        if gate is not None:
            self.gate = gate
        else:
            self.gate = load_production_gate(
                router_path=router_path,
                threshold=learned_threshold,
                fallback=HeuristicTaskGate(),
                strict=strict_router,
            )
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
        self.placement_mode = placement_mode or os.environ.get("PLACEMENT_MODE", "auto")
        if self.placement_mode not in {"auto", "legacy", "constrained"}:
            raise ValueError("PLACEMENT_MODE must be auto, legacy or constrained")
        self.placement_policy = placement_policy
        self._placement_load_error = ""
        self._last_placement_context: Dict = {}
        policy_path = placement_policy_path or os.environ.get("PLACEMENT_POLICY_PATH")
        if placement_policy is None and policy_path:
            try:
                self.placement_policy = ConstrainedPlacementPolicy.load(policy_path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self._placement_load_error = type(exc).__name__

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
        self._last_placement_context = {}
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
        if self._last_placement_context:
            allocation.metadata["placement"] = self._last_placement_context
            trace.metadata["placement"] = self._last_placement_context
            if self.enable_trace:
                allocation.metadata["trace"] = trace.to_dict()
        elif self._placement_load_error:
            allocation.metadata["placement_fallback"] = self._placement_load_error
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
        if self.constrained_enabled(request):
            return self._decide_constrained(request, profile)
        return self._legacy_decide(request, profile)

    def constrained_enabled(self, request: ResourceRequest) -> bool:
        mode = request.metadata.get("placement_mode", self.placement_mode)
        if mode not in {"auto", "legacy", "constrained"}:
            raise ValueError("unknown placement mode")
        return (
            "placement_quality_tiers" in request.metadata
            or mode == "constrained"
            or (mode == "auto" and self.placement_policy is not None and self.placement_policy.updates > 0)
        )

    def _safe_candidates(self, request, profile):
        budget = budget_for(request)
        allowed = request.metadata.get("allowed_tiers")
        quality_tiers = request.metadata.get("placement_quality_tiers")
        if "placement_quality_tiers" in request.metadata and (
            not isinstance(quality_tiers, (list, tuple))
            or any(tier not in ("device", "edge", "cloud") for tier in quality_tiers)
        ):
            raise ValueError("placement_quality_tiers must list device, edge or cloud tiers")
        estimates = request.metadata.get("placement_estimates") or {}
        safe, excluded = [], {}
        for candidate in self._ordered_candidates(request, profile):
            reasons = []
            tier = candidate.tier.value
            if allowed is not None and tier not in allowed:
                reasons.append("unauthorized_tier")
            if quality_tiers is not None and tier not in quality_tiers:
                reasons.append("quality_not_validated")
            if not candidate.supports(profile.complexity):
                reasons.append("insufficient_capability")
            sensitive = profile.requires_trusted_workspace or self.trusted_policy.is_sensitive(profile.sensitivity)
            if sensitive and not profile.human_approved and not (candidate.trusted and self.trusted_policy.is_trusted(candidate.tier)):
                reasons.append("untrusted_workspace")
            if request.metadata.get("strict_route") and profile.metadata.get("route_tier") and tier != profile.metadata["route_tier"]:
                reasons.append("strict_route")
            status = self.last_statuses.get(candidate.tier)
            estimate = dict(estimates.get(tier) or {})
            estimate.setdefault("time_ms", status.latency_ms if status and status.latency_ms is not None else candidate.latency_ms)
            for key, limit in budget["limits"].items():
                if key not in COSTS:
                    raise ValueError(f"unknown placement budget: {key}")
                remaining = finite(limit) - finite(budget["spent"].get(key, 0))
                if remaining <= 0:
                    reasons.append(f"{key}_exhausted")
                elif key == "fee" and ("fee" not in estimate or budget.get("unpriced_call")):
                    reasons.append("fee_measurement_required")
                elif key in estimate and (finite(estimate[key]) < 0 or finite(estimate[key]) > remaining):
                    reasons.append(f"{key}_estimate_exceeds_budget")
            if reasons:
                excluded[tier] = reasons
            else:
                safe.append(candidate)
        horizon = int(request.metadata.get("placement_horizon", 128))
        if horizon < 1 or len(request.state.get(TRAJECTORY_KEY) or []) >= horizon:
            raise PlacementInfeasible("placement horizon exhausted")
        if not safe:
            raise PlacementInfeasible(f"no feasible placement: {excluded}")
        return safe, excluded, budget

    def _decide_constrained(self, request, profile):
        safe, excluded, budget = self._safe_candidates(request, profile)
        features = {candidate.tier.value: action_features(profile, candidate, self.last_statuses.get(candidate.tier), budget) for candidate in safe}
        model = self.placement_policy
        sampled = bool(request.metadata.get("placement_sample", False))
        probabilities = {}
        reason = "constrained_legacy_fallback"
        failure = self._placement_load_error
        if model is not None and (model.updates > 0 or sampled):
            try:
                action, probabilities = model.choose(features, sample=sampled)
                selected = next(candidate for candidate in safe if candidate.tier.value == action)
                reason = "constrained_actor_critic"
            except (ValueError, ArithmeticError, StopIteration, RuntimeError, TypeError, KeyError) as exc:
                failure = type(exc).__name__
                probabilities = {}
        if not probabilities:
            # Same ordering as the previous scheduler, restricted to safe actions.
            selected = self._select_candidate(safe, profile)
        self._last_placement_context = {
            "decision_id": uuid.uuid4().hex,
            "mode": "constrained", "action": selected.tier.value, "features": features,
            "probabilities": probabilities, "policy_version": model.version if model is not None else None,
            "sampled": bool(probabilities) and sampled, "excluded": excluded,
            "fallback_reason": failure or ("policy_not_trained" if not probabilities else ""),
            "budget": {"limits": dict(budget["limits"]), "spent": dict(budget["spent"])},
        }
        return SchedulingDecision(selected.tier, selected.endpoint, profile, self._model_split(selected.tier, profile), reason=reason)

    def preflight(self, request, allocation):
        if not allocation.metadata.get("placement"):
            return
        self.last_statuses = self.monitor.snapshot(self.resources)
        safe, _, _ = self._safe_candidates(request, self.gate.evaluate(request))
        if not any(candidate.tier == allocation.tier and candidate.endpoint == allocation.endpoint for candidate in safe):
            raise PlacementInfeasible("placement became infeasible before execution")

    def record_outcome(self, request, allocation, result, *, elapsed_ms):
        context = allocation.metadata.get("placement")
        if context is None:
            return
        budget = budget_for(request)
        costs = {"time_ms": max(0., finite(elapsed_ms)), "failures": float(not result.success)}
        fee = result.metadata.get("fee", result.metadata.get("cost"))
        if fee is not None:
            costs["fee"] = max(0., finite(fee))
        elif "fee" in budget["limits"]:
            budget["unpriced_call"] = True
        budget.setdefault("base_spent", dict(budget["spent"]))
        now_ms = time.time() * 1000.
        budget.setdefault("charges", {})[context["decision_id"]] = {"costs": costs, "interval": [now_ms - costs["time_ms"], now_ms]}
        recalculate_spent(budget)
        record = {**context, "costs": costs, "success": result.success, "reward": None,
                  "costs_complete": "fee" in costs or "fee" not in budget["limits"],
                  "terminal": False}
        # Success of the transport is not a semantic quality reward. A trusted
        # evaluation caller supplies reward and terminal status before training.
        request.state.setdefault(TRAJECTORY_KEY, []).append(record)
        result.metadata["placement_feedback"] = record

    def _legacy_decide(self, request: ResourceRequest, profile: TaskProfile) -> SchedulingDecision:
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
