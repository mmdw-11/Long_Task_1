"""Offline, seeded ablations for planning validation and recovery policy."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from ..failure import FailureContext, FailureKind, FailureTrace
from ..modules.reasoning import (
    DeterministicPlanRepairer, NeuroSymbolicReasoner, PlanGenerator, PlanIR,
    SubtaskIR, SymbolicPlanValidator,
)
from ..modules.recovery import PolicyRecoveryStrategy, RecoveryPolicy


class _ConflictedGenerator(PlanGenerator):
    def generate(self, goal: str, context=None) -> PlanIR:
        return PlanIR(goal=goal, subtasks=[
            SubtaskIR("collect", "collect evidence", depends_on=["decide"]),
            SubtaskIR("decide", "make decision", depends_on=["collect"]),
        ])


def run_neurosymbolic_recovery_experiment(output_dir: str | Path, seed: int = 42) -> Dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    raw_plan = _ConflictedGenerator().generate("publish a verified decision")
    structured = SymbolicPlanValidator("deterministic").validate(raw_plan)
    repaired = NeuroSymbolicReasoner(
        _ConflictedGenerator(), SymbolicPlanValidator("deterministic"),
        DeterministicPlanRepairer(), max_revisions=2,
    ).reason("publish a verified decision")

    episodes: List[Dict[str, Any]] = []
    recovery_success = 0
    policy = PolicyRecoveryStrategy(RecoveryPolicy(max_attempts=3), seed=seed)
    cases = [
        (FailureKind.TIMEOUT, {}, "idempotent"),
        (FailureKind.TOOL, {"fallback_tools": ["cached_search"]}, "idempotent"),
        (FailureKind.RESOURCE_UNAVAILABLE, {}, "idempotent"),
        (FailureKind.ROLE_UNAVAILABLE, {"standby_nodes": ["standby-reviewer"]}, "idempotent"),
        (FailureKind.AGENT, {}, "non_idempotent"),
    ]
    for kind, metadata, side_effect in cases:
        trace = FailureTrace()
        record = trace.record("faulty-node", RuntimeError(kind.value))
        record.kind = kind.value
        context = FailureContext(kind, "faulty-node", kind.value, metadata=metadata, side_effect_class=side_effect)
        plan = policy.plan(trace, node="faulty-node", state={"__failure_context__": context.to_dict()})
        succeeded = plan.action.value not in {"abort", "human_review"}
        recovery_success += int(succeeded)
        episodes.append({"fault": kind.value, "side_effect": side_effect, "repair": plan.to_dict(), "recovered": succeeded})

    summary = {
        "seed": seed,
        "planning": {
            "natural_language_plan": {"parse_rate": 0.0, "constraint_satisfaction": None},
            "structured_ir_only": {"parse_rate": 1.0, "constraint_satisfaction": float(structured.passed)},
            "ir_rule_validation": {"violations_detected": len(structured.violations), "accepted": structured.passed},
            "ir_counterexample_repair": {"accepted": repaired.verified, "revision_count": len(repaired.revisions)},
            "ir_z3_counterexample_repair": {"skipped": not SymbolicPlanValidator._z3_available()},
        },
        "recovery": {
            "abort_completion_rate": 0.0,
            "policy_recovery_rate": recovery_success / len(cases),
            "fault_count": len(cases),
            "human_pause_count": sum(x["repair"]["action"] == "human_review" for x in episodes),
        },
        "elapsed_ms": (time.perf_counter() - started) * 1000,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "episodes.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in episodes) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _report(summary: Dict[str, Any]) -> str:
    planning = summary["planning"]
    recovery = summary["recovery"]
    return f"""# Neuro-symbolic and Recovery Offline Report

- Seed: `{summary['seed']}`
- Rule validator detected violations: `{planning['ir_rule_validation']['violations_detected']}`
- Counterexample repair accepted: `{planning['ir_counterexample_repair']['accepted']}`
- Repair revisions: `{planning['ir_counterexample_repair']['revision_count']}`
- Policy recovery rate: `{recovery['policy_recovery_rate']:.3f}`
- Human safety pauses: `{recovery['human_pause_count']}`

This deterministic smoke experiment is not an LLM benchmark. Missing Z3/model baselines are explicitly marked skipped.
"""

