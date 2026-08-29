import pytest

from engine import StateGraph
from engine.failure import FailureContext, FailureKind, FailureTrace
from engine.hooks import HookManager
from engine.modules.fault_injection import FaultInjector, FaultSpec
from engine.modules.recovery import (
    PolicyRecoveryStrategy, RecoveryAction, RecoveryPolicy,
)


def _plan(kind, metadata=None, side="idempotent", attempts=1):
    trace = FailureTrace()
    for index in range(attempts):
        record = trace.record("node", RuntimeError("failure"), step=index)
        record.kind = kind.value
    state = {"__failure_context__": FailureContext(
        kind=kind, node="node", message="failure", attempt=attempts,
        side_effect_class=side, metadata=metadata or {},
    ).to_dict()}
    return PolicyRecoveryStrategy(RecoveryPolicy(max_attempts=3)).plan(trace, node="node", state=state)


def test_recovery_policy_is_bounded_and_side_effect_aware():
    assert _plan(FailureKind.TIMEOUT, attempts=1).action == RecoveryAction.RETRY
    assert _plan(FailureKind.TIMEOUT, attempts=3).action == RecoveryAction.ABORT
    assert _plan(FailureKind.AGENT, side="non_idempotent").action == RecoveryAction.HUMAN_REVIEW
    assert _plan(FailureKind.AGENT, {"compensation_node": "undo"}, "compensatable").action == RecoveryAction.COMPENSATE


def test_recovery_policy_maps_cross_layer_faults():
    assert _plan(FailureKind.TOOL, {"fallback_tools": ["backup"]}).action == RecoveryAction.FALLBACK_TOOL
    assert _plan(FailureKind.RESOURCE_UNAVAILABLE).action == RecoveryAction.MIGRATE_RESOURCE
    assert _plan(FailureKind.ROLE_UNAVAILABLE, {"standby_nodes": ["sibling"]}).targets == ["sibling"]
    assert _plan(FailureKind.PLAN_CONFLICT, {"replan_nodes": ["planner"]}).action == RecoveryAction.REPLAN
    assert _plan(FailureKind.EVIDENCE_MISSING, {"evidence_repair_nodes": ["researcher"]}).targets == ["researcher"]


@pytest.mark.asyncio
async def test_fault_injection_recovers_with_trace_and_retry():
    calls = {"count": 0}

    async def worker(state):
        calls["count"] += 1
        return {"done": True}

    graph = StateGraph()
    graph.add_node("worker", worker, metadata={"idempotency": "idempotent"})
    graph.set_entry_point("worker")
    hooks = HookManager(
        recovery_strategy=PolicyRecoveryStrategy(RecoveryPolicy(max_attempts=2)),
        fault_injector=FaultInjector([FaultSpec("timeout", node="worker", attempt=1)]),
    )
    compiled = graph.compile()
    compiled.hooks = hooks
    events = [event async for event in compiled.astream({"run_id": "fault-run"})]
    kinds = [event["type"] for event in events]
    assert "failure_detected" in kinds and "recovery_planned" in kinds
    assert events[-1]["state"]["done"] is True
    assert events[-1]["state"]["__recovery_trace__"]
    assert calls["count"] == 1
