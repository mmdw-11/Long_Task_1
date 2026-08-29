import pytest

from engine import StateGraph
from engine.hooks import HookManager
from engine.modules.reasoning import (
    ConstraintIR, ConstraintKind, DeterministicPlanGenerator,
    DeterministicPlanRepairer, NeuroSymbolicReasoner, PlanGenerator, PlanIR,
    SubtaskIR, SymbolicPlanValidator, ValidationStatus,
)


class CyclicGenerator(PlanGenerator):
    def generate(self, goal, context=None):
        return PlanIR(goal=goal, subtasks=[
            SubtaskIR("a", "analyze", depends_on=["b"]),
            SubtaskIR("b", "execute", depends_on=["a"]),
        ])


def test_plan_ir_roundtrip_and_dag_counterexample():
    plan = CyclicGenerator().generate("ship safely")
    assert PlanIR.from_dict(plan.to_dict()).subtasks[0].depends_on == ["b"]
    result = SymbolicPlanValidator(backend="deterministic").validate(plan)
    assert result.status == ValidationStatus.INVALID
    assert result.counterexamples[0].violation_code == "dependency_cycle"


def test_validator_checks_role_resource_deadline_and_evidence():
    task = SubtaskIR(
        "deploy", "deploy", duration=4, role="operator",
        resource_requirements={"gpu": 2}, acceptance_criteria=[],
    )
    plan = PlanIR(
        goal="deploy", subtasks=[task], resources={"gpu": 1},
        hard_constraints=[
            ConstraintIR(ConstraintKind.DEADLINE, source="deploy", value=2),
            ConstraintIR(ConstraintKind.REQUIRED_EVIDENCE, source="deploy", value="healthcheck"),
        ],
    )
    result = SymbolicPlanValidator(
        backend="deterministic", available_roles={"reviewer"}
    ).validate(plan)
    codes = {item.code for item in result.violations}
    assert {"role_unavailable", "resource_capacity", "deadline_missed", "required_evidence"} <= codes


def test_counterexample_repair_closes_the_loop():
    reasoner = NeuroSymbolicReasoner(
        CyclicGenerator(), SymbolicPlanValidator(backend="deterministic"),
        DeterministicPlanRepairer(), max_revisions=2,
    )
    outcome = reasoner.reason("ship safely")
    assert outcome.verified
    assert outcome.plan.revision == 1
    assert outcome.feedback_capsules
    assert outcome.revisions[0].counterexamples[0].violation_code == "dependency_cycle"


@pytest.mark.asyncio
async def test_reasoner_is_integrated_before_graph_execution():
    graph = StateGraph()
    graph.add_node("worker", lambda state: {"done": True})
    graph.set_entry_point("worker")
    hooks = HookManager(reasoner=NeuroSymbolicReasoner(
        DeterministicPlanGenerator(), SymbolicPlanValidator(backend="deterministic"),
        DeterministicPlanRepairer(),
    ))
    compiled = graph.compile()
    compiled.hooks = hooks
    events = [event async for event in compiled.astream({"goal": "finish", "run_id": "reasoning-run"})]
    assert any(event["type"] == "plan_verified" for event in events)
    final = events[-1]["state"]
    assert final["__plan_validation__"]["passed"] is True
    assert final["done"] is True
