"""Regression coverage for the small chapter-4 runtime alignment."""
import asyncio
import copy
import random

import pytest

from engine import ContextFact, ContextLedgerStore, GraphCheckpoint, StateGraph
from engine.hooks import HookManager, NodeContext
from engine.modules.communication import CommunicationManager, CommunicationPolicy, MessageCapsule
from engine.modules.communication.types import StateDelta
from engine.modules.context import CONTEXT_LEDGER_KEY, ContextInjector
from engine.modules.evaluation import RuleEvaluator
from engine.modules.execution import ExecutorRegistry, InferenceResult, ResilientInferenceRunner
from engine.modules.memory import NoOpMemoryStore
from engine.modules.reasoning import (
    PLAN_IR_KEY, PLAN_VALIDATION_KEY, REASONING_CONTEXT_KEY, COMPLETED_SUBTASKS_KEY,
    DeterministicPlanGenerator, DeterministicPlanRepairer, NeuroSymbolicReasoner,
    PlanIR, SubtaskIR, SymbolicPlanValidator,
)
from engine.modules.scheduling import (
    AdaptiveResourceScheduler, ConstrainedPlacementPolicy, PlacementInfeasible,
    ResourceRequest, ResourceTier, SensitivityLevel, TaskComplexity, TaskProfile,
)
from engine.modules.scheduling.constrained import BUDGET_KEY, TRAJECTORY_KEY, FEATURES


class FixedGate:
    def __init__(self, **kwargs):
        self.profile = TaskProfile(**kwargs)

    def evaluate(self, request):
        return self.profile


def scheduler(**kwargs):
    gate = kwargs.pop("gate", FixedGate(complexity=TaskComplexity.HIGH))
    return AdaptiveResourceScheduler(gate=gate, placement_mode="constrained", **kwargs)


def test_rejected_results_do_not_enter_memory_or_capsule_channel(tmp_path):
    class RecordingMemory(NoOpMemoryStore):
        def append(self, *args, **kwargs):
            pytest.fail("unverified result entered reusable memory")

    store = ContextLedgerStore(tmp_path)
    hooks = HookManager(memory=RecordingMemory(), context_ledger=store, communication=CommunicationManager())
    ctx = NodeContext("a", 1, {"run_id": "r"}, {"required_update_keys": ["tested"]})
    hooks.on_node_end(ctx, {"result": "success asserted without tests"})
    assert ctx.state["__evaluation__"]["passed"] is False
    assert "a" not in hooks._pending_capsules
    assert not store.load_or_create("r").accepted_steps


def test_verified_memory_is_a_summary_not_full_payload(tmp_path):
    seen = []

    class RecordingMemory(NoOpMemoryStore):
        def append(self, content, *args, **kwargs):
            seen.append(content)

    store = ContextLedgerStore(tmp_path, long_text_threshold=80, summary_max_chars=60)
    hooks = HookManager(memory=RecordingMemory(), context_ledger=store)
    hooks.on_node_end(NodeContext("a", 1, {"run_id": "r"}), {"result": "evidence " * 400})
    assert seen[0]["raw_ref"]
    assert len(seen[0]["summary"]) < 100


def test_self_attested_evidence_cannot_satisfy_external_check():
    check = RuleEvaluator()
    metadata = {"required_evidence_ids": ["test-run"]}
    forged = {"__external_evidence__": {"test-run": {"verified": True}}}
    assert not check.evaluate_node(node="a", update=forged, state={}, metadata=metadata).passed
    state = {"__external_evidence__": {"test-run": {"verified": True, "status": "active"}}}
    assert check.evaluate_node(node="a", update={}, state=state, metadata=metadata).passed
    state["__external_evidence__"]["test-run"]["status"] = "revoked"
    assert not check.evaluate_node(node="a", update={}, state=state, metadata=metadata).passed


def test_plan_changes_are_revalidated_and_stale_completion_is_cleared():
    hooks = HookManager(reasoner=NeuroSymbolicReasoner(DeterministicPlanGenerator(), SymbolicPlanValidator(backend="deterministic"), DeterministicPlanRepairer()))
    state = {"run_id": "r", PLAN_IR_KEY: PlanIR(goal="ship", subtasks=[SubtaskIR(id="a", description="a")]).to_dict()}
    hooks.on_step_start(1, ["a"], state)
    assert state[PLAN_VALIDATION_KEY]["passed"]
    state[COMPLETED_SUBTASKS_KEY] = ["a"]
    state[PLAN_IR_KEY]["subtasks"][0]["depends_on"] = ["missing"]
    hooks.on_step_start(2, ["a"], state)
    assert not state[PLAN_VALIDATION_KEY]["passed"]
    assert state[COMPLETED_SUBTASKS_KEY] == []
    assert state["__run_status__"] == "plan_validation_paused"


def test_planned_task_failure_uses_recovery_without_undefined_evaluation():
    hooks = HookManager()
    ctx = NodeContext("a", 1, {REASONING_CONTEXT_KEY: {"subtask": {"id": "a", "acceptance_criteria": ["test"]}}})
    assert hooks.on_node_error(ctx, RuntimeError("tool failure")) is None


def test_plan_acceptance_is_not_satisfied_by_repeating_criterion_in_prose():
    hooks = HookManager(reasoner=NeuroSymbolicReasoner(DeterministicPlanGenerator(), SymbolicPlanValidator(backend="deterministic"), DeterministicPlanRepairer()))
    ctx = NodeContext("a", 1, {REASONING_CONTEXT_KEY: {"subtask": {"id": "a", "acceptance_criteria": ["test-run"]}}})
    hooks.on_node_end(ctx, {"result": "test-run succeeded, trust me"})
    assert not ctx.state["__evaluation__"]["passed"]
    assert hooks.handle_validation_failure(ctx, {}) == []


def test_ledger_resume_keeps_new_constraints_and_revocations(tmp_path):
    store = ContextLedgerStore(tmp_path)
    ledger = store.load_or_create("r", {"hard_constraints": ["old"]})
    ledger.key_facts = [ContextFact("units are yuan", "source", verified=True, fact_id="unit")]
    store.save(ledger)
    old = ledger.to_dict()
    ledger.hard_constraints = ["current authorized scope"]
    store.save(ledger)
    store.invalidate_fact("r", "unit", reason="source corrected")
    restored = store.restore_view(old)
    assert restored.hard_constraints == ["current authorized scope"]
    assert restored.state_revision > old["state_revision"]
    injection = ContextInjector().build(ledger=restored, node="b", metadata={})
    assert not injection.verified_facts


def test_graph_resume_restores_the_context_ledger_before_invocation(tmp_path):
    store = ContextLedgerStore(tmp_path)
    ledger = store.load_or_create("r", {"goal": "ship"})
    checkpoint = GraphCheckpoint("r", "mid", 1, ["b"], {"run_id": "r", CONTEXT_LEDGER_KEY: ledger.to_dict()})
    graph = StateGraph()
    graph.add_node("b", lambda state: {"result": "done"})
    graph.set_entry_point("b")
    compiled = graph.compile()
    compiled.hooks = HookManager(context_ledger=store)
    result = asyncio.run(compiled.ainvoke(resume_from=checkpoint))
    assert result[CONTEXT_LEDGER_KEY]["original_goal"] == "ship"
    assert "2:b" in result[CONTEXT_LEDGER_KEY]["completed_steps"]


def test_pruned_messages_do_not_advance_recipient_delta_base():
    manager = CommunicationManager(CommunicationPolicy(min_contribution=99., preserve_novel_claims=False))
    def capsule(value, protected):
        update = {"output": "same", "value": value, "constraint_delta": {"v": value} if protected else {}}
        return manager.capsule_from_update(update, sender="a", recipients=None, state={}, step=value)
    first, _ = manager.route(capsule(1, True), candidates=["b"])
    skipped, _ = manager.route(capsule(2, False), candidates=["b"])
    last, _ = manager.route(capsule(3, True), candidates=["b", "c"])
    assert first and not skipped
    assert last[0].capsule.state_delta.base_revision == 1
    assert last[0].capsule.state_delta.changed["value"] == 3
    assert last[1].capsule.state_delta.base_revision == 0


def test_mailbox_and_delta_base_survive_restart():
    manager = CommunicationManager()
    first = manager.capsule_from_update({"output": "data", "value": 1}, sender="a", recipients=["b"], state={}, step=1)
    manager.route(first, candidates=["b"])
    restored = CommunicationManager()
    restored.restore(manager.snapshot())
    assert restored.receive("b")[0].capsule.claim == "data"
    assert not restored.route(first, candidates=["b"])[0]
    next_message = restored.capsule_from_update({"output": "next", "value": 2}, sender="a", recipients=["b"], state={}, step=2)
    delivered, _ = restored.route(next_message, candidates=["b"])
    assert delivered[0].capsule.state_delta.base_revision == 1


def test_unknown_delta_base_requests_snapshot_instead_of_applying():
    manager = CommunicationManager()
    capsule = MessageCapsule(sender="a", claim="new", state_delta=StateDelta(base_revision=7, revision=8, changed={"x": 1}))
    delivered, events = manager.route(capsule, candidates=["b"])
    assert not delivered
    assert events[-1]["type"] == "capsule_snapshot_required"


def test_compressor_cannot_modify_protected_semantics():
    class CorruptCompressor:
        def compress(self, capsule, **kwargs):
            altered = copy.deepcopy(capsule)
            altered.claim = "fabricated result"
            return altered
    manager = CommunicationManager(CommunicationPolicy(max_capsule_tokens=1), compressor=CorruptCompressor())
    delivered, events = manager.route(MessageCapsule(sender="a", claim="original result"), candidates=["b"])
    assert delivered[0].capsule.claim == "original result"
    assert any(event["type"] == "capsule_compression_rejected" for event in events)


def test_revocation_notifications_bypass_ordinary_contribution_gate():
    manager = CommunicationManager(CommunicationPolicy(min_contribution=99., preserve_novel_claims=False))
    assert manager.route(MessageCapsule(sender="a", kind="revoke", claim="retract fact"), candidates=["b"])[0]


def test_complementary_team_covers_distinct_capabilities():
    from engine.modules.workflow_runtime import WorkflowNodeRuntimeFactory
    from engine.orchestrator import AgentSpec
    ranked = [{"member": AgentSpec(id=ident, name=name, description=name), "score": score, "breakdown": {}} for ident, name, score in [("a", "search", .9), ("b", "search", .89), ("c", "test", .7)]]
    selected = WorkflowNodeRuntimeFactory._select_complementary(ranked, {}, ["search", "test"], 2)
    assert {item["member"].id for item in selected} == {"a", "c"}


def test_policy_is_masked_before_selection_for_sensitive_tasks():
    model = ConstrainedPlacementPolicy()
    model.updates = 1
    seen = []
    def choose(features, **kwargs):
        seen.extend(features)
        return "edge", {key: 1. / len(features) for key in features}
    model.choose = choose
    sched = scheduler(placement_policy=model, gate=FixedGate(complexity=TaskComplexity.HIGH, sensitivity=SensitivityLevel.SECRET))
    allocation = sched.acquire(ResourceRequest(node="a"))
    assert allocation.tier == ResourceTier.EDGE
    assert seen == ["edge"]
    assert "untrusted_workspace" in allocation.metadata["placement"]["excluded"]["cloud"]


def test_broken_policy_falls_back_inside_safe_candidates():
    model = ConstrainedPlacementPolicy()
    model.updates = 1
    def broken(*args, **kwargs):
        raise RuntimeError("model unavailable")
    model.choose = broken
    allocation = scheduler(placement_policy=model, gate=FixedGate(complexity=TaskComplexity.HIGH, sensitivity=SensitivityLevel.SECRET)).acquire(ResourceRequest(node="a"))
    assert allocation.tier == ResourceTier.EDGE
    assert allocation.metadata["placement"]["fallback_reason"] == "RuntimeError"


def test_constrained_path_rejects_unauthorized_capability_downgrade():
    sched = scheduler(gate=FixedGate(complexity=TaskComplexity.EXTREME, sensitivity=SensitivityLevel.SECRET))
    with pytest.raises(PlacementInfeasible):
        sched.acquire(ResourceRequest(node="a"))


def test_budget_filters_expensive_actions_and_rejects_unpriced_calls():
    sched = scheduler()
    request = ResourceRequest(node="a", metadata={"placement_budget": {"fee": 2}, "placement_estimates": {"cloud": {"fee": 4}, "edge": {"fee": 1}}})
    assert sched.acquire(request).tier == ResourceTier.EDGE
    with pytest.raises(PlacementInfeasible):
        sched.acquire(ResourceRequest(node="a", metadata={"placement_budget": {"fee": 2}}))


def test_failure_budget_stops_retry_and_records_all_consumption():
    class FailingExecutor:
        calls = 0
        def run(self, request):
            self.calls += 1
            return InferenceResult("", "test", "test", success=False, retryable=True, error="503 unavailable", metadata={"fee": .25})
    executor = FailingExecutor()
    request = ResourceRequest(node="a", metadata={"placement_budget": {"failures": 1}})
    result = ResilientInferenceRunner(scheduler=scheduler(), registry=ExecutorRegistry({tier: executor for tier in ResourceTier}), max_attempts=3).run(resource_request=request, prompt="test")
    assert not result.success and result.metadata["placement_blocked"]
    assert executor.calls == 1
    assert request.state[BUDGET_KEY]["spent"]["fee"] == .25
    assert request.state[BUDGET_KEY]["spent"]["failures"] == 1
    assert request.state[TRAJECTORY_KEY][0]["reward"] is None


def test_preflight_rechecks_latest_authorization():
    sched = scheduler()
    request = ResourceRequest(node="a")
    allocation = sched.acquire(request)
    request.metadata["allowed_tiers"] = []
    with pytest.raises(PlacementInfeasible):
        sched.preflight(request, allocation)


def episode(policy):
    features = {"edge": [1., 1.] + [0.] * (FEATURES - 2), "cloud": [1., 0., 1.] + [0.] * (FEATURES - 3)}
    action, probabilities = policy.choose(features, sample=True, rng=random.Random(7))
    return [{"features": features, "action": action, "probabilities": probabilities, "policy_version": policy.version,
        "sampled": True, "reward": 1., "costs": {"fee": 2., "time_ms": 10., "failures": 0.}, "terminal": True}]


def test_actor_critic_learns_and_updates_budget_multiplier(tmp_path):
    model = ConstrainedPlacementPolicy(seed=5)
    before = copy.deepcopy(model.weights)
    report = model.learn_episode(episode(model), {"fee": 1.})
    assert model.weights != before
    assert report["multipliers"]["fee"] > 0
    loaded = ConstrainedPlacementPolicy.load(model.save(tmp_path / "policy.json"))
    assert loaded.probabilities(episode(loaded)[0]["features"]) == model.probabilities(episode(model)[0]["features"])


@pytest.mark.parametrize("mutation", ["stale", "deterministic", "partial", "missing_reward", "wrong_behavior", "unpriced"])
def test_training_rejects_invalid_or_incomplete_episodes(mutation):
    model = ConstrainedPlacementPolicy()
    steps = episode(model)
    if mutation == "stale": steps[0]["policy_version"] = -1
    if mutation == "deterministic": steps[0]["sampled"] = False
    if mutation == "partial": steps[0]["terminal"] = False
    if mutation == "missing_reward": steps[0]["reward"] = None
    if mutation == "wrong_behavior": steps[0]["probabilities"]["edge"] = .99
    if mutation == "unpriced": steps[0]["costs_complete"] = False
    with pytest.raises(ValueError):
        model.learn_episode(steps, {"fee": 1.})


def test_missing_policy_file_preserves_legacy_mode():
    sched = AdaptiveResourceScheduler(gate=FixedGate(complexity=TaskComplexity.HIGH), placement_policy_path="nonexistent-policy.json", placement_mode="auto")
    allocation = sched.acquire(ResourceRequest(node="a"))
    assert allocation.tier == ResourceTier.CLOUD
    assert allocation.metadata["reason"] == "cloud_selected_for_high_complexity"
    assert allocation.metadata["placement_fallback"] == "FileNotFoundError"


def test_auto_connections_use_constrained_retry_and_feedback(monkeypatch):
    from types import SimpleNamespace
    from engine.modules.agent_runtime import AgentRuntimeFactory

    class Connections:
        def auto_status(self):
            return {"ready": True}
        def default_for_tier(self, tier):
            return SimpleNamespace(id=tier)

    sched = scheduler()
    factory = AgentRuntimeFactory(scheduler=sched, model_connection_store=Connections(), max_attempts=2)
    monkeypatch.setattr(factory, "_connection_scheduler", lambda *args, **kwargs: sched)
    def execute(ident, *args):
        return InferenceResult("done" if ident == "edge" else "", "fake", ident,
            success=ident == "edge", error="429 rate limit" if ident == "cloud" else None)
    monkeypatch.setattr(factory, "_run_pinned_model", execute)
    request = ResourceRequest(node="a")
    result = factory._run_auto_model(request, "test", "")
    assert result.success and result.metadata["actual_tier"] == "edge"
    assert result.metadata["selected_tier"] == "cloud"
    assert len(request.state[TRAJECTORY_KEY]) == 2
    assert request.state[BUDGET_KEY]["spent"]["failures"] == 1


def test_training_entry_produces_loadable_deployable_policy(tmp_path):
    from examples.edge_cloud_scheduler.train_constrained_policy import train
    episodes = [{"limits": {"fee": 1.}, "steps": [{"profile": {"complexity": "high", "sensitivity": "public"},
        "estimates": {"edge": {"fee": .1}, "cloud": {"fee": .2}},
        "outcomes": {"edge": {"reward": .8, "fee": .1, "time_ms": 30.}, "cloud": {"reward": 1., "fee": .2, "time_ms": 50.}}}]}]
    path = tmp_path / "policy.json"
    report = train(episodes, path, epochs=3)
    assert report["updates"] == 3
    sched = AdaptiveResourceScheduler(gate=FixedGate(complexity=TaskComplexity.HIGH), placement_policy_path=str(path), placement_mode="auto")
    allocation = sched.acquire(ResourceRequest(node="a"))
    assert allocation.metadata["reason"] == "constrained_actor_critic"
    assert allocation.metadata["placement"]["sampled"] is False


def test_parallel_budget_merge_deduplicates_charges_and_overlapping_time():
    from engine.modules.scheduling.constrained import merge_budget
    old = {"limits": {"fee": 5}, "base_spent": {}, "spent": {"fee": 1, "time_ms": 10},
        "charges": {"a": {"costs": {"fee": 1}, "interval": [10., 20.]}}}
    new = {"limits": {"fee": 5}, "base_spent": {}, "spent": {"fee": 2, "time_ms": 10},
        "charges": {"b": {"costs": {"fee": 2}, "interval": [15., 25.]}}}
    result = merge_budget(old, new)
    assert result["spent"]["fee"] == 3
    assert result["spent"]["time_ms"] == 15
    assert merge_budget(result, result)["spent"] == result["spent"]


def test_runtime_budget_persists_between_agent_nodes():
    from engine.modules.agent_runtime import AgentRuntimeFactory
    from engine.orchestrator import AgentSpec
    class GoodExecutor:
        def run(self, request):
            return InferenceResult("accepted answer", "fake", "fake", metadata={"fee": .1, "simulated": True})
    sched = scheduler()
    factory = AgentRuntimeFactory(scheduler=sched, registry=ExecutorRegistry({tier: GoodExecutor() for tier in ResourceTier}))
    graph = StateGraph()
    for name in ["a", "b"]:
        node = factory(AgentSpec(id=name, name=name, config={"disable_answer_rewrite": True}))
        graph.add_node(name, node.func)
    graph.set_entry_point("a")
    graph.add_edge("a", "b")
    result = asyncio.run(graph.compile().ainvoke({"input": "test"}))
    assert result[BUDGET_KEY]["spent"]["fee"] == pytest.approx(.2)
    assert len(result[TRAJECTORY_KEY]) == 2
