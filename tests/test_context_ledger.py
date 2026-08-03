import json
import asyncio

from engine import (
    BUDGET_PAUSED,
    CONTEXT_LEDGER_KEY,
    PAUSE_REASON_KEY,
    RUN_STATUS_KEY,
    ContextBudgetController,
    ContextCheckpointStore,
    ContextFact,
    ContextLedgerStore,
    ContextPolicy,
    DriftDetector,
    Orchestrator,
)
from engine import StateGraph
from engine.hooks import HookManager
from engine.modules.recovery import RecoveryAction, RecoveryStrategy, RepairPlan


def test_context_ledger_initializes_from_run_state(tmp_path):
    store = ContextLedgerStore(tmp_path)

    ledger = store.load_or_create(
        "run-1",
        {
            "goal": "finish the long task",
            "hard_constraints": ["do not leak secrets"],
            "current_plan": ["step one", "step two"],
        },
    )

    assert ledger.run_id == "run-1"
    assert ledger.original_goal == "finish the long task"
    assert ledger.hard_constraints == ["do not leak secrets"]
    assert ledger.current_plan == ["step one", "step two"]
    assert store.path_for("run-1").exists()
    memory_text = store.memory_path_for("run-1").read_text(encoding="utf-8")
    assert "## Original Goal" in memory_text
    assert "finish the long task" in memory_text
    assert "do not leak secrets" in memory_text


def test_context_ledger_records_node_completion_and_facts(tmp_path):
    store = ContextLedgerStore(tmp_path)
    state = {"run_id": "run-2", "goal": "collect facts"}

    store.on_step_start(run_id="run-2", step=1, frontier=["worker"], state=state)
    ledger = store.on_node_end(
        run_id="run-2",
        node="worker",
        step=1,
        update={"result": "weather is rainy", "score": 0.9},
        state=state,
    )

    assert ledger.completed_steps == ["1:worker"]
    assert ledger.pending_steps == []
    assert ledger.tool_summaries[0].tool_name == "worker"
    assert "weather is rainy" in ledger.tool_summaries[0].short_summary
    assert ledger.key_facts[0].source == "node_update"

    reloaded = json.loads(store.path_for("run-2").read_text(encoding="utf-8"))
    assert reloaded["completed_steps"] == ["1:worker"]
    memory_text = store.memory_path_for("run-2").read_text(encoding="utf-8")
    assert "## Completed" in memory_text
    assert "1:worker" in memory_text
    assert "weather is rainy" in memory_text


def test_context_ledger_archives_long_node_update_with_raw_ref(tmp_path):
    store = ContextLedgerStore(tmp_path, long_text_threshold=80, summary_max_chars=40)
    long_output = "important raw browser page " + ("x" * 200)

    ledger = store.on_node_end(
        run_id="raw-ref-run",
        node="browser",
        step=1,
        update={"result": long_output},
        state={"run_id": "raw-ref-run", "goal": "archive long result"},
    )

    summary = ledger.tool_summaries[0]
    assert summary.raw_ref
    assert len(summary.short_summary) < len(long_output)
    raw_path = tmp_path / summary.raw_ref
    assert raw_path.exists()
    assert "important raw browser page" in raw_path.read_text(encoding="utf-8")

    memory_text = store.memory_path_for("raw-ref-run").read_text(encoding="utf-8")
    assert summary.raw_ref in memory_text


def test_context_ledger_records_failures(tmp_path):
    store = ContextLedgerStore(tmp_path)
    state = {"run_id": "run-3", "input": "recover"}
    error = RuntimeError("tool timeout")

    ledger = store.on_node_error(
        run_id="run-3",
        node="browser",
        step=2,
        error=error,
        state=state,
        recoverable=True,
    )

    failure = ledger.failure_summaries[0]
    assert failure.node == "browser"
    assert failure.step == 2
    assert failure.error_type == "RuntimeError"
    assert failure.message == "tool timeout"
    assert failure.recoverable is True
    memory_text = store.memory_path_for("run-3").read_text(encoding="utf-8")
    assert "## Recent Failures" in memory_text
    assert "RuntimeError: tool timeout" in memory_text


def test_orchestrator_writes_context_ledger_to_state_and_disk(tmp_path):
    store = ContextLedgerStore(tmp_path)
    orch = Orchestrator()
    orch.set_context_ledger(store)
    a = orch.create_agent("a")
    b = orch.add_sub_agent(a, name="b")
    orch.set_entry(a)

    state = asyncio.run(
        orch.build_graph().ainvoke(
            {
                "run_id": "run-graph",
                "goal": "preserve context",
                "hard_constraints": ["keep original goal"],
                "input": "hello",
            }
        )
    )

    assert CONTEXT_LEDGER_KEY in state
    ledger_data = state[CONTEXT_LEDGER_KEY]
    assert ledger_data["run_id"] == "run-graph"
    assert ledger_data["original_goal"] == "preserve context"
    assert "keep original goal" in ledger_data["hard_constraints"]
    assert ledger_data["completed_steps"] == ["1:a", "2:b"]

    persisted = json.loads(store.path_for("run-graph").read_text(encoding="utf-8"))
    assert persisted["completed_steps"] == ["1:a", "2:b"]
    assert any("a:" in item["short_summary"] for item in persisted["tool_summaries"])
    memory_text = store.memory_path_for("run-graph").read_text(encoding="utf-8")
    assert "preserve context" in memory_text
    assert "1:a" in memory_text
    assert "2:b" in memory_text


class RetryOnce(RecoveryStrategy):
    def plan(self, failure_trace, *, node, state):
        return RepairPlan(action=RecoveryAction.REROUTE, targets=["repair"], reason="test")


def test_context_ledger_records_graph_node_error(tmp_path):
    store = ContextLedgerStore(tmp_path)
    graph = StateGraph()

    async def bad(state):
        raise ValueError("bad output")

    async def repair(state):
        return {"fixed": True}

    graph.add_node("bad", bad)
    graph.add_node("repair", repair)
    graph.set_entry_point("bad")
    compiled = graph.compile()
    compiled.hooks = HookManager(
        context_ledger=store,
        recovery_strategy=RetryOnce(),
    )

    state = asyncio.run(
        compiled.ainvoke({"run_id": "run-error", "goal": "handle failure"})
    )

    assert state["fixed"] is True
    persisted = json.loads(store.path_for("run-error").read_text(encoding="utf-8"))
    assert persisted["failure_summaries"][0]["node"] == "bad"
    assert persisted["failure_summaries"][0]["recoverable"] is True
    assert persisted["completed_steps"] == ["2:repair"]


def test_context_budget_allows_execution_when_under_limit(tmp_path):
    store = ContextLedgerStore(tmp_path)
    orch = Orchestrator()
    orch.set_context_ledger(store)
    orch.set_context_budget(ContextBudgetController(max_context_tokens=1000))
    node = orch.create_agent("worker")
    orch.set_entry(node)

    state = asyncio.run(
        orch.build_graph().ainvoke({"run_id": "budget-ok", "input": "small task"})
    )

    assert state.get(RUN_STATUS_KEY) is None
    assert state["messages"][0]["agent"] == "worker"
    ledger = json.loads(store.path_for("budget-ok").read_text(encoding="utf-8"))
    assert ledger["budget"]["paused"] is False
    assert ledger["budget"]["used_context_tokens"] > 0


def test_context_budget_pauses_before_node_execution(tmp_path):
    store = ContextLedgerStore(tmp_path)
    orch = Orchestrator()
    orch.set_context_ledger(store)
    orch.set_context_budget(
        ContextBudgetController(max_context_tokens=20, reserved_output_tokens=5)
    )
    node = orch.create_agent("worker")
    orch.set_entry(node)

    state = asyncio.run(
        orch.build_graph().ainvoke(
            {
                "run_id": "budget-paused",
                "goal": "x" * 200,
                "input": "x" * 200,
            }
        )
    )

    assert state[RUN_STATUS_KEY] == BUDGET_PAUSED
    assert "context budget exceeded" in state[PAUSE_REASON_KEY]
    assert "messages" not in state

    ledger = json.loads(store.path_for("budget-paused").read_text(encoding="utf-8"))
    assert ledger["budget"]["paused"] is True
    assert "context budget exceeded" in ledger["budget"]["pause_reason"]
    memory_text = store.memory_path_for("budget-paused").read_text(encoding="utf-8")
    assert "paused=True" in memory_text
    assert "context budget exceeded" in memory_text


def test_context_injector_adds_goal_constraints_and_fact_boundaries(tmp_path):
    store = ContextLedgerStore(tmp_path)
    ledger = store.load_or_create(
        "inject-run",
        {
            "goal": "complete the report",
            "hard_constraints": ["never omit tests"],
        },
    )
    ledger.key_facts.append(
        ContextFact(
            text="database migration passed",
            source="test",
            confidence=0.9,
            verified=True,
            node="tester",
            step=1,
        )
    )
    ledger.key_facts.append(
        ContextFact(
            text="browser result may be stale",
            source="browser",
            confidence=0.4,
            verified=False,
            node="browser",
            step=2,
        )
    )
    store.save(ledger)

    orch = Orchestrator()
    orch.set_context_ledger(store)
    node = orch.create_agent(
        "worker",
        description="writer node",
        config={"objective": "draft final section", "output_contract": "return markdown"},
    )
    orch.set_entry(node)

    state = asyncio.run(
        orch.build_graph().ainvoke({"run_id": "inject-run", "input": "continue"})
    )

    injection = state["__context_injection__"]
    assert injection["original_goal"] == "complete the report"
    assert injection["hard_constraints"] == ["never omit tests"]
    assert injection["current_node_role"] == "writer node"
    assert injection["current_step_objective"] == "draft final section"
    assert injection["expected_output_contract"] == "return markdown"
    assert injection["verified_facts"] == ["database migration passed"]
    assert injection["things_not_to_assume"] == ["browser result may be stale"]
    assert "Original Goal: complete the report" in state["worker"]
    assert "never omit tests" in state["worker"]


def test_post_execution_validation_marks_unverified_and_records_failure(tmp_path):
    store = ContextLedgerStore(tmp_path)
    graph = StateGraph()

    async def worker(state):
        return {"result": "missing required output key"}

    graph.add_node(
        "worker",
        worker,
        metadata={"required_update_keys": ["final_answer"]},
    )
    graph.set_entry_point("worker")
    compiled = graph.compile()
    compiled.hooks = HookManager(context_ledger=store)

    state = asyncio.run(
        compiled.ainvoke({"run_id": "validation-run", "goal": "validate output"})
    )

    assert state["__evaluation__"]["passed"] is False
    assert "missing required update key: final_answer" in state["__evaluation__"]["findings"]
    assert "result" not in state
    assert state["__run_status__"] == "validation_failed"

    persisted = json.loads(store.path_for("validation-run").read_text(encoding="utf-8"))
    assert persisted["key_facts"][0]["verified"] is False
    assert persisted["failure_summaries"][0]["error_type"] == "EvaluationFailed"
    assert "final_answer" in persisted["failure_summaries"][0]["message"]


def test_post_execution_validation_marks_valid_fact_verified(tmp_path):
    store = ContextLedgerStore(tmp_path)
    graph = StateGraph()

    async def worker(state):
        return {"final_answer": "done"}

    graph.add_node(
        "worker",
        worker,
        metadata={"required_update_keys": ["final_answer"]},
    )
    graph.set_entry_point("worker")
    compiled = graph.compile()
    compiled.hooks = HookManager(context_ledger=store)

    state = asyncio.run(
        compiled.ainvoke({"run_id": "validation-ok", "goal": "validate output"})
    )

    assert state["__evaluation__"]["passed"] is True
    persisted = json.loads(store.path_for("validation-ok").read_text(encoding="utf-8"))
    assert persisted["key_facts"][0]["verified"] is True
    assert persisted["failure_summaries"] == []


def test_post_execution_validation_can_reroute_without_merging_bad_update(tmp_path):
    store = ContextLedgerStore(tmp_path)
    graph = StateGraph()

    async def worker(state):
        return {"bad_result": "should not enter state"}

    async def repair(state):
        return {"fixed": True}

    graph.add_node(
        "worker",
        worker,
        metadata={
            "required_update_keys": ["final_answer"],
            "validation_failure_action": "reroute",
            "validation_failure_targets": ["repair"],
        },
    )
    graph.add_node("repair", repair)
    graph.set_entry_point("worker")
    compiled = graph.compile()
    compiled.hooks = HookManager(context_ledger=store)

    state = asyncio.run(
        compiled.ainvoke({"run_id": "validation-reroute", "goal": "repair bad output"})
    )

    assert "bad_result" not in state
    assert state["fixed"] is True


def test_drift_detector_records_repeated_node_pattern(tmp_path):
    store = ContextLedgerStore(tmp_path)
    graph = StateGraph()

    async def loop(state):
        return {"count": state.get("count", 0) + 1}

    graph.add_node("loop", loop)
    graph.set_entry_point("loop")
    graph.add_loop("loop", lambda state: state.get("count", 0) < 4)
    compiled = graph.compile()
    compiled.hooks = HookManager(
        context_ledger=store,
        drift_detector=DriftDetector(repeat_node_limit=2),
    )

    state = asyncio.run(
        compiled.ainvoke({"run_id": "drift-run", "goal": "detect loops"})
    )

    assert state["count"] == 4
    assert state["__context_drift__"]["drifted"] is True
    assert "node loop repeated 2 times" in state["__context_drift__"]["reasons"]

    persisted = json.loads(store.path_for("drift-run").read_text(encoding="utf-8"))
    assert any(
        item["error_type"] == "ContextDrift"
        for item in persisted["failure_summaries"]
    )


def test_drift_detector_records_semantically_repeated_attempts(tmp_path):
    store = ContextLedgerStore(tmp_path)
    graph = StateGraph()

    async def edit_a(state):
        return {"result": "retry edit src/app.py with same patch failure 1"}

    async def edit_b(state):
        return {"result": "retry edit src/app.py with same patch failure 2"}

    async def edit_c(state):
        return {"result": "retry edit src/app.py with same patch failure 3"}

    graph.add_node("edit_a", edit_a)
    graph.add_node("edit_b", edit_b)
    graph.add_node("edit_c", edit_c)
    graph.set_entry_point("edit_a")
    graph.add_edge("edit_a", "edit_b")
    graph.add_edge("edit_b", "edit_c")
    compiled = graph.compile()
    compiled.hooks = HookManager(
        context_ledger=store,
        drift_detector=DriftDetector(
            repeat_node_limit=0,
            repeated_summary_limit=3,
        ),
    )

    state = asyncio.run(
        compiled.ainvoke({"run_id": "semantic-drift-run", "goal": "detect repeated attempts"})
    )

    assert state["__context_drift__"]["drifted"] is True
    assert any(
        "similar output summary repeated 3 times" in reason
        for reason in state["__context_drift__"]["reasons"]
    )


def test_sub_agent_output_is_isolated_until_parent_validation(tmp_path):
    store = ContextLedgerStore(tmp_path)
    orch = Orchestrator()
    orch.set_context_ledger(store)
    parent = orch.create_agent("parent")
    orch.add_sub_agent(parent, name="child")
    orch.set_entry(parent)

    state = asyncio.run(
        orch.build_graph().ainvoke({"run_id": "isolation-run", "input": "delegate"})
    )

    assert state["__evaluation__"]["passed"] is False
    assert "sub-agent output requires parent validation" in state["__evaluation__"]["findings"]

    persisted = json.loads(store.path_for("isolation-run").read_text(encoding="utf-8"))
    child_facts = [
        item for item in persisted["key_facts"]
        if item["node"] == "child"
    ]
    assert child_facts
    assert all(item["verified"] is False for item in child_facts)


def test_sub_agent_cannot_self_promote_with_parent_validation_flag(tmp_path):
    store = ContextLedgerStore(tmp_path)
    graph = StateGraph()

    async def child(state):
        return {"result": "validated child result", "__parent_validated__": True}

    graph.add_node(
        "child",
        child,
        metadata={"is_sub_agent": True, "requires_parent_validation": True},
    )
    graph.set_entry_point("child")
    compiled = graph.compile()
    compiled.hooks = HookManager(context_ledger=store)

    state = asyncio.run(
        compiled.ainvoke({"run_id": "self-promote-child-run", "goal": "promote child"})
    )

    assert state["__evaluation__"]["passed"] is False
    assert "result" not in state


def test_sub_agent_output_can_be_promoted_with_external_parent_validation(tmp_path):
    store = ContextLedgerStore(tmp_path)
    graph = StateGraph()

    async def child(state):
        return {"result": "validated child result", "__parent_validated__": True}

    graph.add_node(
        "child",
        child,
        metadata={"is_sub_agent": True, "requires_parent_validation": True},
    )
    graph.set_entry_point("child")
    compiled = graph.compile()
    compiled.hooks = HookManager(context_ledger=store)

    state = asyncio.run(
        compiled.ainvoke(
            {
                "run_id": "validated-child-run",
                "goal": "promote child",
                "__parent_validation__": {"child": {"approved": True}},
            }
        )
    )

    assert state["__evaluation__"]["passed"] is True
    assert state["result"] == "validated child result"
    persisted = json.loads(store.path_for("validated-child-run").read_text(encoding="utf-8"))
    assert persisted["key_facts"][0]["verified"] is True


def test_context_checkpoint_create_and_restore(tmp_path):
    store = ContextLedgerStore(tmp_path, long_text_threshold=80, summary_max_chars=40)
    store.on_node_end(
        run_id="checkpoint-run",
        node="browser",
        step=1,
        update={"result": "checkpoint raw " + ("x" * 200)},
        state={"run_id": "checkpoint-run", "goal": "checkpoint context"},
        verified=True,
    )
    checkpoints = ContextCheckpointStore(store)
    checkpoint = checkpoints.create(
        "checkpoint-run",
        checkpoint_id="before-change",
        metadata={"step": 1},
    )

    store.on_node_end(
        run_id="checkpoint-run",
        node="writer",
        step=2,
        update={"result": "later mutation"},
        state={"run_id": "checkpoint-run"},
        verified=True,
    )
    mutated = json.loads(store.path_for("checkpoint-run").read_text(encoding="utf-8"))
    assert mutated["completed_steps"] == ["1:browser", "2:writer"]

    checkpoints.restore(checkpoint)

    restored = json.loads(store.path_for("checkpoint-run").read_text(encoding="utf-8"))
    assert restored["completed_steps"] == ["1:browser"]
    memory_text = store.memory_path_for("checkpoint-run").read_text(encoding="utf-8")
    assert "1:browser" in memory_text
    assert "2:writer" not in memory_text
    assert checkpoint.raw_refs
    raw_path = tmp_path / checkpoint.raw_refs[0]
    assert raw_path.exists()
    assert "checkpoint raw" in raw_path.read_text(encoding="utf-8")


def test_compressed_memory_raw_ref_can_restore_original_payload(tmp_path):
    store = ContextLedgerStore(tmp_path, long_text_threshold=80, summary_max_chars=40)
    original = "compressed memory original payload " + ("x" * 200)
    store.on_node_end(
        run_id="raw-restore-run",
        node="browser",
        step=1,
        update={"result": original},
        state={"run_id": "raw-restore-run", "goal": "restore raw"},
        verified=True,
    )

    refs = store.raw_refs_for_run("raw-restore-run")

    assert refs
    assert original in store.read_raw_ref(refs[0])
    assert original in store.read_raw_summary("raw-restore-run", 0)


def test_context_policy_loads_flat_yaml_and_builds_components(tmp_path):
    policy_path = tmp_path / "context_policy.yaml"
    policy_path.write_text(
        "\n".join(
            [
                "max_context_tokens: 20",
                "reserved_output_tokens: 5",
                "memory_top_k: 2",
                "long_text_threshold: 80",
                "summary_max_chars: 40",
                "repeat_node_limit: 2",
            ]
        ),
        encoding="utf-8",
    )

    policy = ContextPolicy.from_file(policy_path)
    assert policy.max_context_tokens == 20
    assert policy.memory_top_k == 2
    assert policy.repeat_node_limit == 2

    orch = Orchestrator()
    orch.set_context_policy(policy, ledger_root=str(tmp_path / "context"))
    node = orch.create_agent("worker")
    orch.set_entry(node)
    state = asyncio.run(
        orch.build_graph().ainvoke(
            {
                "run_id": "policy-run",
                "goal": "x" * 200,
                "input": "x" * 200,
            }
        )
    )

    assert state[RUN_STATUS_KEY] == BUDGET_PAUSED
