"""Offline acceptance tests for the stateful τ experiment integration."""

import json

import pytest

from engine.experiments.tau_data import (
    deterministic_sample, load_tau_tasks, prepare_tau_dataset, split_train_ids,
)
from engine.experiments.tau_runtime import (
    paired_trial_seed, rank_tau_skills, select_tau_skills, simulation_key, summarize_tau_rows,
)
from engine.experiments.tau_analysis import initial_intent_families, task_profile
from engine.experiments.tau_skills import (
    TauSkill, TauSkillValidationError, group_training_tasks, parse_generated_skill, validate_tau_skill,
)


def _skill(**overrides):
    payload = {
        "id": "skill-retail", "name": "Returns", "domain": "retail",
        "applicable_when": ["a supported return"],
        "not_applicable_when": ["outside retail"],
        "identity_verification": ["verify identity"],
        "read_steps": [{"tool": "get_order_details", "depends_on": ["order_id"]}],
        "write_steps": [{"tool": "return_delivered_order_items", "depends_on": ["order", "items"], "requires_confirmation": True}],
        "branches": [], "refusal_conditions": ["not eligible"],
        "handoff_conditions": ["policy requires it"],
        "recovery_paths": [{"on": "tool error", "then": "re-read"}],
        "final_state_checks": ["verify return"], "communicate_info": ["refund timing"],
        "source_task_ids": ["retail:train-1"],
    }
    payload.update(overrides)
    return TauSkill(**payload)


def test_tau_split_and_sampling_are_stable_and_disjoint():
    first = split_train_ids([str(i) for i in range(20)], seed=42)
    second = split_train_ids(reversed([str(i) for i in range(20)]), seed=42)
    assert first == second
    assert set(first["skill_train"]).isdisjoint(first["skill_validation"])
    assert deterministic_sample([str(i) for i in range(20)], 5, 42, "retail") == deterministic_sample(
        reversed([str(i) for i in range(20)]), 5, 42, "retail"
    )


def test_tau_converter_maps_official_schema_and_detects_no_leakage(tmp_path):
    checkout = tmp_path / "tau"
    for domain in ("retail", "airline"):
        root = checkout / "data" / "tau2" / "domains" / domain
        root.mkdir(parents=True)
        tasks = [
            {"id": str(i), "user_scenario": {"instructions": f"task {domain} {i}"},
             "evaluation_criteria": {"actions": [{"name": "get_record", "arguments": {}}],
                                     "reward_basis": ["DB"], "communicate_info": []}}
            for i in range(6)
        ]
        (root / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
        (root / "split_tasks.json").write_text(json.dumps({"train": ["0", "1", "2", "3"], "test": ["4", "5"]}), encoding="utf-8")
        (root / "policy.md").write_text("policy", encoding="utf-8")
        (root / "db.json").write_text("{}", encoding="utf-8")
    result = prepare_tau_dataset(
        checkout, tmp_path / "out", test_count_per_domain=2,
        tool_schema_loader=lambda _: [{"type": "function", "function": {"name": "get_record", "parameters": {}}}],
    )
    rows = load_tau_tasks(result["tasks"])
    assert result["leakage"] == {"id_overlap": False, "scenario_overlap": False}
    assert len([row for row in rows if row.split == "test"]) == 4
    assert all(row.tool_schemas for row in rows)
    assert all(row.agent_view().keys().isdisjoint({"reference_actions", "reward_basis", "env_assertions"}) for row in rows)


def test_tau_skill_rejects_unknown_tool_missing_confirmation_and_test_leak():
    skill = _skill(
        write_steps=[{"tool": "invented_write", "depends_on": ["x"], "requires_confirmation": False}],
        source_task_ids=["retail:test-1"],
    )
    with pytest.raises(TauSkillValidationError) as exc:
        validate_tau_skill(skill, available_tools={"get_order_details"}, test_ids={"retail:test-1"})
    assert "unknown tools" in str(exc.value)
    assert "confirmation" in str(exc.value)
    assert skill.status == "rejected"


def test_tau_skill_failed_replay_cannot_publish():
    skill = _skill()
    with pytest.raises(TauSkillValidationError):
        validate_tau_skill(
            skill,
            available_tools={"get_order_details", "return_delivered_order_items"},
            replay=lambda _: {"reward": 0.0, "policy_violations": 1, "baseline_policy_violations": 0},
            baseline_reward=1.0,
        )
    assert skill.status == "rejected"


def test_tau_generated_json_parser_is_strict():
    text = "```json\n" + json.dumps(_skill().to_dict()) + "\n```"
    assert parse_generated_skill(text, domain="retail").id == "skill-retail"
    with pytest.raises((TauSkillValidationError, json.JSONDecodeError)):
        parse_generated_skill('{"id":"partial"}', domain="retail")


def test_tau_training_groups_are_action_families_only():
    from engine.experiments.types import TauToolTask
    common = dict(domain="retail", user_scenario={}, domain_policy="p", tool_schemas=[], initial_state_ref="h")
    tasks = [
        TauToolTask(id="a", split="skill_train", reference_actions=[{"name": "get_order"}, {"name": "cancel_order"}], **common),
        TauToolTask(id="b", split="skill_train", reference_actions=[{"name": "get_order"}], **common),
    ]
    assert set(group_training_tasks(tasks)) == {"cancel_order", "read_or_refuse"}


def test_tau_paired_keys_and_summary_require_complete_matrix():
    assert paired_trial_seed(42, 1, "retail:1") == paired_trial_seed(42, 1, "retail:1")
    key = simulation_key("retail", "no_skill", 0, "1")
    row = {
        "key": key, "domain": "retail", "method": "no_skill", "trial": 0,
        "task_id": "1", "reward": 1.0, "passed": True, "error": None,
        "elapsed_seconds": 1.0, "tool_calls": 2, "total_cost": 0.0,
        "cost_available": False,
    }
    complete = summarize_tau_rows([row], expected_keys={key})
    incomplete = summarize_tau_rows([row], expected_keys={key, "missing"})
    assert complete["complete"]
    assert complete["methods"]["no_skill"]["total_cost"] is None
    assert complete["methods"]["no_skill"]["cost_per_success"] is None
    assert not incomplete["complete"] and incomplete["missing_keys"] == ["missing"]


def test_tau_retrieval_routes_action_families_and_ignores_negated_noise():
    exchange = _skill(id="exchange", name="Exchange", metadata={"action_family": "exchange_delivered_order_items"})
    modify = _skill(id="modify", name="Modify", metadata={"action_family": "modify_pending_order_items"})
    unrelated = _skill(id="airline", name="Airline", domain="airline", metadata={"action_family": "cancel_reservation"})
    query = "Exchange the delivered laptop. I do not want to cancel, return, or book anything else."
    ranked = rank_tau_skills([modify, unrelated, exchange], query, "retail")
    assert ranked[0][0].id == "exchange"
    assert [item.id for item, _ in select_tau_skills([modify, unrelated, exchange], query, "retail")][0] == "exchange"
    assert not select_tau_skills([modify, unrelated, exchange], "Tell me a joke about space.", "retail")
    first = select_tau_skills([modify, exchange], "Change an item in my pending order.", "retail")
    second = select_tau_skills([modify, exchange], "I also need to exchange a delivered laptop.", "retail")
    assert {item.id for item, _ in first + second} == {"modify", "exchange"}


def test_tau_task_profile_labels_stateful_difficulty_without_agent_leakage():
    from engine.experiments.types import TauToolTask
    task = TauToolTask(
        id="retail:test", domain="retail", split="test",
        user_scenario={"instructions": {"reason_for_call": "Modify two items in my pending order.", "task_instructions": "First check it.\nConfirm fees.\nThen update it."}},
        domain_policy="policy", tool_schemas=[], initial_state_ref="hash",
        reference_actions=[
            {"name": "get_order_details"},
            {"name": "modify_pending_order_items"},
            {"name": "modify_pending_order_items"},
        ],
        metadata={"nl_assertions": ["communicate the new total"]},
    )
    profile = task_profile(task)
    assert profile["requires_confirmation"]
    assert profile["write_actions"] == 2
    assert profile["communication_requirements"] == 1
    assert initial_intent_families(task) == {"modify_pending_order_items"}
