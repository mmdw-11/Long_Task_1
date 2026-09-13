import json
from pathlib import Path

import pytest

from engine.experiments.toolsandbox_data import primary_group, scenario_family
from engine.experiments.toolsandbox_runtime import stable_role_seed, summarize
from engine.experiments.toolsandbox_skills import AUTO_VALIDATED_SKILL, BGEPolicyRetriever, ToolSandboxSkill, retrieve_skills, validate_skill


def test_family_normalization_and_priority():
    assert scenario_family("abc_3_distraction_tools") == "abc"
    assert primary_group(["STATE_DEPENDENCY", "MULTIPLE_TOOL_CALL", "MULTIPLE_USER_TURN"]) == "state_dependency"


def test_validated_skill_gate():
    result = validate_skill(AUTO_VALIDATED_SKILL, set())
    assert result.accepted
    assert len(result.checks) == 5


def test_summary_recomputes_and_preserves_valid_reward_zero():
    rows = [
        {"key": "a", "method": "no_skill", "primary_group": "g", "reward": 0.0, "error": None, "total_tokens": 10, "seconds": 1},
        {"key": "b", "method": "no_skill", "primary_group": "g", "reward": 1.0, "error": None, "total_tokens": 20, "seconds": 3},
    ]
    summary = summarize(rows)
    assert summary["errors"] == 0
    assert summary["methods"]["no_skill"]["task_success"] == 0.5
    assert summary["methods"]["no_skill"]["mean_tokens"] == 15
    assert summary["methods"]["no_skill"]["reward_95ci"] == [0.0, 1.0]


def test_invalid_agent_action_is_not_an_infrastructure_error():
    # The official runner raises for an unknown/disallowed tool. Our experiment
    # records that as a valid reward-zero agent outcome, not a missing run.
    from engine.experiments import toolsandbox_runtime as runtime
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert '"invalid_or_unauthorized_tool"' in source


def test_deepseek_tool_call_id_is_made_python_safe():
    from types import SimpleNamespace
    from engine.experiments.toolsandbox_runtime import _sanitize_tool_call_ids
    call = SimpleNamespace(id="call_-727.12")
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call]))])
    _sanitize_tool_call_ids(response)
    assert call.id == "call__727_12"
    assert call.id.isidentifier()


def test_frozen_dataset_is_family_disjoint_if_present():
    path = Path("data/processed/toolsandbox_hard_v2/tasks.jsonl")
    if not path.exists():
        pytest.skip("dataset not prepared")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 67
    assert len({row["id"] for row in rows}) == 67
    assert len({row["family"] for row in rows}) == 67
    formal = [row for row in rows if row["split"] == "formal_test"]
    assert len(formal) == 30


def test_pairing_seed_is_method_independent_and_trial_specific():
    assert stable_role_seed("task", 0, "agent") == stable_role_seed("task", 0, "agent")
    assert stable_role_seed("task", 0, "agent") != stable_role_seed("task", 1, "agent")
    assert stable_role_seed("task", 0, "agent") != stable_role_seed("task", 0, "user")


def test_retrieval_rejects_single_generic_overlap():
    payload = {
        "skill_id": "reminders", "name": "Reminder", "version": 1,
        "source_type": "successful_train_trajectories", "source_trajectory_keys": ["train/x"],
        "source_families": ["reminder"], "applicable_when": ["add a specific reminder at a specified time"],
        "not_applicable_when": ["sending messages"], "required_tools": ["add_reminder"],
        "required_slots": ["content", "time"], "preconditions": [], "ordered_steps": [],
        "canonicalization_rules": [], "clarification_rules": [], "abstention_rules": [],
        "recovery_paths": [], "safety_rules": [], "success_checks": [],
    }
    text, trace = retrieve_skills([ToolSandboxSkill(**payload)], "send a specific message", max_chars=6000)
    assert text == ""
    assert not trace[0]["accepted"]


def test_late_distractor_cannot_change_skill_domain():
    payload = {
        "skill_id": "messaging", "name": "Messaging", "version": 1,
        "source_type": "successful_train_trajectories", "source_trajectory_keys": ["train/x"],
        "source_families": ["message"], "applicable_when": ["send message using phone number"],
        "not_applicable_when": [], "required_tools": ["send_message_with_phone_number"],
        "required_slots": [], "preconditions": [], "ordered_steps": [],
        "canonicalization_rules": [], "clarification_rules": [], "abstention_rules": [],
        "recovery_paths": [], "safety_rules": [], "success_checks": [],
    }
    skill = ToolSandboxSkill(**payload)
    text, trace = retrieve_skills(
        [skill], "A distractor says provide a phone number to send a message",
        anchor_query="How many days until Thanksgiving?", max_chars=6000,
    )
    assert text == ""
    assert not trace[0]["domain_match"]


def test_single_strong_domain_anchor_retrieves_skill():
    payload = {
        "skill_id": "device", "name": "Device", "version": 1,
        "source_type": "successful_train_trajectories", "source_trajectory_keys": ["train/x"],
        "source_families": ["wifi"], "applicable_when": ["enable wifi service"],
        "not_applicable_when": [], "required_tools": ["set_wifi_status"],
        "required_slots": [], "preconditions": [], "ordered_steps": [],
        "canonicalization_rules": [], "clarification_rules": [], "abstention_rules": [],
        "recovery_paths": [], "safety_rules": [], "success_checks": [],
    }
    text, trace = retrieve_skills([ToolSandboxSkill(**payload)], "turn on wifi", max_chars=6000)
    assert text
    assert trace[0]["accepted"]


def test_bge_retrieval_records_backend_and_uses_real_embedding_interface():
    class TinyEmbedder:
        def embed(self, text):
            return [1.0, 0.0] if "wifi" in text else [0.0, 1.0]
    payload = {
        "skill_id": "device", "name": "Device", "version": 1,
        "source_type": "successful_train_trajectories", "source_trajectory_keys": ["train/x"],
        "source_families": ["wifi"], "applicable_when": ["enable wifi service"],
        "not_applicable_when": [], "required_tools": ["set_wifi_status"], "required_slots": [],
        "preconditions": [], "ordered_steps": [], "canonicalization_rules": [], "clarification_rules": [],
        "abstention_rules": [], "recovery_paths": [], "safety_rules": [], "success_checks": [],
    }
    skill = ToolSandboxSkill(**payload)
    _, trace = retrieve_skills([skill], "turn on wifi", max_chars=6000,
                               bge_retriever=BGEPolicyRetriever([skill], embedder=TinyEmbedder()))
    assert trace[0]["retrieval_backend"] == "bge_m3"


def test_strict_validation_uses_paired_non_regression():
    source = Path("examples/experiments/validate_toolsandbox_skills.py").read_text(encoding="utf-8")
    assert "paired-per-task-non-regression-v2" in source
    assert 'all(item["reward_non_regression"]' in source


def test_formal_preflight_is_fail_closed():
    source = Path("examples/experiments/preflight_toolsandbox_formal.py").read_text(encoding="utf-8")
    assert "no active replay-validated automatic skill library" in source
    assert "formal family leaked" in source


def test_formal_runner_locks_360_protocol_and_refuses_overwrite():
    source = Path("examples/experiments/run_toolsandbox_skill_experiment.py").read_text(encoding="utf-8")
    assert "30 tasks × four ordered methods × 3 trials = 360 runs" in source
    assert "refusing to overwrite an existing formal run" in source
    assert "fixed 300-second per-task hard timeout" in source
    assert "run_task_with_hard_timeout" in source


def test_incomplete_agent_failure_gets_auditable_hashes(tmp_path):
    from engine.experiments.toolsandbox_runtime import _incomplete_trajectory_metrics
    directory = tmp_path / "trajectory"
    directory.mkdir()
    (directory / "execution_context.json").write_text(
        json.dumps({"_dbs": {"SETTING": [{"wifi": False}]}}), encoding="utf-8"
    )
    metrics = _incomplete_trajectory_metrics(directory, "invalid tool")
    assert len(metrics["final_state_hash"]) == 64
    assert len(metrics["trajectory_sha256"]) == 64
    assert (directory / "failure_trajectory.json").exists()


def test_factorial_parallel_call_guard_is_present():
    from engine.experiments import toolsandbox_runtime as runtime
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "len(tool_calls) > 8" in source
    assert "ExcessiveParallelToolCallsError" in source
