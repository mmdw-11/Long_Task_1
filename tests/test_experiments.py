"""实验代码的基础可运行性测试。"""

import pytest

from engine.experiments import (
    MemoryExperimentConfig,
    LongTaskExperimentConfig,
    SkillExperimentConfig,
    WorkflowExperimentConfig,
    build_long_task_dataset_from_memory,
    build_skill_reuse_dataset,
    build_workflow_dataset,
    build_long_task_dataset,
    run_memory_experiment,
    run_long_task_experiment,
    run_skill_experiment,
    run_workflow_experiment,
)
from engine.experiments.types import MemoryExample, SkillExample
from engine.experiments.datasets import load_memory_dataset, memory_examples_to_rows, sample_memory_examples
from engine.experiments.long_task import memory_contains_expected
from engine.experiments.memory import (
    _disable_mem0_thinking,
    _engine_memory_context,
    _engine_retrieved_text,
    _mem0_source_text,
    _qa_judge_prompt,
    _select_evidence_chunks,
    _yes_verdict,
)
from engine.modules.memory.judge import OpenAIMemoryJudge


@pytest.mark.parametrize(
    ("expected", "memory"),
    [
        ("Photography", "Calvin recently got into photography."),
        ("two", "Calvin owns 2 Ferraris."),
        ("ABC-123", "历史确认码是 abc 123。"),
        ("苹果", "小明最喜欢的水果是苹果。"),
    ],
)
def test_memory_match_normalizes_harmless_surface_differences(expected, memory):
    assert memory_contains_expected(expected, memory)


def test_memory_match_does_not_match_inside_unrelated_word():
    assert not memory_contains_expected("two", "The network is working.")


def test_long_task_dataset_selects_fact_on_token_boundaries():
    examples = [
        MemoryExample(
            id="q1",
            question="What is the adopted pup's name?",
            answer="Ned",
            memories=[
                "John signed up for a programming class.",
                "James adopted a pup named Ned.",
            ],
            source="locomo",
        )
    ]

    tasks = build_long_task_dataset_from_memory(examples, count=1, seed=42)

    assert tasks[0].memory_fact == "James adopted a pup named Ned."


def test_locomo_nested_qa_ids_use_sample_id(tmp_path):
    dataset = tmp_path / "locomo.json"
    dataset.write_text(
        '[{"sample_id":"conv-a","conversation":"A1 context",'
        '"qa":[{"question":"Q1","answer":"A1"}]},'
        '{"sample_id":"conv-b","conversation":"A2 context",'
        '"qa":[{"question":"Q2","answer":"A2"}]}]',
        encoding="utf-8",
    )

    examples = load_memory_dataset(dataset, source="locomo")

    assert [item.id for item in examples] == ["conv-a-q0", "conv-b-q0"]
    assert len({item.id for item in examples}) == 2


def test_stratified_memory_sampling_round_robins_question_types():
    examples = [
        MemoryExample(
            id=f"{question_type}-{index}",
            question="Q",
            answer="A",
            memories=["A"],
            metadata={"question_type": question_type},
        )
        for question_type in ("temporal", "update")
        for index in range(3)
    ]

    selected = sample_memory_examples(examples, count=4, seed=42, stratified=True)

    assert {item.metadata["question_type"] for item in selected} == {"temporal", "update"}


def test_memory_experiment_engine_backend(tmp_path):
    examples = [
        MemoryExample(
            id="m1",
            question="小明喜欢什么水果？",
            answer="苹果",
            memories=["小明最喜欢的水果是 苹果。", "小红喜欢香蕉。"],
            source="unit",
        )
    ]

    report = run_memory_experiment(
        examples,
        MemoryExperimentConfig(output_root=str(tmp_path), top_k=2, use_llm_judge=False, qa_solver="extractive"),
    )

    assert report.summary()["pass_rate"] == 1.0


def test_evidence_selection_respects_budget_and_keeps_multiple_sessions():
    sessions = [
        "irrelevant opening. The first evidence says alpha is the correct value. " * 30,
        "another history. The second evidence confirms alpha with a timestamp. " * 30,
    ]
    selected, stats = _select_evidence_chunks(
        "What is the alpha value?",
        sessions,
        token_budget=300,
        chunk_chars=300,
        overlap_chars=40,
        max_chunks_per_session=1,
    )

    assert stats["candidate_sessions"] == 2
    assert stats["selected_tokens"] <= 300
    assert selected

def test_memory_control_baselines_report_tokens(tmp_path):
    examples = [
        MemoryExample(
            id="baseline",
            question="确认码是什么？",
            answer="ABC-123",
            memories=["确认码是 ABC-123。", "无关记录"],
            source="unit",
        )
    ]
    no_memory = run_memory_experiment(
        examples,
        MemoryExperimentConfig(backend="no_memory", output_root=str(tmp_path), qa_solver="extractive"),
    )
    full_context = run_memory_experiment(
        examples,
        MemoryExperimentConfig(backend="full_context", output_root=str(tmp_path), qa_solver="extractive"),
    )
    budgeted = run_memory_experiment(
        examples,
        MemoryExperimentConfig(
            backend="full_context_budgeted",
            output_root=str(tmp_path),
            context_token_budget=2,
            qa_solver="extractive",
        ),
    )

    assert no_memory.summary()["pass_rate"] == 0.0
    assert full_context.summary()["pass_rate"] == 1.0
    assert full_context.summary()["avg_context_tokens"] > no_memory.summary()["avg_context_tokens"]
    assert budgeted.summary()["avg_context_tokens"] <= 2


def test_longmemeval_judge_accepts_only_explicit_yes_verdict():
    assert _yes_verdict("yes")
    assert _yes_verdict("Yes.")
    assert not _yes_verdict("no")
    assert not _yes_verdict("The answer says yes, but it is incorrect.")


def test_longmemeval_judge_prompt_uses_task_specific_criterion():
    example = MemoryExample(
        id="temporal-question",
        question="How many days passed?",
        answer="six days",
        memories=["The event dates were recorded."],
        source="longmemeval",
        metadata={"question_type": "temporal-reasoning"},
    )

    prompt = _qa_judge_prompt(example, "It was 7 days.")

    assert "off-by-one" in prompt
    assert "six days" in prompt


def test_processed_memory_dataset_preserves_question_type_metadata(tmp_path):
    original = MemoryExample(
        id="q1",
        question="When did it happen?",
        answer="Tuesday",
        memories=["It happened on Tuesday."],
        source="longmemeval",
        metadata={"question_type": "temporal-reasoning"},
    )
    dataset = tmp_path / "processed.jsonl"
    import json
    dataset.write_text(
        json.dumps(memory_examples_to_rows([original])[0], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    loaded = load_memory_dataset(dataset, source="longmemeval")

    assert loaded[0].metadata["question_type"] == "temporal-reasoning"


def test_mem0_requests_disable_deepseek_thinking():
    calls = []

    class Completions:
        def create(self, *args, **kwargs):
            calls.append(kwargs)
            return "response"

    class Object:
        pass

    memory = Object()
    memory.llm = Object()
    memory.llm.client = Object()
    memory.llm.client.chat = Object()
    memory.llm.client.chat.completions = Completions()

    _disable_mem0_thinking(memory)
    result = memory.llm.client.chat.completions.create(
        model="deepseek-v4-flash",
        extra_body={"unrelated": "preserved", "thinking": {"type": "enabled"}},
    )

    assert result == "response"
    assert calls[0]["extra_body"] == {
        "unrelated": "preserved",
        "thinking": {"type": "disabled"},
    }


def test_mem0_retrieval_metrics_use_source_session_provenance():
    example = MemoryExample(
        id="q1",
        question="What was the fare?",
        answer="$6",
        memories=["irrelevant session", "The taxi cost $14 and the train cost $8."],
        evidence=["The taxi cost $14 and the train cost $8."],
        source="longmemeval",
    )
    extracted = {
        "memory": "User paid more for a taxi.",
        "metadata": {"index": 1},
    }

    assert _mem0_source_text(extracted, example) == example.evidence[0]


def test_engine_memory_examples_use_isolated_project_scopes():
    first = MemoryExample(
        id="q1",
        question="Q1",
        answer="A1",
        memories=["history one"],
        source="longmemeval",
    )
    second = MemoryExample(
        id="q2",
        question="Q2",
        answer="A2",
        memories=["history two"],
        source="longmemeval",
    )

    first_context = _engine_memory_context(first)
    second_context = _engine_memory_context(second)

    assert first_context.project_id != second_context.project_id
    assert first_context.project_id == "longmemeval-q1"
    assert second_context.project_id == "longmemeval-q2"


def test_engine_retrieval_hydrates_archived_raw_text():
    class Item:
        raw_ref = "project_archive/example.md"
        content = "truncated summary"

    class Store:
        def expand(self, item):
            assert item.raw_ref
            return "full archived session with the answer"

    assert _engine_retrieved_text(Store(), Item()) == "full archived session with the answer"


def test_memory_update_judge_disables_thinking_and_tracks_calls():
    calls = []

    class Message:
        content = '{"actions":[{"id":"old","action":"update"}]}'

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return Response()

    class Object:
        pass

    judge = OpenAIMemoryJudge.__new__(OpenAIMemoryJudge)
    judge._model = "deepseek-v4-flash"
    judge._temperature = 0.0
    judge.call_count = 0
    judge.parse_error_count = 0
    judge.thinking_disabled = True
    judge._client = Object()
    judge._client.chat = Object()
    judge._client.chat.completions = Completions()

    actions = judge.judge("new", [{"id": "old", "content": "old"}])

    assert actions == [{"id": "old", "action": "update"}]
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert calls[0]["max_tokens"] == 500
    assert judge.call_count == 1
    assert judge.parse_error_count == 0


def test_long_task_joint_experiment_exercises_all_controls(tmp_path):
    examples = build_long_task_dataset(size=2, seed=1)
    full = run_long_task_experiment(
        examples,
        LongTaskExperimentConfig(method="ours_full", output_root=str(tmp_path)),
    )
    plain = run_long_task_experiment(
        examples,
        LongTaskExperimentConfig(method="plain", output_root=str(tmp_path)),
    )

    assert full.summary()["pass_rate"] == 1.0
    assert full.summary()["avg_memory_use_accuracy"] == 1.0
    assert full.summary()["avg_pause_correctness"] == 1.0
    assert full.summary()["avg_recovery_success"] == 1.0
    assert plain.summary()["pass_rate"] == 0.0
    assert plain.summary()["avg_goal_retention"] == 1.0
    assert plain.summary()["avg_constraint_compliance"] == 1.0
    assert plain.summary()["avg_pause_correctness"] == 1.0
    assert all(
        row.metadata["evaluation_policy"]
        == "shared_prompt_shared_disturbances_layered_readiness_v5"
        for row in full.rows + plain.rows
    )


def test_long_task_reports_basic_readiness_separately_from_disturbance_handling(tmp_path):
    examples = build_long_task_dataset(size=1, seed=7)
    memory_only = run_long_task_experiment(
        examples,
        LongTaskExperimentConfig(method="memory_only", output_root=str(tmp_path)),
    )
    row = memory_only.rows[0]

    assert row.metrics["basic_task_readiness"] == 1
    assert row.metrics["disturbance_handling_score"] == pytest.approx(1 / 3)
    assert row.metrics["final_task_success"] == 0
    assert row.metrics["recovery_success"] == 0


def test_skill_experiment_generates_retrievable_skill(tmp_path):
    examples = [
        SkillExample(
            id=f"s{index}",
            task="安排会议并检查冲突",
            trajectory="先读取日历，再检查冲突，确认时区，最后创建提醒。",
            expected_steps=["读取日历", "检查冲突", "确认时区", "创建提醒"],
            task_type="calendar",
            source="unit",
            metadata={"split": "train" if index == 0 else "test", "critical_steps": ["确认时区"]},
        )
        for index in range(4)
    ]

    report = run_skill_experiment(examples, SkillExperimentConfig(output_root=str(tmp_path)))

    assert report.summary()["pass_rate"] == 1.0
    assert report.summary()["total"] == 3
    assert report.summary()["avg_automatic_skill_generation_success"] == 1.0


def test_no_skill_uses_shared_base_plan_without_fake_retrieval(tmp_path):
    examples = build_skill_reuse_dataset(size=30, seed=42)

    report = run_skill_experiment(
        examples,
        SkillExperimentConfig(
            output_root=str(tmp_path / "no-skill"),
            method="no_skill",
        ),
    )
    summary = report.summary()

    assert summary["total"] == 21
    assert summary["avg_task_success"] == 0.0
    assert summary["avg_basic_task_readiness"] == 1.0
    assert summary["avg_base_plan_coverage"] == 0.5
    assert summary["avg_step_coverage"] == 0.5
    assert summary["avg_critical_step_coverage"] == 0.0
    assert summary["avg_retrieval_hit"] == 0.0
    assert summary["avg_skill_coverage_gain"] == 0.0
    assert all(
        row.metadata["plan_policy"] == "shared_task_native_base_plus_optional_skill"
        for row in report.rows
    )


@pytest.mark.asyncio
async def test_workflow_experiment_runs_all_patterns():
    examples = build_workflow_dataset(size=5, seed=1)
    report = await run_workflow_experiment(examples, WorkflowExperimentConfig())

    assert report.summary()["total"] == 5
    assert report.summary()["pass_rate"] == 1.0
