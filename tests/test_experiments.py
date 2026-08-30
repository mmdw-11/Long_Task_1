"""实验代码的基础可运行性测试。"""

import pytest

from engine.experiments import (
    MemoryExperimentConfig,
    LongTaskExperimentConfig,
    SkillExperimentConfig,
    WorkflowExperimentConfig,
    build_long_task_dataset_from_memory,
    build_workflow_dataset,
    build_long_task_dataset,
    run_memory_experiment,
    run_long_task_experiment,
    run_skill_experiment,
    run_workflow_experiment,
)
from engine.experiments.types import MemoryExample, SkillExample
from engine.experiments.datasets import load_memory_dataset
from engine.experiments.long_task import memory_contains_expected


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

    assert no_memory.summary()["pass_rate"] == 0.0
    assert full_context.summary()["pass_rate"] == 1.0
    assert full_context.summary()["avg_context_tokens"] > no_memory.summary()["avg_context_tokens"]


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


def test_skill_experiment_generates_retrievable_skill(tmp_path):
    examples = [
        SkillExample(
            id="s1",
            task="安排会议并检查冲突",
            trajectory="先读取日历，再检查冲突，最后创建提醒。",
            expected_steps=["读取日历", "检查冲突", "创建提醒"],
            task_type="calendar",
            source="unit",
        )
    ]

    report = run_skill_experiment(examples, SkillExperimentConfig(output_root=str(tmp_path)))

    assert report.summary()["pass_rate"] == 1.0


@pytest.mark.asyncio
async def test_workflow_experiment_runs_all_patterns():
    examples = build_workflow_dataset(size=5, seed=1)
    report = await run_workflow_experiment(examples, WorkflowExperimentConfig())

    assert report.summary()["total"] == 5
    assert report.summary()["pass_rate"] == 1.0
