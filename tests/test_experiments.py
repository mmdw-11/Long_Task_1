"""实验代码的基础可运行性测试。"""

import pytest

from engine.experiments import (
    MemoryExperimentConfig,
    LongTaskExperimentConfig,
    SkillExperimentConfig,
    WorkflowExperimentConfig,
    build_workflow_dataset,
    build_long_task_dataset,
    run_memory_experiment,
    run_long_task_experiment,
    run_routing_experiment,
    run_skill_experiment,
    run_workflow_experiment,
)
from engine.experiments.types import MemoryExample, SkillExample
from engine import PseudoCascadeTeacher, build_route_dataset, default_training_texts


def test_memory_experiment_engine_backend(tmp_path):
    examples = [
        MemoryExample(
            id="m1",
            question="小明喜欢什么水果？",
            answer="苹果",
            memories=["小明喜欢苹果，也喜欢梨。", "小红喜欢香蕉。"],
            source="unit",
        )
    ]

    report = run_memory_experiment(
        examples,
        MemoryExperimentConfig(output_root=str(tmp_path), top_k=2, use_llm_judge=False),
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
        MemoryExperimentConfig(backend="no_memory", output_root=str(tmp_path)),
    )
    full_context = run_memory_experiment(
        examples,
        MemoryExperimentConfig(backend="full_context", output_root=str(tmp_path)),
    )

    assert no_memory.summary()["pass_rate"] == 0.0
    assert full_context.summary()["pass_rate"] == 1.0
    assert full_context.summary()["avg_tokens"] > no_memory.summary()["avg_tokens"]


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


def test_routing_experiment_includes_four_controls():
    dataset = build_route_dataset(default_training_texts(), teacher=PseudoCascadeTeacher())
    reports = run_routing_experiment(dataset)

    assert set(reports) == {"all_device", "all_cloud", "heuristic", "learned"}
    assert all(report.summary()["total"] > 0 for report in reports.values())
    assert reports["all_cloud"].summary()["avg_cloud_call"] == 1.0
    assert reports["all_device"].summary()["avg_cloud_call"] == 0.0
