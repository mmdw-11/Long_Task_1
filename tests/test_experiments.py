"""实验代码的基础可运行性测试。"""

import pytest

from engine.experiments import (
    MemoryExperimentConfig,
    SkillExperimentConfig,
    WorkflowExperimentConfig,
    build_workflow_dataset,
    run_memory_experiment,
    run_skill_experiment,
    run_workflow_experiment,
)
from engine.experiments.types import MemoryExample, SkillExample


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
        MemoryExperimentConfig(output_root=str(tmp_path), top_k=2),
    )

    assert report.summary()["pass_rate"] == 1.0


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
