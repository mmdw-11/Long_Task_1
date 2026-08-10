"""实验评测工具包。

这个包把科研实验需要的“数据读取、样本标准化、模块评测、结果落盘”封装成可复用
代码。CLI 示例放在 examples/experiments 下，后端或测试也可以直接 import 这里的函数。
"""

from .datasets import (
    build_workflow_dataset,
    load_memory_dataset,
    load_skill_dataset,
    save_jsonl,
)
from .memory import MemoryExperimentConfig, run_memory_experiment
from .reports import ExperimentReport, save_report
from .skills import SkillExperimentConfig, run_skill_experiment
from .workflow import WorkflowExperimentConfig, run_workflow_experiment

__all__ = [
    "ExperimentReport",
    "MemoryExperimentConfig",
    "SkillExperimentConfig",
    "WorkflowExperimentConfig",
    "build_workflow_dataset",
    "load_memory_dataset",
    "load_skill_dataset",
    "run_memory_experiment",
    "run_skill_experiment",
    "run_workflow_experiment",
    "save_jsonl",
    "save_report",
]
