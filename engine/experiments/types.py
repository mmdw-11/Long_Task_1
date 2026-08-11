"""实验样本与结果的数据结构。

这里定义的是项目内部统一格式。公开数据集字段各不相同，先在 datasets.py 中转成这些
结构，再交给 memory / skills / workflow runner，避免每个实验重复写字段兼容逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryExample:
    """一条长期记忆问答样本。"""

    id: str
    question: str
    answer: str
    memories: List[str]
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillExample:
    """一条技能演化样本。

    task 是当前任务；trajectory 是过去成功经验；expected_steps 是答案或技能中应出现的
    关键步骤；task_type 用来对齐本项目 SkillRetriever 的 task_type 过滤。
    """

    id: str
    task: str
    trajectory: str
    expected_steps: List[str]
    task_type: str = "general"
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExample:
    """一条小型编排工作流样本。"""

    id: str
    pattern: str
    prompt: str
    expected_agents: List[str]
    branch: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentRow:
    """单样本评测结果。"""

    id: str
    passed: bool
    score: float
    prediction: str = ""
    expected: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "passed": self.passed,
            "score": self.score,
            "prediction": self.prediction,
            "expected": self.expected,
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRow":
        """从 rows.jsonl 里的字典恢复实验结果，供断点续跑合并使用。"""
        return cls(
            id=str(data.get("id") or ""),
            passed=bool(data.get("passed", False)),
            score=float(data.get("score") or 0.0),
            prediction=str(data.get("prediction") or ""),
            expected=str(data.get("expected") or ""),
            metrics=dict(data.get("metrics") or {}),
            metadata=dict(data.get("metadata") or {}),
        )
