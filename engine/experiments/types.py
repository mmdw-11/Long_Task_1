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
    # Gold evidence is optional because some public releases only provide an
    # answer. Retrieval metrics are reported as answer-text fallback metrics
    # when this list is empty; QA metrics remain the primary outcome.
    evidence: List[str] = field(default_factory=list)
    trajectory_id: str = ""
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
class TauToolTask:
    """A normalized, stateful task from the official tau2-bench repository.

    Gold fields are retained in the frozen evaluation artifact for replay and
    scoring, but runners must use :meth:`agent_view` when constructing prompts.
    This keeps reference actions and target-state assertions out of the agent
    and generated-skill context.
    """

    id: str
    domain: str
    split: str
    user_scenario: Dict[str, Any]
    domain_policy: str
    tool_schemas: List[Dict[str, Any]]
    initial_state_ref: str
    reference_actions: List[Dict[str, Any]] = field(default_factory=list)
    reward_basis: List[str] = field(default_factory=list)
    communicate_info: List[str] = field(default_factory=list)
    env_assertions: List[Dict[str, Any]] = field(default_factory=list)
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    read_actions: List[str] = field(default_factory=list)
    write_actions: List[str] = field(default_factory=list)
    confirmation_boundaries: List[str] = field(default_factory=list)
    source_commit: str = ""
    source_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def agent_view(self) -> Dict[str, Any]:
        """Return only information that is legal to expose at test time."""
        return {
            "id": self.id,
            "domain": self.domain,
            "user_scenario": self.user_scenario,
            "domain_policy": self.domain_policy,
            "tool_schemas": self.tool_schemas,
            "initial_state_ref": self.initial_state_ref,
        }


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
