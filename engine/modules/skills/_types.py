"""技能演化核心数据模型：统一在线检索、离线验证与发布状态。"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


class SkillStatus(str, enum.Enum):
    DRAFT = "draft"
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    PUBLISHED = "published"
    RETIRED = "retired"
    REJECTED = "rejected"


@dataclass
class SkillManifest:
    skill_id: str
    name: str
    version: str = "0.1.0"
    status: SkillStatus = SkillStatus.DRAFT
    description: str = ""
    task_types: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    applicable_nodes: List[str] = field(default_factory=list)
    incompatible_skills: List[str] = field(default_factory=list)
    source_runs: List[str] = field(default_factory=list)
    parent_version: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    approved_by: str = ""

    def __post_init__(self) -> None:
        if not self.skill_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for ch in self.skill_id):
            raise ValueError("skill_id must contain only letters, numbers, '_', '-' or '.'")
        if not self.name.strip():
            raise ValueError("skill name cannot be empty")
        if not _valid_version(self.version):
            raise ValueError(f"invalid semantic version: {self.version}")
        if isinstance(self.status, str):
            self.status = SkillStatus(self.status)

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillManifest":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass
class Skill:
    manifest: SkillManifest
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("skill content cannot be empty")
        if len(self.content) > 100_000:
            raise ValueError("skill content exceeds 100000 characters")

    def to_dict(self) -> Dict[str, Any]:
        return {"manifest": self.manifest.to_dict(), "content": self.content}


@dataclass
class SkillMatch:
    skill: Skill
    score: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill.manifest.skill_id,
            "version": self.skill.manifest.version,
            "score": self.score,
            "reasons": list(self.reasons),
        }


@dataclass
class ValidationReport:
    accepted: bool
    baseline_score: float
    candidate_score: float
    safety_passed: bool
    regression_passed: bool
    findings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _valid_version(value: str) -> bool:
    parts = value.split(".")
    return len(parts) == 3 and all(part.isdigit() for part in parts)

