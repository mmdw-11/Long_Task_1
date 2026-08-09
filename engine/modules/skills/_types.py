"""过程性技能共享数据结构。

核心结构是 SkillRecord；SkillManifest 和 Skill 用于兼容早期业务代码里的技能对象写法。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


SKILL_CONTEXT_KEY = "__skill_context__"
SKILL_CONTEXT_TEXT_KEY = "__skill_context_text__"


class SkillStatus(str, Enum):
    DRAFT = "draft"
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    PUBLISHED = "published"
    REJECTED = "rejected"
    RETIRED = "retired"


@dataclass
class SkillManifest:
    """旧版技能清单结构，保留给已有调用方平滑迁移。"""

    skill_id: str
    name: str
    version: str = "1.0.0"
    status: SkillStatus = SkillStatus.DRAFT
    description: str = ""
    task_types: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    applicable_nodes: List[str] = field(default_factory=list)
    approved_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Skill:
    """旧版技能对象，内部会被仓库转换为 SkillRecord 存储。"""

    manifest: SkillManifest
    content: str


@dataclass
class SkillRecord:
    """A persisted Markdown skill plus lifecycle metadata."""

    id: str
    name: str
    status: SkillStatus
    content: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_run_ids: List[str] = field(default_factory=list)
    approved_by: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    version: int = 1

    @property
    def manifest(self) -> SkillManifest:
        """兼容旧版 skill.manifest 读取方式。"""
        legacy = dict(self.metadata.get("legacy_manifest") or {})
        return SkillManifest(
            skill_id=self.id,
            name=self.name,
            version=str(legacy.get("version") or self.version),
            status=self.status,
            description=self.description,
            task_types=[str(item) for item in legacy.get("task_types") or []],
            tags=list(self.tags),
            applicable_nodes=[str(item) for item in legacy.get("applicable_nodes") or []],
            approved_by=self.approved_by,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "description": self.description,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "source_run_ids": list(self.source_run_ids),
            "approved_by": self.approved_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillRecord":
        if not isinstance(data, dict):
            raise ValueError("skill payload must be an object")
        name = str(data.get("name") or "").strip()
        content = str(data.get("content") or "").strip()
        if not name:
            raise ValueError("skill.name is required")
        if not content:
            raise ValueError("skill.content is required")
        return cls(
            id=str(data.get("id") or "").strip(),
            name=name,
            status=SkillStatus(str(data.get("status") or SkillStatus.DRAFT.value)),
            content=content,
            description=str(data.get("description") or ""),
            tags=[str(item) for item in data.get("tags") or []],
            metadata=dict(data.get("metadata") or {}),
            source_run_ids=[str(item) for item in data.get("source_run_ids") or []],
            approved_by=data.get("approved_by"),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
            version=int(data.get("version") or 1),
        )


@dataclass
class SkillMatch:
    """A retrieved skill with a deterministic score."""

    skill: SkillRecord
    score: float
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill.id,
            "version": str(self.skill.manifest.version),
            "score": self.score,
            "reason": self.reason,
            "skill": self.skill.to_dict(),
        }


@dataclass
class SkillValidationReport:
    """Validation result used before a candidate can be published."""

    skill_id: str
    passed: bool
    score: float
    findings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        """兼容旧版评测报告命名。"""
        return self.passed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "passed": self.passed,
            "score": self.score,
            "findings": list(self.findings),
            "metadata": dict(self.metadata),
        }
