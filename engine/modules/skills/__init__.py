"""过程性技能生命周期模块。

技能以 Markdown 规程形式持久化，运行时只检索已发布技能；候选技能必须经过
验证和审批后才会进入在线执行链路。
"""

from ._types import (
    SKILL_CONTEXT_KEY,
    SKILL_CONTEXT_TEXT_KEY,
    Skill,
    SkillManifest,
    SkillMatch,
    SkillRecord,
    SkillStatus,
    SkillValidationReport,
)
from .evolution import SkillEvolutionService
from .repository import SkillRepository
from .installations import SkillInstallationStore
from .semantic import SkillSemanticIndex
from .retrieval import SkillRetriever
from .trace import SkillTraceEvent, SkillTraceStore

ValidationReport = SkillValidationReport

__all__ = [
    "SKILL_CONTEXT_KEY",
    "SKILL_CONTEXT_TEXT_KEY",
    "Skill",
    "SkillEvolutionService",
    "SkillManifest",
    "SkillMatch",
    "SkillRecord",
    "SkillRepository",
    "SkillInstallationStore",
    "SkillSemanticIndex",
    "SkillRetriever",
    "SkillStatus",
    "SkillTraceEvent",
    "SkillTraceStore",
    "SkillValidationReport",
    "ValidationReport",
]
