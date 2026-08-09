"""Deterministic retrieval for published procedural skills.

The retriever uses lightweight token overlap today and keeps the interface ready
for a future BGE-M3 vector index without changing callers.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ._types import SKILL_CONTEXT_KEY, SKILL_CONTEXT_TEXT_KEY, SkillMatch, SkillStatus
from .repository import SkillRepository


class SkillRetriever:
    """Retrieve published Markdown skills for the current node execution."""

    def __init__(self, repository: SkillRepository, *, top_k: int = 3) -> None:
        self.repository = repository
        self.top_k = max(1, top_k)

    def retrieve(
        self,
        query: str,
        *,
        node: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[SkillMatch]:
        metadata = metadata or {}
        query_terms = _terms(" ".join([query, node, _metadata_text(metadata)]))
        if not query_terms:
            return []
        matches: List[SkillMatch] = []
        for skill in self.repository.list(status=SkillStatus.PUBLISHED):
            skill_terms = _terms(
                " ".join([skill.name, skill.description, " ".join(skill.tags), skill.content])
            )
            if not skill_terms:
                continue
            overlap = query_terms & skill_terms
            if not overlap:
                continue
            score = len(overlap) / len(query_terms | skill_terms)
            matches.append(
                SkillMatch(
                    skill=skill,
                    score=round(score, 6),
                    reason=f"matched terms: {', '.join(sorted(overlap)[:8])}",
                )
            )
        limit = top_k or self.top_k
        return sorted(matches, key=lambda item: item.score, reverse=True)[:limit]

    def inject(self, state: Dict[str, Any], *, node: str, metadata: Dict[str, Any]) -> None:
        query = str(state.get("input") or state.get("task") or "")
        matches = self.retrieve(query, node=node, metadata=metadata)
        if not matches:
            state.pop(SKILL_CONTEXT_KEY, None)
            state.pop(SKILL_CONTEXT_TEXT_KEY, None)
            return
        state[SKILL_CONTEXT_KEY] = [match.to_dict() for match in matches]
        state[SKILL_CONTEXT_TEXT_KEY] = render_skill_context(matches)


def render_skill_context(matches: List[SkillMatch]) -> str:
    sections = ["可复用技能："]
    for match in matches:
        sections.append(
            "\n".join(
                [
                    f"- {match.skill.name} ({match.skill.id}, score={match.score})",
                    match.skill.content.strip(),
                ]
            )
        )
    return "\n\n".join(sections)


def _terms(text: str) -> set[str]:
    return {item.lower() for item in re.findall(r"[\w\u4e00-\u9fff]{2,}", text or "")}


def _metadata_text(metadata: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key} {value}")
    return " ".join(parts)
