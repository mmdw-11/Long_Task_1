"""在线技能检索与预算注入：先硬过滤，再轻量相关性排序并处理冲突。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from ._types import Skill, SkillMatch
from .repository import SkillRepository


SKILL_CONTEXT_KEY = "__skill_context__"
SKILL_CONTEXT_TEXT_KEY = "__skill_context_text__"


class SkillRetriever:
    def __init__(
        self,
        repository: SkillRepository,
        *,
        top_k: int = 3,
        min_score: float = 0.15,
        max_chars: int = 6000,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0 <= min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")
        if max_chars < 256:
            raise ValueError("max_chars must be at least 256")
        self.repository = repository
        self.top_k = top_k
        self.min_score = min_score
        self.max_chars = max_chars

    def retrieve(
        self,
        query: str,
        *,
        node: str = "",
        model: str = "",
        task_type: str = "",
        available_tools: Sequence[str] = (),
    ) -> List[SkillMatch]:
        if not query.strip():
            return []
        query_tokens = _tokens(" ".join((query, node, task_type)))
        available = set(available_tools)
        matches: List[SkillMatch] = []
        for skill in self.repository.list():
            manifest = skill.manifest
            if manifest.applicable_nodes and node not in manifest.applicable_nodes:
                continue
            if manifest.models and model and model not in manifest.models:
                continue
            if manifest.task_types and task_type and task_type not in manifest.task_types:
                continue
            if manifest.tools and available and not set(manifest.tools).issubset(available):
                continue
            haystack = " ".join([manifest.name, manifest.description, *manifest.tags, *manifest.task_types])
            skill_tokens = _tokens(haystack)
            overlap = len(query_tokens & skill_tokens)
            score = overlap / max(1, len(query_tokens | skill_tokens))
            # Explicit metadata matches are stronger and make Chinese/no-whitespace retrieval useful.
            reasons: List[str] = []
            lowered = query.lower()
            for tag in [*manifest.tags, *manifest.task_types]:
                if tag and tag.lower() in lowered:
                    score += 0.25
                    reasons.append(f"matched:{tag}")
            if node and node in manifest.applicable_nodes:
                score += 0.2
                reasons.append(f"node:{node}")
            if score >= self.min_score:
                matches.append(SkillMatch(skill, min(score, 1.0), reasons))
        matches.sort(key=lambda item: (-item.score, item.skill.manifest.skill_id))
        return self._remove_conflicts(matches[: self.top_k])

    def inject(self, state: Dict[str, Any], *, node: str, metadata: Dict[str, Any]) -> List[SkillMatch]:
        query = str(state.get("goal") or state.get("input") or state.get("task") or node)
        matches = self.retrieve(
            query,
            node=node,
            model=str(metadata.get("model") or ""),
            task_type=str(state.get("task_type") or metadata.get("task_type") or ""),
            available_tools=metadata.get("available_tools") or (),
        )
        selected: List[SkillMatch] = []
        chunks: List[str] = []
        used = 0
        for match in matches:
            skill = match.skill
            chunk = f"## Skill: {skill.manifest.name} ({skill.manifest.skill_id}@{skill.manifest.version})\n{skill.content.strip()}"
            if used + len(chunk) > self.max_chars:
                continue
            chunks.append(chunk)
            selected.append(match)
            used += len(chunk)
        state[SKILL_CONTEXT_KEY] = [match.to_dict() for match in selected]
        state[SKILL_CONTEXT_TEXT_KEY] = "\n\n".join(chunks)
        return selected

    @staticmethod
    def _remove_conflicts(matches: List[SkillMatch]) -> List[SkillMatch]:
        selected: List[SkillMatch] = []
        selected_ids = set()
        for match in matches:
            conflicts = set(match.skill.manifest.incompatible_skills)
            if conflicts & selected_ids:
                continue
            if any(match.skill.manifest.skill_id in set(item.skill.manifest.incompatible_skills) for item in selected):
                continue
            selected.append(match)
            selected_ids.add(match.skill.manifest.skill_id)
        return selected


def _tokens(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9_\-]+|[\u4e00-\u9fff]{2,}", text.lower()))
    return {word for word in words if len(word) > 1}

