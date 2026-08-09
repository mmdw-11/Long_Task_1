"""过程性技能检索与上下文注入。

当前使用轻量关键词重叠检索，并在检索阶段接入灰度发布判断。
后续替换为 BGE-M3 向量检索时，调用方接口保持不变。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ._types import SKILL_CONTEXT_KEY, SKILL_CONTEXT_TEXT_KEY, SkillMatch, SkillStatus
from .repository import SkillRepository


class SkillRetriever:
    """为当前节点执行检索已发布且命中灰度策略的 Markdown 技能。"""

    def __init__(
        self,
        repository: SkillRepository,
        *,
        top_k: int = 3,
        min_score: float = 0.0,
        max_chars: int = 4000,
    ) -> None:
        self.repository = repository
        self.top_k = max(1, top_k)
        self.min_score = max(0.0, float(min_score))
        self.max_chars = max(1, int(max_chars))

    def retrieve(
        self,
        query: str,
        *,
        node: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        task_type: str = "",
    ) -> List[SkillMatch]:
        metadata = metadata or {}
        effective_task_type = task_type or str(metadata.get("task_type") or "")
        query_terms = _terms(" ".join([query, node, effective_task_type, _metadata_text(metadata)]))
        if not query_terms:
            return []
        matches: List[SkillMatch] = []
        for skill in self.repository.list(status=SkillStatus.PUBLISHED):
            # 灰度比例由仓库统一判断，检索器只负责过滤不可生效的技能。
            if not self.repository.is_rollout_enabled(skill, query=query, node=node):
                continue
            manifest = skill.manifest
            if manifest.applicable_nodes and node and node not in manifest.applicable_nodes:
                continue
            if manifest.task_types and effective_task_type and effective_task_type not in manifest.task_types:
                continue
            skill_terms = _terms(
                " ".join(
                    [
                        skill.name,
                        skill.description,
                        " ".join(skill.tags),
                        " ".join(manifest.task_types),
                        skill.content,
                    ]
                )
            )
            if not skill_terms:
                continue
            overlap = query_terms & skill_terms
            if not overlap:
                continue
            score = len(overlap) / len(query_terms | skill_terms)
            if score < self.min_score:
                continue
            matches.append(
                SkillMatch(
                    skill=skill,
                    score=round(score, 6),
                    reason=f"matched terms: {', '.join(sorted(overlap)[:8])}",
                )
            )
        limit = top_k or self.top_k
        return sorted(matches, key=lambda item: item.score, reverse=True)[:limit]

    def inject(self, state: Dict[str, Any], *, node: str, metadata: Dict[str, Any]) -> List[SkillMatch]:
        query = " ".join(
            str(state.get(key) or "")
            for key in ("input", "task", "goal")
            if state.get(key) is not None
        )
        task_type = str(state.get("task_type") or metadata.get("task_type") or "")
        matches = self.retrieve(query, node=node, metadata=metadata, task_type=task_type)
        if not matches:
            state.pop(SKILL_CONTEXT_KEY, None)
            state.pop(SKILL_CONTEXT_TEXT_KEY, None)
            return []
        state[SKILL_CONTEXT_KEY] = [match.to_dict() for match in matches]
        state[SKILL_CONTEXT_TEXT_KEY] = render_skill_context(matches)[: self.max_chars]
        return matches


def render_skill_context(matches: List[SkillMatch]) -> str:
    sections = ["可复用技能："]
    for match in matches:
        block = "\n".join(
            [
                f"- {match.skill.name} ({match.skill.id}, score={match.score})",
                match.skill.content.strip(),
            ]
        )
        sections.append(block)
    return "\n\n".join(sections)


def _terms(text: str) -> set[str]:
    terms: set[str] = set()
    for item in re.findall(r"[\w\u4e00-\u9fff]{2,}", text or ""):
        token = item.lower()
        terms.add(token)
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", token)
        if len(chinese_chars) >= 2:
            # 中文没有天然空格，补充二字滑窗能显著提升短语匹配稳定性。
            terms.update(
                "".join(chinese_chars[index : index + 2])
                for index in range(len(chinese_chars) - 1)
            )
    return terms


def _metadata_text(metadata: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key} {value}")
    return " ".join(parts)
