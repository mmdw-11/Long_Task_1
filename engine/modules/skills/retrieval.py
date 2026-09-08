"""过程性技能检索与上下文注入。

当前使用轻量关键词重叠检索，并在检索阶段接入灰度发布判断。
后续替换为 BGE-M3 向量检索时，调用方接口保持不变。
"""

from __future__ import annotations

import re
import math
import time
from typing import Any, Dict, List, Optional

from ._types import SKILL_CONTEXT_KEY, SKILL_CONTEXT_TEXT_KEY, SkillMatch, SkillStatus
from .repository import SkillRepository
from .semantic import SkillSemanticIndex


class SkillRetriever:
    """为当前节点执行检索已发布且命中灰度策略的 Markdown 技能。"""

    def __init__(
        self,
        repository: SkillRepository,
        *,
        top_k: int = 3,
        min_score: float = 0.0,
        max_chars: int = 4000,
        semantic_index: Optional[SkillSemanticIndex] = None,
    ) -> None:
        self.repository = repository
        self.top_k = max(1, top_k)
        self.min_score = max(0.0, float(min_score))
        self.max_chars = max(1, int(max_chars))
        self.semantic_index = semantic_index
        self.last_diagnostics: Dict[str, Any] = {}

    def retrieve(
        self,
        query: str,
        *,
        node: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        task_type: str = "",
        mode: str = "hybrid",
    ) -> List[SkillMatch]:
        details = self.retrieve_detailed(query,node=node,metadata=metadata,top_k=top_k,task_type=task_type,mode=mode)
        return [item["match"] for item in details["results"]]

    def retrieve_detailed(self, query: str, *, node: str = "", metadata: Optional[Dict[str, Any]] = None,
                          top_k: Optional[int] = None, task_type: str = "", mode: str = "hybrid",
                          candidate_pool: bool = False) -> Dict[str, Any]:
        started=time.perf_counter()
        metadata = metadata or {}
        effective_task_type = task_type or str(metadata.get("task_type") or "")
        query_terms = _terms(" ".join([query, node, effective_task_type, _metadata_text(metadata)]))
        allowed_ids = None
        if "skill_ids" in metadata:
            allowed_ids = {str(item) for item in metadata.get("skill_ids") or []}
            if not allowed_ids:
                return {"results":[],"mode":mode,"requested_mode":mode,"degraded":False,"degraded_reason":"","embedding_model":self.semantic_index.embedding_model if self.semantic_index else None,"reason":"没有允许使用的 Skill","latency_ms":0.0}
        if not query_terms:
            return {"results":[],"mode":mode,"requested_mode":mode,"degraded":False,"degraded_reason":"","embedding_model":self.semantic_index.embedding_model if self.semantic_index else None,"reason":"查询内容为空","latency_ms":0.0}
        semantic_scores:Dict[str,float]={};degraded=False;degraded_reason=""
        if mode in {"semantic","hybrid"}:
            if self.semantic_index is None:
                degraded=True;degraded_reason="语义索引未配置"
            else:
                try:
                    health=self.semantic_index.health()
                    if not health.get("ready"):
                        degraded=True;degraded_reason=health.get("load_error") or "语义索引尚未建立"
                    else:
                        semantic_scores={x["skill_id"]:float(x["semantic_score"]) for x in self.semantic_index.search(query,allowed_ids=allowed_ids,limit=20)}
                except Exception as exc:degraded=True;degraded_reason=str(exc)[:300]
        matches: List[SkillMatch] = []
        details:List[Dict[str,Any]]=[]
        for skill in self.repository.list(status=SkillStatus.PUBLISHED):
            if allowed_ids is not None and skill.id not in allowed_ids:
                continue
            # 灰度比例由仓库统一判断，检索器只负责过滤不可生效的技能。
            if not self.repository.is_rollout_enabled(skill, query=query, node=node):
                continue
            manifest = skill.manifest
            if manifest.applicable_nodes and node and node not in manifest.applicable_nodes:
                continue
            if manifest.task_types and effective_task_type and effective_task_type not in manifest.task_types:
                continue
            name_terms = _terms(skill.name)
            tag_terms = _terms(" ".join([*skill.tags, *manifest.task_types]))
            description_terms = _terms(skill.description)
            content_terms = _terms(skill.content)
            skill_terms = name_terms | tag_terms | description_terms | content_terms
            if not skill_terms:
                continue
            overlap = query_terms & skill_terms
            if not overlap and skill.id not in semantic_scores:
                continue
            # Do not divide by the full SKILL.md vocabulary: a well-documented
            # Skill would otherwise receive a lower score simply because its
            # instructions are longer.  Matching title/tags is the strongest
            # signal, while body-only overlap remains deliberately weak.
            signal = sum(
                2.4 if term in name_terms else
                2.0 if term in tag_terms else
                1.0 if term in description_terms else 0.25
                for term in overlap
            )
            rule_score = 1.0 - math.exp(-signal / 5.0)
            semantic_score=semantic_scores.get(skill.id,0.0)
            context_score=1.0 if allowed_ids is not None and skill.id in allowed_ids else 0.0
            if mode=="rules" or degraded:score=rule_score
            elif mode=="semantic":score=semantic_score
            else:
                weighted=.65*semantic_score+.25*rule_score+.10*context_score
                # Strong explicit evidence must not be drowned out by a weak
                # embedding backend (notably the development hashing model).
                # Dense cosine is already calibrated as a relevance signal.
                # Hybrid mode may improve it with explicit/context evidence,
                # but must not make a valid semantic hit disappear merely
                # because the short user query has no literal overlap.
                score=max(weighted,.85*rule_score,semantic_score)
            effective_min=max(self.min_score,0.5 if mode in {"semantic","hybrid"} and not degraded else self.min_score)
            if score < effective_min and not candidate_pool:
                continue
            reason=f"命中关键词：{', '.join(sorted(overlap)[:8])}" if overlap else "BGE-M3 判断语义相关"
            match=SkillMatch(
                    skill=skill,
                    score=round(score, 6),
                    reason=reason,
                );matches.append(match);details.append({"match":match,"semantic_score":round(semantic_score,6),"rule_score":round(rule_score,6),"context_score":round(context_score,6),"rerank_score":None,"selected":True})
        limit = top_k or self.top_k
        ordered=sorted(details,key=lambda item:item["match"].score,reverse=True)[:limit]
        result={"results":ordered,"mode":"rules" if degraded else mode,"requested_mode":mode,"degraded":degraded,"degraded_reason":degraded_reason,"embedding_model":self.semantic_index.embedding_model if self.semantic_index else None,"latency_ms":round((time.perf_counter()-started)*1000,2)}
        self.last_diagnostics=result
        return result

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
    normalized = (text or "").lower()
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
    # Lightweight concept normalization covers common paraphrases without
    # requiring an embedding service. When any alias appears, all aliases in
    # that business concept become comparable terms.
    for aliases in _CONCEPT_ALIASES:
        if any(alias in normalized or alias in terms for alias in aliases):
            terms.update(aliases)
    return terms


_CONCEPT_ALIASES = (
    {"旅游", "旅行", "出游", "行程", "游玩", "度假", "出去玩", "出去走走", "散心"},
    {"邮件", "电邮", "回信", "回复邮件", "商务函件"},
    {"会议", "开会", "纪要", "会议记录", "会后总结"},
    {"研究", "调研", "调查", "研究报告", "分析报告"},
    {"网页", "网站", "页面", "前端", "web"},
    {"开发", "编程", "编码", "写代码", "软件工程"},
    {"数据分析", "统计分析", "指标分析", "报表", "数据洞察"},
    {"写作", "撰写", "起草", "写一份", "生成文案"},
    {"总结", "归纳", "概括", "摘要", "提炼"},
    {"计划", "规划", "安排", "方案"},
)


def _metadata_text(metadata: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key} {value}")
    return " ".join(parts)
