"""Skill candidate generation, validation, and approval workflow.

This service is deliberately conservative: it creates Markdown candidates from
observed runs, validates structure and evidence, then requires an approver
before a skill becomes retrievable in production execution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..workflows import RunRecord, RunStore
from ._types import SkillRecord, SkillStatus, SkillValidationReport
from .repository import SkillRepository
from .trace import SkillTraceStore


class SkillEvolutionService:
    """Coordinate the backend-only skill lifecycle."""

    def __init__(
        self,
        *,
        repository: SkillRepository,
        run_store: Optional[RunStore] = None,
        trace_store: Optional[SkillTraceStore] = None,
    ) -> None:
        self.repository = repository
        self.run_store = run_store
        self.trace_store = trace_store

    def create_candidate_from_run(
        self,
        run_id: str,
        *,
        name: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SkillRecord:
        if self.run_store is None:
            raise ValueError("run_store is required to create a skill from a run")
        run = self.run_store.get(run_id)
        if run.status != "succeeded":
            raise ValueError("only succeeded runs can generate skill candidates")
        content = self._render_candidate(run)
        return self.repository.create(
            name=name or f"Skill from {run.id}",
            description=description or "Candidate generated from a successful execution run.",
            tags=tags or [],
            metadata={**(metadata or {}), "source": "run", "run_status": run.status},
            source_run_ids=[run.id],
            status=SkillStatus.CANDIDATE,
            content=content,
        )

    def validate(self, skill_id: str) -> SkillValidationReport:
        skill = self.repository.get(skill_id)
        findings: List[str] = []
        if skill.status not in {SkillStatus.CANDIDATE, SkillStatus.VALIDATED}:
            findings.append("skill must be candidate or validated before validation")
        if len(skill.content.strip()) < 80:
            findings.append("skill content is too short")
        required_sections = ["## 适用场景", "## 执行步骤", "## 校验方式"]
        missing = [section for section in required_sections if section not in skill.content]
        if missing:
            findings.append("missing sections: " + ", ".join(missing))
        if not skill.source_run_ids:
            findings.append("source_run_ids is required for traceability")
        passed = not findings
        score = 1.0 if passed else max(0.0, 1.0 - 0.25 * len(findings))
        report = SkillValidationReport(
            skill_id=skill.id,
            passed=passed,
            score=round(score, 4),
            findings=findings,
            metadata={"required_sections": required_sections},
        )
        if passed and skill.status == SkillStatus.CANDIDATE:
            self.repository.transition(
                skill.id,
                SkillStatus.VALIDATED,
                metadata={"last_validation": report.to_dict()},
            )
        else:
            skill.metadata = {**skill.metadata, "last_validation": report.to_dict()}
            self.repository.save(skill)
        return report

    def publish(self, skill_id: str, *, approved_by: str) -> SkillRecord:
        if not approved_by.strip():
            raise ValueError("approved_by is required")
        skill = self.repository.get(skill_id)
        if skill.status == SkillStatus.CANDIDATE:
            report = self.validate(skill_id)
            if not report.passed:
                raise ValueError("skill validation failed; publish rejected")
            skill = self.repository.get(skill_id)
        if skill.status != SkillStatus.VALIDATED:
            raise ValueError("only validated skills can be published")
        return self.repository.transition(
            skill_id,
            SkillStatus.PUBLISHED,
            approved_by=approved_by,
            metadata={"published_by": approved_by},
        )

    def reject(self, skill_id: str, *, reason: str = "") -> SkillRecord:
        metadata = {"rejected_reason": reason} if reason else None
        return self.repository.transition(skill_id, SkillStatus.REJECTED, metadata=metadata)

    def retire(self, skill_id: str, *, reason: str = "") -> SkillRecord:
        metadata = {"retired_reason": reason} if reason else None
        return self.repository.transition(skill_id, SkillStatus.RETIRED, metadata=metadata)

    def _render_candidate(self, run: RunRecord) -> str:
        messages = run.state.get("messages") or []
        agents = [str(item.get("agent")) for item in messages if isinstance(item, dict)]
        route_events = [event for event in run.events if event.get("type") == "route"]
        trace_count = len(self.trace_store.list(run.id)) if self.trace_store else 0
        input_text = str(run.input.get("input") or run.input)
        output_text = str(messages[-1].get("content") if messages else run.state)
        return "\n".join(
            [
                f"# Skill from run {run.id}",
                "",
                "## 适用场景",
                f"- 当任务输入包含类似信息时可复用：{input_text[:300]}",
                f"- 已验证经过节点：{', '.join(agents) if agents else 'unknown'}",
                "",
                "## 执行步骤",
                "1. 读取任务输入，确认目标、约束和可用上下文。",
                "2. 按已保存工作流的节点顺序执行，并在每个节点结束后记录输出。",
                "3. 如路由事件存在，使用运行时状态中的分支字段选择后继节点。",
                "4. 将最终节点输出作为下游输入或最终结果返回。",
                "",
                "## 校验方式",
                f"- 来源运行状态必须为 succeeded，当前为：{run.status}",
                f"- 路由事件数量：{len(route_events)}",
                f"- 技能轨迹事件数量：{trace_count}",
                f"- 最终输出摘要：{output_text[:500]}",
            ]
        )
