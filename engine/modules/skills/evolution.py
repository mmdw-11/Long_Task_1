"""技能候选生成、验证与审批发布服务。

服务默认采用保守流程：从成功运行生成 Markdown 候选，结构校验通过后必须审批发布。
同时保留旧版多轨迹候选生成接口，便于已有业务代码平滑迁移到当前后端闭环。
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from ..workflows import RunRecord, RunStore
from ._types import SkillRecord, SkillStatus, SkillValidationReport
from .repository import SkillRepository
from .trace import SkillTraceStore


class SkillEvolutionService:
    """Coordinate the backend-only skill lifecycle."""

    def __init__(
        self,
        *legacy_args: Any,
        repository: Optional[SkillRepository] = None,
        run_store: Optional[RunStore] = None,
        trace_store: Optional[SkillTraceStore] = None,
        minimum_score_gain: float = 0.0,
        max_edit_ratio: float = 1.0,
        max_edit_chars: int = 10000,
    ) -> None:
        if legacy_args:
            repository = legacy_args[0]
            if len(legacy_args) > 1:
                trace_store = legacy_args[1]
        if repository is None:
            raise ValueError("repository is required")
        self.repository = repository
        self.run_store = run_store
        self.trace_store = trace_store
        self.minimum_score_gain = max(0.0, float(minimum_score_gain))
        self.max_edit_ratio = max(0.0, float(max_edit_ratio))
        self.max_edit_chars = max(0, int(max_edit_chars))

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

    def publish(
        self,
        skill_id: str,
        version: Optional[str | int] = None,
        *,
        approved_by: str,
    ) -> SkillRecord:
        if not approved_by.strip():
            raise ValueError("approved_by is required")
        skill = self.repository.get(skill_id, version)
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

    def create_candidate_from_runs(
        self,
        *,
        skill_id: str,
        name: str,
        run_ids: List[str],
        version: str = "1.0.0",
        task_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        content: Optional[str] = None,
    ) -> SkillRecord:
        """兼容旧接口：根据多条轨迹生成候选技能，并执行文本编辑预算控制。"""
        if not skill_id.strip():
            raise ValueError("skill_id is required")
        if not run_ids:
            raise ValueError("run_ids is required")
        existing = self.repository.get(skill_id) if self.repository.exists(skill_id) else None
        candidate_content = content or self._render_candidate_from_traces(run_ids)
        if existing is not None:
            self._validate_edit_budget(existing.content, candidate_content)
        return self.repository.create(
            skill_id=skill_id,
            name=name,
            content=candidate_content,
            status=SkillStatus.CANDIDATE,
            tags=tags or [],
            source_run_ids=run_ids,
            metadata={
                "legacy_manifest": {
                    "version": version,
                    "task_types": task_types or [],
                    "applicable_nodes": [],
                }
            },
        )

    def validate_candidate(
        self,
        skill_id: str,
        version: Optional[str | int],
        evaluator: Callable[[Optional[SkillRecord]], Dict[str, Any]],
    ) -> SkillValidationReport:
        """兼容旧接口：比较基线与候选分数，失败时写入拒绝编辑缓冲区。"""
        skill = self.repository.get(skill_id, version)
        baseline = evaluator(None) or {}
        candidate = evaluator(skill) or {}
        baseline_score = float(baseline.get("score") or 0.0)
        candidate_score = float(candidate.get("score") or 0.0)
        safety_passed = bool(candidate.get("safety_passed", True))
        regression_passed = bool(candidate.get("regression_passed", True))
        findings = [str(item) for item in candidate.get("findings") or []]
        accepted = (
            safety_passed
            and regression_passed
            and candidate_score >= baseline_score + self.minimum_score_gain
        )
        report = SkillValidationReport(
            skill_id=skill.id,
            passed=accepted,
            score=round(candidate_score, 4),
            findings=findings,
            metadata={
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "safety_passed": safety_passed,
                "regression_passed": regression_passed,
            },
        )
        if accepted:
            self.repository.transition(
                skill.id,
                SkillStatus.VALIDATED,
                metadata={"last_validation": report.to_dict()},
            )
        else:
            self.repository.transition(
                skill.id,
                SkillStatus.REJECTED,
                metadata={"last_validation": report.to_dict()},
            )
            self._append_rejected_edit(skill, report)
        return report

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

    def _render_candidate_from_traces(self, run_ids: List[str]) -> str:
        nodes: List[str] = []
        if self.trace_store is not None:
            for run_id in run_ids:
                for event in self.trace_store.list(run_id):
                    if event.node:
                        nodes.append(event.node)
        unique_nodes = sorted(set(nodes)) or ["unknown"]
        return "\n".join(
            [
                "# Learned procedural skill",
                "",
                "## 适用场景",
                "- 当任务与历史成功轨迹相似时复用。",
                "",
                "## 执行步骤",
                *[f"- 调用 `{node}` 节点完成对应子步骤。" for node in unique_nodes],
                "",
                "## 校验方式",
                "- 确认关键节点均已执行，并检查最终输出符合任务目标。",
            ]
        )

    def _validate_edit_budget(self, old_content: str, new_content: str) -> None:
        delta = abs(len(new_content) - len(old_content))
        changed = sum(1 for left, right in zip(old_content, new_content) if left != right) + delta
        allowed = min(self.max_edit_chars, max(1, int(len(old_content) * self.max_edit_ratio)))
        if changed > allowed:
            raise ValueError(f"candidate edit exceeds budget: {changed} > {allowed}")

    def _append_rejected_edit(self, skill: SkillRecord, report: SkillValidationReport) -> None:
        path = self.repository.root_dir / "rejected_edits.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {"skill_id": skill.id, "version": skill.version, "report": report.to_dict()},
                    ensure_ascii=False,
                )
                + "\n"
            )
