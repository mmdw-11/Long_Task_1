"""离线 SkillOpt-lite：从脱敏轨迹生成候选，执行硬门槛验证并审批发布。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ._types import Skill, SkillManifest, SkillStatus, ValidationReport
from .repository import SkillRepository
from .trace import SkillTraceStore


class SkillEvolutionService:
    def __init__(
        self,
        repository: SkillRepository,
        trace_store: SkillTraceStore,
        *,
        max_edit_ratio: float = 0.10,
        max_edit_chars: int = 300,
        minimum_score_gain: float = 0.01,
    ) -> None:
        if not 0 < max_edit_ratio <= 1:
            raise ValueError("max_edit_ratio must be in (0, 1]")
        if max_edit_chars <= 0:
            raise ValueError("max_edit_chars must be positive")
        self.repository = repository
        self.trace_store = trace_store
        self.max_edit_ratio = max_edit_ratio
        self.max_edit_chars = max_edit_chars
        self.minimum_score_gain = minimum_score_gain
        self.rejected_path = repository.root_dir / "rejected_edits.jsonl"

    def create_candidate_from_runs(
        self,
        *,
        skill_id: str,
        name: str,
        run_ids: Iterable[str],
        version: str,
        description: str = "",
        task_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        content: Optional[str] = None,
    ) -> Skill:
        unique_runs = list(dict.fromkeys(str(item) for item in run_ids if str(item)))
        if not unique_runs:
            raise ValueError("at least one source run is required")
        events = [event for run_id in unique_runs for event in self.trace_store.read(run_id)]
        if not events:
            raise ValueError("source runs contain no trace events")
        if content is None:
            content = self._render_candidate(name, events)
        parent = self._published_or_none(skill_id)
        if parent is not None:
            self._enforce_edit_budget(parent.content, content)
        manifest = SkillManifest(
            skill_id=skill_id,
            name=name,
            version=version,
            status=SkillStatus.CANDIDATE,
            description=description,
            task_types=list(task_types or []),
            tags=list(tags or []),
            source_runs=unique_runs,
            parent_version=parent.manifest.version if parent else "",
        )
        candidate = Skill(manifest, content)
        self.repository.save(candidate)
        return candidate

    def validate_candidate(
        self,
        skill_id: str,
        version: str,
        evaluator: Callable[[Optional[Skill]], Dict[str, Any]],
    ) -> ValidationReport:
        candidate = self.repository.get(skill_id, version, status=SkillStatus.CANDIDATE)
        baseline = evaluator(self._published_or_none(skill_id))
        result = evaluator(candidate)
        baseline_score = _score(baseline)
        candidate_score = _score(result)
        safety = bool(result.get("safety_passed", False))
        regression = bool(result.get("regression_passed", False))
        findings = [str(item) for item in result.get("findings") or []]
        accepted = safety and regression and candidate_score >= baseline_score + self.minimum_score_gain
        report = ValidationReport(
            accepted=accepted,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            safety_passed=safety,
            regression_passed=regression,
            findings=findings,
            metrics={"baseline": baseline, "candidate": result},
        )
        if accepted:
            manifest = SkillManifest.from_dict(candidate.manifest.to_dict())
            manifest.status = SkillStatus.VALIDATED
            manifest.metrics = report.to_dict()
            self.repository.save(Skill(manifest, candidate.content), overwrite=True)
        else:
            rejected_manifest = SkillManifest.from_dict(candidate.manifest.to_dict())
            rejected_manifest.status = SkillStatus.REJECTED
            rejected_manifest.metrics = report.to_dict()
            self.repository.save(Skill(rejected_manifest, candidate.content), overwrite=True)
            self._record_rejection(candidate, report)
        return report

    def publish(self, skill_id: str, version: str, *, approved_by: str) -> Skill:
        return self.repository.publish(skill_id, version, approved_by=approved_by)

    def _published_or_none(self, skill_id: str) -> Optional[Skill]:
        try:
            return self.repository.get(skill_id)
        except KeyError:
            return None

    def _enforce_edit_budget(self, old: str, new: str) -> None:
        prefix = 0
        for left, right in zip(old, new):
            if left != right:
                break
            prefix += 1
        old_tail, new_tail = old[prefix:], new[prefix:]
        suffix = 0
        for left, right in zip(reversed(old_tail), reversed(new_tail)):
            if left != right:
                break
            suffix += 1
        changed = max(len(old_tail) - suffix, len(new_tail) - suffix)
        budget = max(1, min(self.max_edit_chars, int(len(old) * self.max_edit_ratio)))
        if changed > budget:
            raise ValueError(f"candidate edit exceeds budget: changed={changed}, budget={budget}")

    @staticmethod
    def _render_candidate(name: str, events: List[Dict[str, Any]]) -> str:
        failures = []
        successful_nodes = []
        for event in events:
            if event.get("event_type") == "node_error":
                failures.append(str(event.get("payload") or "unknown failure"))
            if event.get("event_type") == "node_end" and (event.get("evaluation") or {}).get("passed", True):
                successful_nodes.append(str(event.get("node") or ""))
        steps = list(dict.fromkeys(item for item in successful_nodes if item))
        lines = [f"# {name}", "", "## 适用条件", "", "用于与来源轨迹同类型且工具、权限和环境条件一致的任务。", "", "## 执行步骤", ""]
        lines.extend(f"{idx}. 执行并核验 `{node}` 阶段的输出。" for idx, node in enumerate(steps, 1))
        if not steps:
            lines.append("1. 明确输入契约，执行任务并用确定性断言核验结果。")
        lines.extend(["", "## 失败处理", ""])
        if failures:
            for failure in list(dict.fromkeys(failures))[:5]:
                lines.append(f"- 遇到 `{failure[:160]}` 时停止传播未验证结果，并按恢复策略重试或改道。")
        else:
            lines.append("- 任何关键断言失败时，不得将结果提升为已验证事实。")
        lines.extend(["", "## 输出核验", "", "- 核对必需字段、外部副作用状态、父级授权与安全脱敏结果。", "- 未通过核验时返回结构化失败，不得声称任务成功。"])
        return "\n".join(lines) + "\n"

    def _record_rejection(self, candidate: Skill, report: ValidationReport) -> None:
        entry = {"timestamp": time.time(), "skill_id": candidate.manifest.skill_id, "version": candidate.manifest.version, "report": report.to_dict()}
        with self.rejected_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def _score(result: Dict[str, Any]) -> float:
    try:
        score = float(result.get("score", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluator result score must be numeric") from exc
    if not 0 <= score <= 1:
        raise ValueError("evaluator score must be between 0 and 1")
    return score
