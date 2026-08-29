"""Small, auditable skill-reuse experiment with three required controls."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from engine.modules.context.budget import rough_token_count
from engine.modules.skills import SkillEvolutionService, SkillRepository, SkillRetriever, SkillStatus

from .reports import ExperimentReport
from .types import ExperimentRow, SkillExample


SKILL_METHODS = ("no_skill", "manual_skill", "ours_skill_loop")


@dataclass
class SkillExperimentConfig:
    output_root: str = "runs/experiments/skills"
    method: str = "ours_skill_loop"
    min_score: float = 0.01
    clean: bool = True


def run_skill_experiment(
    examples: List[SkillExample], config: SkillExperimentConfig | None = None
) -> ExperimentReport:
    cfg = config or SkillExperimentConfig()
    if cfg.method not in SKILL_METHODS:
        raise ValueError(f"unsupported skill method: {cfg.method}")
    root = Path(cfg.output_root)
    if cfg.clean and root.exists():
        shutil.rmtree(root)
    repo = SkillRepository(root / "repo")
    retriever = SkillRetriever(repo, min_score=cfg.min_score, top_k=3, max_chars=6000)
    lifecycle = SkillEvolutionService(repository=repo)
    rows = [_run_one(example, cfg, repo, retriever, lifecycle) for example in examples]
    return ExperimentReport(
        name=f"skill-{cfg.method}", rows=rows,
        metadata={"method": cfg.method, "min_score": cfg.min_score, "repo": str(root / "repo"),
                  "evaluator_scope": "deterministic procedural-plan coverage; not an LLM task solver"},
    )


def _run_one(
    example: SkillExample, cfg: SkillExperimentConfig, repo: SkillRepository,
    retriever: SkillRetriever, lifecycle: SkillEvolutionService,
) -> ExperimentRow:
    skill_generated = validation_passed = False
    if cfg.method != "no_skill":
        content = _manual_skill(example) if cfg.method == "manual_skill" else _learned_skill_from_trajectory(example)
        skill = repo.create(
            skill_id=_safe_skill_id(f"{cfg.method}-{example.id}"), name=f"技能-{example.id}", content=content,
            status=SkillStatus.CANDIDATE, tags=[example.task_type, example.source],
            source_run_ids=[f"dataset-{example.id}"], source_type="manual" if cfg.method == "manual_skill" else "trajectory",
            metadata={"legacy_manifest": {"version": "1.0.0", "task_types": [example.task_type], "applicable_nodes": ["solver"]}},
        )
        skill_generated = True
        validation = lifecycle.validate(skill.id)
        validation_passed = validation.passed
        if validation_passed:
            lifecycle.publish(skill.id, approved_by="experiment")

    matches = retriever.retrieve(example.task, node="solver", task_type=example.task_type, top_k=3)
    combined = "\n".join(match.skill.content for match in matches)
    coverage = _coverage(example.expected_steps, combined)
    retrieval_hit = bool(matches)
    task_success = coverage >= 0.5 and retrieval_hit
    return ExperimentRow(
        id=example.id, passed=task_success, score=coverage, prediction=combined,
        expected="; ".join(example.expected_steps),
        metrics={
            "task_success": int(task_success), "step_coverage": coverage,
            "retrieval_hit": int(retrieval_hit), "skill_generation_success": int(skill_generated),
            "validation_pass_rate": int(validation_passed), "retrieved": len(matches),
            "tokens": rough_token_count(combined), "top_score": matches[0].score if matches else 0.0,
        },
        metadata={"method": cfg.method, "task_type": example.task_type},
    )


def _learned_skill_from_trajectory(example: SkillExample) -> str:
    steps = _trajectory_steps(example.trajectory)
    return _render_skill(example, steps)


def _manual_skill(example: SkillExample) -> str:
    library: Dict[str, List[str]] = {
        "email": ["读取邮件", "提取行动项", "生成并核对回复"],
        "calendar": ["读取参与人日历", "检查时间冲突", "创建提醒"],
        "travel_report": ["确认行程约束", "比较候选方案", "生成最终报告"],
    }
    return _render_skill(example, library.get(example.task_type, ["确认目标", "执行关键步骤", "核对结果"]))


def _render_skill(example: SkillExample, steps: List[str]) -> str:
    numbered = "\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1))
    return "\n".join([
        f"# {example.task_type} 任务执行技能", "", "## 适用场景", f"- 任务类型：{example.task_type}",
        "", "## 执行步骤", numbered or "1. 确认任务目标。", "", "## 校验方式",
        "- 检查关键步骤是否覆盖。", "- 检查输出是否符合任务约束。", "",
    ])


def _trajectory_steps(trajectory: str) -> List[str]:
    cleaned = re.sub(r"^(成功轨迹|成功经验)[:：]", "", trajectory.strip())
    parts = [part.strip(" 。；;，,") for part in re.split(r"[。；;，,\n]", cleaned) if part.strip(" 。；;，,")]
    return parts[:6] or ["确认任务目标", "执行历史成功步骤", "核对最终结果"]


def _coverage(expected_steps: List[str], content: str) -> float:
    if not expected_steps:
        return 1.0 if content.strip() else 0.0
    normalized = _norm(content)
    return round(sum(1 for step in expected_steps if _norm(step) in normalized) / len(expected_steps), 6)


def _norm(text: str) -> str:
    return "".join(str(text).lower().split())


def _safe_skill_id(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)
    return clean.strip("-_") or "skill-example"
