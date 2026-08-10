"""技能演化实验 runner。

用于 SkillEvolBench 小子集。实验不直接要求真实 LLM：它把历史轨迹压成候选 Markdown
技能，走本项目的验证、发布、检索注入链路，再检查任务需要的关键步骤是否能被技能命中。
这样可以先证明后端技能闭环本身可复现，之后再接真实 Solver 比较最终任务成功率。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

from engine.modules.skills import SkillRepository, SkillRetriever, SkillStatus

from .reports import ExperimentReport
from .types import ExperimentRow, SkillExample


@dataclass
class SkillExperimentConfig:
    """技能实验配置。"""

    output_root: str = "runs/experiments/skills"
    min_score: float = 0.01
    clean: bool = True


def run_skill_experiment(
    examples: List[SkillExample],
    config: SkillExperimentConfig | None = None,
) -> ExperimentReport:
    """运行技能生成、发布、检索复用实验。"""
    cfg = config or SkillExperimentConfig()
    root = Path(cfg.output_root)
    if cfg.clean and root.exists():
        shutil.rmtree(root)
    repo = SkillRepository(root / "repo")
    retriever = SkillRetriever(repo, min_score=cfg.min_score, top_k=3, max_chars=6000)
    rows: List[ExperimentRow] = []

    for example in examples:
        skill_id = _safe_skill_id(example.id)
        content = _render_skill_markdown(example)
        candidate = repo.create(
            skill_id=skill_id,
            name=f"技能-{example.id}",
            content=content,
            status=SkillStatus.CANDIDATE,
            tags=[example.task_type, example.source],
            source_run_ids=[f"dataset-{example.id}"],
            metadata={
                "legacy_manifest": {
                    "version": "1.0.0",
                    "task_types": [example.task_type],
                    "applicable_nodes": ["solver"],
                },
                "source": example.source,
            },
        )
        repo.transition(candidate.id, SkillStatus.VALIDATED, metadata={"validated_by": "experiment"})
        repo.publish(candidate.id, approved_by="experiment")

        matches = retriever.retrieve(example.task, node="solver", task_type=example.task_type, top_k=3)
        combined = "\n".join(match.skill.content for match in matches)
        covered = _coverage(example.expected_steps, combined)
        rows.append(
            ExperimentRow(
                id=example.id,
                passed=covered >= 0.5 and bool(matches),
                score=covered,
                prediction=combined,
                expected="; ".join(example.expected_steps),
                metrics={
                    "coverage": covered,
                    "retrieved": len(matches),
                    "top_score": matches[0].score if matches else 0.0,
                },
                metadata={"skill_id": skill_id, "task_type": example.task_type},
            )
        )

    return ExperimentReport(
        name="skill-evolution",
        rows=rows,
        metadata={"min_score": cfg.min_score, "repo": str(root / "repo")},
    )


def _render_skill_markdown(example: SkillExample) -> str:
    # 这些标题与 SkillEvolutionService.validate 的结构要求保持一致。
    steps = "\n".join(f"{index}. {step}" for index, step in enumerate(example.expected_steps, 1))
    return "\n".join(
        [
            f"# {example.task_type} 任务执行技能",
            "",
            "## 适用场景",
            "",
            f"- 任务类型：{example.task_type}",
            f"- 当前任务：{example.task}",
            "",
            "## 执行步骤",
            "",
            steps or "1. 阅读历史轨迹并抽取可复用动作。",
            "",
            "## 校验方式",
            "",
            "- 检查输出是否覆盖关键步骤。",
            "- 检查是否复用了历史轨迹中的成功经验。",
            "",
            "## 历史轨迹",
            "",
            example.trajectory,
            "",
        ]
    )


def _coverage(expected_steps: List[str], content: str) -> float:
    if not expected_steps:
        return 1.0 if content.strip() else 0.0
    content_norm = _norm(content)
    hits = sum(1 for step in expected_steps if _norm(step) and _norm(step) in content_norm)
    return round(hits / len(expected_steps), 6)


def _norm(text: str) -> str:
    return "".join(str(text).lower().split())


def _safe_skill_id(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)
    return clean.strip("-_") or "skill-example"
