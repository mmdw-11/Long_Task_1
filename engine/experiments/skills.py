"""Held-out procedural skill-reuse experiment.

Skills are learned from a small training partition and evaluated only on unseen
tasks.  This prevents a test task from creating and then retrieving its own
trajectory-derived skill.
"""

from __future__ import annotations

import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
    train_per_type: int = 3
    clean: bool = True


def run_skill_experiment(
    examples: List[SkillExample], config: SkillExperimentConfig | None = None
) -> ExperimentReport:
    cfg = config or SkillExperimentConfig()
    if cfg.method not in SKILL_METHODS:
        raise ValueError(f"unsupported skill method: {cfg.method}")
    train_examples, test_examples = _split_train_test(examples, cfg.train_per_type)
    if not test_examples:
        raise ValueError("skill evaluation requires at least one held-out test example")
    root = Path(cfg.output_root)
    if cfg.clean and root.exists():
        shutil.rmtree(root)
    repo = SkillRepository(root / "repo")
    retriever = SkillRetriever(repo, min_score=cfg.min_score, top_k=3, max_chars=6000)
    lifecycle = SkillEvolutionService(repository=repo)
    corpus = _publish_skill_corpus(train_examples, cfg, repo, lifecycle)
    rows = [_evaluate_held_out(example, cfg, retriever, corpus) for example in test_examples]
    return ExperimentReport(
        name=f"skill-{cfg.method}",
        rows=rows,
        metadata={
            "method": cfg.method,
            "min_score": cfg.min_score,
            "train_per_type": cfg.train_per_type,
            "train_examples": len(train_examples),
            "test_examples": len(test_examples),
            "skill_corpus": corpus,
            "repo": str(root / "repo"),
            "evaluator_scope": "held-out procedural-plan coverage; not an LLM task solver",
        },
    )


def _split_train_test(examples: List[SkillExample], train_per_type: int) -> Tuple[List[SkillExample], List[SkillExample]]:
    grouped: Dict[str, List[SkillExample]] = defaultdict(list)
    for example in examples:
        grouped[example.task_type].append(example)
    train: List[SkillExample] = []
    test: List[SkillExample] = []
    for task_type, group in grouped.items():
        explicit_train = [item for item in group if item.metadata.get("split") == "train"]
        explicit_test = [item for item in group if item.metadata.get("split") == "test"]
        if explicit_train or explicit_test:
            if not explicit_train or not explicit_test:
                raise ValueError(f"task_type {task_type!r} must contain both train and test examples")
            train.extend(explicit_train)
            test.extend(explicit_test)
            continue
        ordered = sorted(group, key=lambda item: item.id)
        if len(ordered) <= train_per_type:
            raise ValueError(f"task_type {task_type!r} needs more than {train_per_type} examples")
        train.extend(ordered[:train_per_type])
        test.extend(ordered[train_per_type:])
    return train, test


def _publish_skill_corpus(
    train_examples: List[SkillExample],
    cfg: SkillExperimentConfig,
    repo: SkillRepository,
    lifecycle: SkillEvolutionService,
) -> Dict[str, Dict[str, int]]:
    grouped: Dict[str, List[SkillExample]] = defaultdict(list)
    for example in train_examples:
        grouped[example.task_type].append(example)
    corpus: Dict[str, Dict[str, int]] = {}
    if cfg.method == "no_skill":
        return corpus
    for task_type, group in sorted(grouped.items()):
        prototype = group[0]
        if cfg.method == "manual_skill":
            content = _manual_skill(prototype)
            source_run_ids = [f"manual-{task_type}"]
            source_type = "manual"
        else:
            content = _learned_skill_from_trajectories(group)
            source_run_ids = [f"train-{item.id}" for item in group]
            source_type = "trajectory"
        skill = repo.create(
            skill_id=_safe_skill_id(f"{cfg.method}-{task_type}"),
            name=f"{task_type} 技能",
            content=content,
            status=SkillStatus.CANDIDATE,
            tags=[task_type, prototype.source],
            source_run_ids=source_run_ids,
            source_type=source_type,
            metadata={"legacy_manifest": {"version": "1.0.0", "task_types": [task_type], "applicable_nodes": ["solver"]}},
        )
        validation = lifecycle.validate(skill.id)
        if validation.passed:
            lifecycle.publish(skill.id, approved_by="experiment")
        corpus[task_type] = {
            "published": int(validation.passed),
            "source_examples": len(group) if cfg.method == "ours_skill_loop" else 0,
            "automatic_generation": int(cfg.method == "ours_skill_loop" and validation.passed),
        }
    return corpus


def _evaluate_held_out(
    example: SkillExample,
    cfg: SkillExperimentConfig,
    retriever: SkillRetriever,
    corpus: Dict[str, Dict[str, int]],
) -> ExperimentRow:
    matches = retriever.retrieve(example.task, node="solver", task_type=example.task_type, top_k=3)
    combined = "\n".join(match.skill.content for match in matches)
    coverage = _coverage(example.expected_steps, combined)
    critical_steps = [str(item) for item in example.metadata.get("critical_steps") or []]
    critical_coverage = _coverage(critical_steps, combined) if critical_steps else coverage
    retrieval_hit = bool(matches)
    task_success = coverage >= 0.5 and retrieval_hit
    corpus_info = corpus.get(example.task_type, {})
    return ExperimentRow(
        id=example.id,
        passed=task_success,
        score=coverage,
        prediction=combined,
        expected="; ".join(example.expected_steps),
        metrics={
            "task_success": int(task_success),
            "step_coverage": coverage,
            "critical_step_coverage": critical_coverage,
            "retrieval_hit": int(retrieval_hit),
            "automatic_skill_generation_success": corpus_info.get("automatic_generation", 0),
            "validation_pass_rate": corpus_info.get("published", 0),
            "training_source_examples": corpus_info.get("source_examples", 0),
            "manual_skill_provided": int(cfg.method == "manual_skill"),
            "retrieved": len(matches),
            "tokens": rough_token_count(combined),
            "top_score": matches[0].score if matches else 0.0,
        },
        metadata={"method": cfg.method, "task_type": example.task_type, "split": "test"},
    )


def _learned_skill_from_trajectories(examples: List[SkillExample]) -> str:
    steps: List[str] = []
    for example in examples:
        for step in _trajectory_steps(example.trajectory):
            if _norm(step) not in {_norm(item) for item in steps}:
                steps.append(step)
    return _render_skill(examples[0], steps)


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
    return [part.strip(" 。；;，,") for part in re.split(r"[。；;，,\n]", cleaned) if part.strip(" 。；;，,")][:6]


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
