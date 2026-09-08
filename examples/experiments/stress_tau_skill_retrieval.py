"""Run a zero-API retrieval stress test over all frozen τ test tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.experiments.tau_analysis import retrieval_metrics
from engine.experiments.tau_data import load_tau_tasks
from engine.experiments.tau_runtime import rank_tau_skills, select_tau_skills
from engine.experiments.tau_skills import TauSkill


def _load_skills(root: Path) -> list[TauSkill]:
    result = []
    for path in sorted(root.glob("*.json")):
        if path.name.endswith(".validation.json"):
            continue
        skill = TauSkill(**json.loads(path.read_text(encoding="utf-8")))
        if skill.status == "published":
            result.append(skill)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/tau_skill_v1/tasks.jsonl")
    parser.add_argument("--split", choices=("skill_train", "skill_validation", "test"), default="skill_validation")
    parser.add_argument("--skill-root", default="runs/experiments/tau_skill_v1/skills/ours_full")
    parser.add_argument("--output", default="runs/experiments/tau_skill_v1/offline/retrieval_stress.json")
    args = parser.parse_args()
    tasks = load_tau_tasks(args.dataset, split=args.split)
    skills = _load_skills(Path(args.skill_root))
    clean = retrieval_metrics(tasks, skills, rank_tau_skills, top_k=2)
    eventual = retrieval_metrics(tasks, skills, rank_tau_skills, top_k=2, oracle="eventual_action")
    full = retrieval_metrics(tasks, skills, rank_tau_skills, include_hidden_instructions=True, top_k=2)
    negated_noise = retrieval_metrics(
        tasks, skills, rank_tau_skills, top_k=2,
        query_suffix="I do not want to cancel, return, exchange, or book anything else.",
    )
    abstention_queries = {
        "retail": "Tell me a joke about astronomy.",
        "airline": "Write a poem about mountains.",
    }
    abstentions = {
        domain: not select_tau_skills(skills, query, domain)
        for domain, query in abstention_queries.items()
    }
    cross_domain_violations = sum(
        skill.domain != task.domain
        for task in tasks
        for skill, _score in select_tau_skills(skills, "cancel and change my booking or order", task.domain)
    )
    payload = {
        "skills": len(skills),
        "split": args.split,
        "clean_first_request": clean,
        "eventual_action_diagnostic": eventual,
        "full_scenario_analysis_only": full,
        "negated_distractor": negated_noise,
        "irrelevant_query_abstention": abstentions,
        "cross_domain_violations": cross_domain_violations,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print({
        "output": str(output), "skills": len(skills),
        "clean_hit_at_1": clean["hit_at_1"], "clean_recall_at_2": clean["recall_at_k"],
        "noise_hit_at_1": negated_noise["hit_at_1"], "noise_recall_at_2": negated_noise["recall_at_k"],
        "abstention": abstentions, "cross_domain_violations": cross_domain_violations,
    })


if __name__ == "__main__":
    main()
