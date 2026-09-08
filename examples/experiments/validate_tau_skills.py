"""Execute candidate skills on skill_validation tasks before publication."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.experiments.tau_data import load_tau_tasks
from engine.experiments.tau_runtime import TauRunConfig, _load_official_tasks, load_skill, run_one_tau_task
from engine.experiments.tau_skills import save_tau_skill


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/tau_skill_v1/tasks.jsonl")
    parser.add_argument("--skill-root", default="runs/experiments/tau_skill_v1/skills")
    parser.add_argument("--domain", choices=["retail", "airline"], required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--agent-model", default=os.getenv("TAU_AGENT_MODEL", "openai/deepseek-v4-flash"))
    parser.add_argument("--user-model", default=os.getenv("TAU_USER_MODEL", "openai/deepseek-v4-flash"))
    parser.add_argument("--evaluator-model", default=os.getenv("TAU_EVALUATOR_MODEL", "openai/deepseek-v4-flash"))
    args = parser.parse_args()

    root = Path(args.skill_root) / "ours_full"
    skills = [load_skill(path) for path in sorted(root.glob(f"{args.domain}*.json")) if not path.name.endswith(".validation.json")]
    skills = [skill for skill in skills if skill is not None]
    if not skills:
        raise FileNotFoundError(f"no {args.domain} candidate skills in {root}")
    selected = load_tau_tasks(args.dataset, split="skill_validation", domains=[args.domain])[: args.limit]
    official = _load_official_tasks(args.domain)
    config = TauRunConfig(
        seed=args.seed, max_steps=args.max_steps, timeout=args.timeout, trials=1,
        agent_model=args.agent_model, user_model=args.user_model,
        evaluator_model=args.evaluator_model,
    )
    rows = []
    for item in selected:
        task = official[item.metadata["official_task_id"]]
        baseline = run_one_tau_task(domain=args.domain, task=task, method="no_skill", trial=0, config=config)
        candidate = run_one_tau_task(domain=args.domain, task=task, method="ours_full", trial=0, config=config, skill=skills)
        rows.append({"task_id": str(task.id), "baseline": baseline, "candidate": candidate})
        print({"task": task.id, "baseline": baseline["reward"], "candidate": candidate["reward"]}, flush=True)
    baseline_reward = sum(row["baseline"]["reward"] for row in rows) / max(1, len(rows))
    candidate_reward = sum(row["candidate"]["reward"] for row in rows) / max(1, len(rows))
    regressions = sum(row["candidate"]["reward"] < row["baseline"]["reward"] for row in rows)
    tool_error_delta = sum(row["candidate"]["tool_errors"] - row["baseline"]["tool_errors"] for row in rows)
    unauthorized = sum(
        row["candidate"]["write_calls"] > 0 and row["candidate"]["write_confirmation_rate"] != 1.0
        for row in rows
    )
    accepted = candidate_reward >= baseline_reward and regressions == 0 and tool_error_delta <= 0 and unauthorized == 0
    report = {
        "domain": args.domain, "tasks": len(rows), "accepted": accepted,
        "baseline_reward": baseline_reward, "candidate_reward": candidate_reward,
        "regressions": regressions, "tool_error_delta": tool_error_delta,
        "unauthorized_mutations": unauthorized, "rows": rows,
    }
    report_path = root / f"{args.domain}.validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    for skill in skills:
        skill.status = "published" if accepted else "rejected"
        skill.metadata.update({
            "validation_report": str(report_path), "validation_reward": candidate_reward,
            "baseline_validation_reward": baseline_reward, "validation_accepted": accepted,
            "publication_blocked_pending_replay": False,
        })
        save_tau_skill(skill, root)
    print({"report": str(report_path), "status": "published" if accepted else "rejected", "skills": len(skills)})
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
