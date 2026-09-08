"""Run the stateful τ skill experiment in the dedicated .venv-tau."""

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
from engine.experiments.tau_runtime import (
    TAU_METHODS, TauRunConfig, _load_official_tasks, load_skill,
    run_one_tau_task, simulation_key, summarize_tau_rows,
)
from engine.experiments.tau_skills import manual_skill


def _jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _tools(environment) -> set[str]:
    return {tool.name for tool in environment.get_tools()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/tau_skill_v1/tasks.jsonl")
    parser.add_argument("--output-root", default="runs/experiments/tau_skill_v1")
    parser.add_argument("--domains", nargs="+", default=["retail", "airline"])
    parser.add_argument("--methods", nargs="+", choices=TAU_METHODS, default=list(TAU_METHODS))
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--limit-per-domain",
        type=int,
        help="Deterministically keep this many test tasks from every selected domain.",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--skill-context-budget-chars", type=int, default=12_000)
    parser.add_argument("--agent-model", default=os.getenv("TAU_AGENT_MODEL", "openai/deepseek-v4-flash"))
    parser.add_argument("--user-model", default=os.getenv("TAU_USER_MODEL", "openai/deepseek-v4-flash"))
    parser.add_argument("--evaluator-model", default=os.getenv("TAU_EVALUATOR_MODEL", "openai/deepseek-v4-flash"))
    parser.add_argument("--skill-root", default="runs/experiments/tau_skill_v1/skills")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    config = TauRunConfig(
        dataset=args.dataset, output_root=args.output_root, methods=tuple(args.methods),
        domains=tuple(args.domains), trials=args.trials, seed=args.seed,
        agent_model=args.agent_model, user_model=args.user_model, evaluator_model=args.evaluator_model,
        max_steps=args.max_steps, max_errors=args.max_errors, timeout=args.timeout,
        skill_context_budget_chars=args.skill_context_budget_chars,
        fresh=args.fresh, limit=args.limit,
    )
    selected = load_tau_tasks(config.dataset, split="test", domains=config.domains)
    selected = sorted(selected, key=lambda item: (item.domain, item.id))
    if args.limit_per_domain is not None:
        per_domain = {domain: 0 for domain in config.domains}
        balanced = []
        for item in selected:
            if per_domain[item.domain] < args.limit_per_domain:
                balanced.append(item)
                per_domain[item.domain] += 1
        selected = balanced
    if config.limit is not None:
        selected = selected[: config.limit]
    output = Path(config.output_root)
    rows_path = output / "rows.jsonl"
    if config.fresh and rows_path.exists():
        rows_path.unlink()
    rows = _jsonl_rows(rows_path)
    completed = {row["key"] for row in rows}
    official = {domain: _load_official_tasks(domain) for domain in config.domains}

    expected = {
        simulation_key(item.domain, method, trial, item.metadata["official_task_id"])
        for item in selected for method in config.methods for trial in range(config.trials)
    }
    skill_root = Path(args.skill_root)
    for item in selected:
        task = official[item.domain][item.metadata["official_task_id"]]
        for trial in range(config.trials):
            for method in config.methods:
                key = simulation_key(item.domain, method, trial, str(task.id))
                if key in completed:
                    continue
                if method == "no_skill":
                    skill = None
                elif method == "manual_skill":
                    from tau2.runner.build import build_environment
                    skill = [manual_skill(item.domain, available_tools=_tools(build_environment(item.domain)))]
                else:
                    skill_dir = skill_root / method
                    skill = [load_skill(path) for path in sorted(skill_dir.glob("*.json")) if not path.name.endswith(".validation.json")]
                    skill = [candidate for candidate in skill if candidate is not None and candidate.status == "published"]
                    if not any(candidate.domain == item.domain for candidate in skill):
                        raise RuntimeError(f"published {item.domain} skill required in {skill_dir}")
                row = run_one_tau_task(domain=item.domain, task=task, method=method, trial=trial, config=config, skill=skill)
                _append(rows_path, row)
                rows.append(row)
                completed.add(key)
                print({"key": key, "reward": row["reward"], "error": row["error"]}, flush=True)

    summary = summarize_tau_rows(rows, expected_keys=expected)
    summary.update({
        "dataset": config.dataset, "agent_model": config.agent_model, "user_model": config.user_model,
        "evaluator_model": config.evaluator_model,
        "seed": config.seed, "trials": config.trials, "max_steps": config.max_steps,
        "max_errors": config.max_errors, "timeout": config.timeout,
        "skill_context_budget_chars": config.skill_context_budget_chars,
        "limit_per_domain": args.limit_per_domain,
    })
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print({"rows": str(rows_path), "summary": str(summary_path), "complete": summary["complete"]})


if __name__ == "__main__":
    main()
