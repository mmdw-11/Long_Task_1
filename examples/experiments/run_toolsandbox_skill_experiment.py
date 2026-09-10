"""Run ToolSandbox canary or pilot; formal_test requires explicit --allow-formal."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from engine.experiments.toolsandbox_runtime import (
    load_tasks, repair_incomplete_failure_artifacts, run_task_with_hard_timeout, save_results,
)
from engine.experiments.toolsandbox_skills import METHODS, load_skill_library


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("skill_train", "skill_validation", "canary", "pilot", "formal_test"), required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--skill-root", type=Path, required=True)
    parser.add_argument("--skill-budget-chars", type=int, default=6000)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--task-timeout", type=float, default=300.0)
    parser.add_argument("--allow-formal", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.split == "formal_test" and not args.allow_formal:
        raise SystemExit("formal_test is frozen; pass --allow-formal only for the authorized 360-run")
    load_dotenv(ROOT / ".env")
    model = args.model
    tasks = load_tasks(args.dataset, args.split)
    if args.split == "formal_test":
        if len(tasks) != 30 or tuple(args.methods) != METHODS or args.trials != 3:
            raise SystemExit("formal protocol requires exactly 30 tasks × four ordered methods × 3 trials = 360 runs")
        if args.limit is not None or args.model != "deepseek-v4-flash":
            raise SystemExit("formal protocol forbids --limit and requires deepseek-v4-flash")
        if args.task_timeout != 300.0:
            raise SystemExit("formal protocol requires the fixed 300-second per-task hard timeout")
        for method in METHODS:
            load_skill_library(args.skill_root, method)
        registry = json.loads((args.skill_root / "registry.json").read_text(encoding="utf-8"))
        if registry.get("protocol") != "paired-per-task-non-regression-v2" or not registry.get("published"):
            raise SystemExit("formal protocol requires an active strict validation registry")
    if args.limit is not None:
        tasks = tasks[: args.limit]
    rows_path = args.output_root / "rows.jsonl"
    if args.split == "formal_test" and args.fresh and rows_path.exists():
        raise SystemExit("refusing to overwrite an existing formal run; omit --fresh to resume or choose a new output root")
    existing = [] if args.fresh or not rows_path.exists() else [json.loads(x) for x in rows_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if repair_incomplete_failure_artifacts(existing, args.output_root):
        save_results(existing, args.output_root)
    by_key = {row["key"]: row for row in existing}
    for trial in range(args.trials):
        for method in args.methods:
            for task in tasks:
                key = f'{args.split}/{method}/trial-{trial}/{task["id"]}'
                if key in by_key and not by_key[key].get("error"):
                    continue
                row = run_task_with_hard_timeout(
                    task, method, trial, args.output_root, model,
                    skill_root=args.skill_root, skill_budget_chars=args.skill_budget_chars,
                    timeout_seconds=args.task_timeout,
                )
                by_key[key] = row
                save_results(list(by_key.values()), args.output_root)
                print(json.dumps({"key": key, "reward": row["reward"], "error": row["error"]}, ensure_ascii=False), flush=True)
    print(json.dumps(save_results(list(by_key.values()), args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
