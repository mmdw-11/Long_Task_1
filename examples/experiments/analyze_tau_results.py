"""Independently recompute τ summaries and paired integrity checks from rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.experiments.tau_data import load_tau_tasks
from engine.experiments.tau_analysis import task_profile
from engine.experiments.tau_runtime import TAU_METHODS, simulation_key, summarize_tau_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="runs/experiments/tau_skill_v1/rows.jsonl")
    parser.add_argument("--dataset", default="data/processed/tau_skill_v1/tasks.jsonl")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--limit-per-domain", type=int)
    parser.add_argument("--output", default="runs/experiments/tau_skill_v1/recomputed_summary.json")
    args = parser.parse_args()
    with Path(args.rows).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    tasks = sorted(load_tau_tasks(args.dataset, split="test"), key=lambda item: (item.domain, item.id))
    if args.limit_per_domain is not None:
        counts: dict[str, int] = {}
        limited = []
        for task in tasks:
            counts.setdefault(task.domain, 0)
            if counts[task.domain] < args.limit_per_domain:
                limited.append(task)
                counts[task.domain] += 1
        tasks = limited
    expected = {
        simulation_key(task.domain, method, trial, task.metadata["official_task_id"])
        for task in tasks for method in TAU_METHODS for trial in range(args.trials)
    }
    summary = summarize_tau_rows(rows, expected_keys=expected)
    hashes: dict[str, set[str]] = {}
    for row in rows:
        hashes.setdefault(f"{row['domain']}:{row['task_id']}:{row['trial']}", set()).add(str(row.get("initial_state_hash")))
    summary["initial_state_hash_mismatches"] = sorted(key for key, values in hashes.items() if len(values) != 1)
    summary["acceptance_ready"] = summary["acceptance_ready"] and not summary["initial_state_hash_mismatches"]
    baselines = {method: details["task_success"] for method, details in summary["methods"].items()}
    if "ours_full" in baselines:
        summary["ours_full_absolute_delta"] = {
            method: baselines["ours_full"] - value for method, value in baselines.items() if method != "ours_full"
        }
    profiles = {
        (task.domain, str(task.metadata["official_task_id"])): task_profile(task)
        for task in tasks
    }
    stratified: dict[str, dict[str, dict[str, float | int]]] = {}
    strata = {
        "difficulty": lambda profile: profile["difficulty"],
        "multi_goal": lambda profile: "yes" if profile["multi_goal"] else "no",
        "confirmation": lambda profile: "required" if profile["requires_confirmation"] else "not_required",
        "refusal_or_handoff": lambda profile: "yes" if profile["refusal_or_handoff"] else "no",
        "trajectory_length": lambda profile: (
            "short" if profile["action_depth"] <= 2 else "medium" if profile["action_depth"] <= 5 else "long"
        ),
    }
    for method in TAU_METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        stratified[method] = {}
        for name, labeler in strata.items():
            buckets: dict[str, list[float]] = {}
            for row in method_rows:
                profile = profiles.get((row["domain"], str(row["task_id"])))
                if profile is None:
                    continue
                buckets.setdefault(str(labeler(profile)), []).append(float(row["reward"]))
            stratified[method][name] = {
                label: {"runs": len(values), "task_success": sum(values) / len(values)}
                for label, values in sorted(buckets.items())
            }
    summary["stratified_success"] = stratified
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print({"output": str(output), "rows": len(rows), "acceptance_ready": summary["acceptance_ready"]})


if __name__ == "__main__":
    main()
