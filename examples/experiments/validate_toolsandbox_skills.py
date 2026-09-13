"""Paired state replay validation and gated publication of induced skills."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from engine.experiments.toolsandbox_runtime import load_tasks, run_task, save_results
from engine.experiments.toolsandbox_skills import BGEPolicyRetriever, parse_skill, retrieve_skills


def mean(rows, field):
    return sum(float(row.get(field) or 0) for row in rows) / max(1, len(rows))


def invalid_count(rows):
    return sum(row.get("failure_reason") == "invalid_or_unauthorized_tool" for row in rows)


def paired_non_regression(baseline, candidate, reward_tolerance):
    base = {row["task_id"]: row for row in baseline}
    cand = {row["task_id"]: row for row in candidate}
    details = []
    for task_id in sorted(base):
        b, c = base[task_id], cand[task_id]
        applied = bool(c.get("skill_injections"))
        details.append({
            "task_id": task_id,
            "baseline_reward": b["reward"], "candidate_reward": c["reward"],
            "baseline_minefield": b["minefield_similarity"],
            "candidate_minefield": c["minefield_similarity"],
            "skill_applied": applied,
            # A candidate that was not actually injected supplies no evidence
            # of non-regression.  Never substitute the baseline score here.
            "reward_delta": c["reward"] - b["reward"] if applied else None,
            "reward_non_regression": applied and c["reward"] + reward_tolerance >= b["reward"],
            "minefield_non_regression": applied and c["minefield_similarity"] <= b["minefield_similarity"] + 1e-12,
        })
    return details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--skill-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--skill-budget-chars", type=int, default=6000)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--reward-tolerance", type=float, default=0.01)
    parser.add_argument("--candidates", nargs="*")
    parser.add_argument("--merge-registry", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    model = args.model
    tasks = load_tasks(args.dataset, "skill_validation")
    baseline = [run_task(task, "no_skill", 0, args.output_root / "baseline", model,
                         skill_root=args.skill_root, skill_budget_chars=args.skill_budget_chars) for task in tasks]
    save_results(baseline, args.output_root / "baseline")
    published = args.skill_root / "ours_full"; rejected = args.skill_root / "rejected"
    published.mkdir(parents=True, exist_ok=True); rejected.mkdir(parents=True, exist_ok=True)
    # Do not let artifacts from an older validation protocol masquerade as
    # current publications. Preserve them in-place, but only this run's
    # manifest defines the active published set.
    previous = {}
    if args.merge_registry and (args.skill_root / "registry.json").exists():
        previous = json.loads((args.skill_root / "registry.json").read_text(encoding="utf-8"))
    active_publications = list(previous.get("published", []))
    reports = [item for item in previous.get("reports", []) if item.get("skill_id") not in set(args.candidates or [])]
    paths = sorted((args.skill_root / "ours_full_candidates").glob("*.json"))
    if args.candidates:
        paths = [path for path in paths if path.stem in set(args.candidates)]
    for path in paths:
        skill = parse_skill(json.loads(path.read_text(encoding="utf-8")))
        retriever = BGEPolicyRetriever([skill])
        applicable_tasks = [
            task for task in tasks
            if retrieve_skills([skill], task["id"].replace("_", " "), max_chars=args.skill_budget_chars,
                               bge_retriever=retriever)[1][0]["accepted"]
        ]
        candidate_rows = [run_task(task, "ours_full", 0, args.output_root / skill.skill_id, model,
                                   skill_root=args.skill_root, skill_budget_chars=args.skill_budget_chars,
                                   skills_override=[skill]) for task in applicable_tasks]
        save_results(candidate_rows, args.output_root / skill.skill_id)
        report = {
            "skill_id": skill.skill_id, "validation_task_ids": [t["id"] for t in applicable_tasks],
            "baseline_mean_reward": mean(baseline, "reward"), "candidate_mean_reward": mean(candidate_rows, "reward"),
            "baseline_minefield": mean(baseline, "minefield_similarity"), "candidate_minefield": mean(candidate_rows, "minefield_similarity"),
            "baseline_invalid_tools": invalid_count(baseline), "candidate_invalid_tools": invalid_count(candidate_rows),
            "baseline_errors": sum(bool(r.get("error")) for r in baseline),
            "candidate_errors": sum(bool(r.get("error")) for r in candidate_rows),
        }
        report["paired"] = paired_non_regression(
            [row for row in baseline if row["task_id"] in {item["task_id"] for item in candidate_rows}],
            candidate_rows, args.reward_tolerance,
        )
        report["reward_non_regression_tolerance"] = args.reward_tolerance
        applied_rows = [row for row in candidate_rows if row.get("skill_injections")]
        report["validation_covered_tasks"] = len(applied_rows)
        report["validation_rejected_tasks"] = len(candidate_rows) - len(applied_rows)
        report["fixed_pairing_seeds"] = all(
            next(b for b in baseline if b["task_id"] == c["task_id"])["agent_seed"] == c["agent_seed"]
            and next(b for b in baseline if b["task_id"] == c["task_id"])["user_seed"] == c["user_seed"]
            for c in candidate_rows
        )
        report["accepted"] = (
            report["validation_covered_tasks"] > 0
            and all(item["reward_non_regression"] and item["minefield_non_regression"] for item in report["paired"])
            and invalid_count(applied_rows) == 0
            and sum(bool(r.get("error")) for r in applied_rows) == 0
            and report["fixed_pairing_seeds"]
        )
        report_path = args.output_root / f"{skill.skill_id}_validation.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        skill.validation_report = str(report_path)
        skill.status = "published" if report["accepted"] else "rejected"
        skill.metadata["validation"] = report
        target = published if report["accepted"] else rejected
        (target / path.name).write_text(json.dumps(skill.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        if report["accepted"]:
            if path.name not in active_publications:
                active_publications.append(path.name)
        elif path.name in active_publications:
            active_publications.remove(path.name)
        reports.append(report); print(report, flush=True)
    registry = {
        "protocol": "paired-per-task-non-regression-v2",
        "model": model,
        "skill_budget_chars": args.skill_budget_chars,
        "published": active_publications,
        "rejected": [f"{item['skill_id']}.json" for item in reports if not item["accepted"]],
        "reports": reports,
        "skill_versions": {
            path.stem: parse_skill(json.loads(path.read_text(encoding="utf-8"))).version
            for path in (args.skill_root / "ours_full_candidates").glob("*.json")
            if path.name in active_publications
        },
    }
    (args.output_root / "validation_summary.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    history = args.skill_root / "registry_history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = json.dumps(registry, ensure_ascii=False, indent=2)
    (history / f"registry_{stamp}.json").write_text(payload, encoding="utf-8")
    (args.skill_root / "registry.json").write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
