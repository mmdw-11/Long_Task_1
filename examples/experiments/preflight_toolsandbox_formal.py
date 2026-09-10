"""Fail-closed acceptance check before the frozen 360-run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.experiments.toolsandbox_skills import METHODS, load_skill_library


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--skill-root", type=Path, required=True)
    parser.add_argument("--expected-formal", type=int, default=30)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--expected-runs", type=int, default=360)
    args = parser.parse_args()
    rows = [json.loads(x) for x in args.dataset.read_text(encoding="utf-8").splitlines() if x.strip()]
    ids = [row["id"] for row in rows]
    families = [row["family"] for row in rows]
    if len(ids) != len(set(ids)) or len(families) != len(set(families)):
        raise SystemExit("FAIL: duplicate task ID or cross-split family leakage")
    formal = [row for row in rows if row["split"] == "formal_test"]
    if len(formal) != args.expected_formal:
        raise SystemExit(f"FAIL: expected {args.expected_formal} formal tasks, got {len(formal)}")
    if len(formal) * len(METHODS) * args.trials != args.expected_runs:
        raise SystemExit("FAIL: formal task/method/trial product does not equal expected runs")
    libraries = {method: len(load_skill_library(args.skill_root, method)) for method in METHODS}
    registry_path = args.skill_root / "registry.json"
    if not registry_path.exists():
        raise SystemExit("FAIL: validation registry missing")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("protocol") != "paired-per-task-non-regression-v2":
        raise SystemExit("FAIL: obsolete validation protocol")
    if not registry.get("published") or libraries["ours_full"] != len(registry["published"]):
        raise SystemExit("FAIL: no active replay-validated automatic skill library")
    formal_families = {row["family"] for row in formal}
    for method in ("ours_no_validation", "ours_full"):
        for skill in load_skill_library(args.skill_root, method):
            if formal_families & set(skill.source_families):
                raise SystemExit(f"FAIL: formal family leaked into {skill.skill_id}")
    report = {
        "status": "PASS", "dataset_sha256": sha256(args.dataset),
        "formal_tasks": len(formal), "trials": args.trials,
        "methods": list(METHODS), "expected_runs": args.expected_runs,
        "formal_groups": dict(Counter(row["primary_group"] for row in formal)),
        "skill_libraries": libraries, "validation_protocol": registry["protocol"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
