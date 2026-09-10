"""Normalize known agent failures and independently rebuild a ToolSandbox summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.experiments.toolsandbox_runtime import repair_incomplete_failure_artifacts, save_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-runs", type=int)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    path = args.root / "rows.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        error = row.get("error") or ""
        if error.startswith('KeyError: "Agent tool call'):
            row["diagnostic"] = error
            row["failure_reason"] = "invalid_or_unauthorized_tool"
            row["error"] = None
    repaired = repair_incomplete_failure_artifacts(rows, args.root)
    summary = save_results(rows, args.root)
    task_trials = {}
    for row in rows:
        task_trials.setdefault((row["task_id"], row["trial"]), set()).add(row["initial_state_hash"])
    audit = {
        "rows": len(rows), "unique_keys": len({row["key"] for row in rows}),
        "errors": sum(bool(row.get("error")) for row in rows),
        "state_hash_mismatches": sum(len(values) != 1 for values in task_trials.values()),
        "missing_final_state_hash": sum(not row.get("final_state_hash") for row in rows),
        "missing_trajectory_hash": sum(not row.get("trajectory_sha256") for row in rows),
        "summary": summary,
        "repaired_failure_artifacts": repaired,
    }
    if args.expected_runs is not None:
        audit["expected_runs"] = args.expected_runs
        audit["complete"] = len(rows) == args.expected_runs and audit["unique_keys"] == args.expected_runs
    if args.strict and (
        not audit.get("complete", True) or audit["errors"] or audit["state_hash_mismatches"]
        or audit["missing_final_state_hash"] or audit["missing_trajectory_hash"]
    ):
        (args.root / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit("strict result audit FAILED; inspect audit.json")
    audit["status"] = "PASS"
    (args.root / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
