"""Run the frozen 200-run dynamic-topology matrix with safe resume support."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.experiments.run_dynamic_topology_experiment import bootstrap_env_connection
from engine.experiments.dynamic_topology import dataset_manifest, load_dynamic_topology_dataset, run_dynamic_topology_case, save_dynamic_topology_run
from engine.experiments.io import read_jsonl, write_json, write_jsonl


def matrix(rows):
    jobs = []
    for repeat in (1, 2):
        for case in rows:
            jobs.extend([(case, "fixed_sparse", repeat), (case, "ours", repeat)])
        for case in rows:
            if case["family"] in {"handoff", "fallback"}:
                jobs.append((case, "dynamic_selection_only", repeat))
    assert len(jobs) == 200
    return jobs


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="examples/experiments/sample_data/dynamic_topology_tasks.jsonl")
    parser.add_argument("--output-dir", default="runs/experiments/dynamic_topology/formal-v1")
    parser.add_argument("--bootstrap-from-env", action="store_true", help="Create an isolated DeepSeek-compatible connection from .env.")
    parser.add_argument("--max-runs", type=int, default=200, help="Run only this many pending jobs; use for staged execution.")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing rows; do not use after a valid run has started.")
    args = parser.parse_args()
    if not args.bootstrap_from_env:
        parser.error("正式 runner 仅接受 --bootstrap-from-env，以保证不回退为本地模型")
    output = Path(args.output_dir)
    model_connection = bootstrap_env_connection(output)
    rows = load_dynamic_topology_dataset(args.dataset)
    manifest = dataset_manifest(rows)
    write_json(output / "manifest.json", {**manifest, "model_connection": model_connection, "matrix": "B1/Ours all cases twice; A1 handoff+fallback twice"})
    existing = {} if args.fresh else {str(item.get("job_id")): item for item in read_jsonl(output / "rows.jsonl")} if (output / "rows.jsonl").exists() else {}
    pending = [(case, method, repeat) for case, method, repeat in matrix(rows) if f"{case['id']}::{method}::r{repeat}" not in existing]
    if args.max_runs >= 0:
        pending = pending[:args.max_runs]
    all_rows = list(existing.values())
    for case, method, repeat in pending:
        job_id = f"{case['id']}::{method}::r{repeat}"
        try:
            result = await run_dynamic_topology_case(case, method=method, model_connection=model_connection, root=output / "runtime")
            save_dynamic_topology_run(result, output / "cases" / f"r{repeat}")
            row = {"job_id": job_id, "repeat": repeat, **{key: result[key] for key in ("id", "family", "method", "passed", "metrics")}}
        except Exception as exc:  # Preserve failure as data; never silently retry/replace it.
            row = {"job_id": job_id, "repeat": repeat, "id": case["id"], "family": case["family"], "method": method, "passed": False, "metrics": {"real_llm": 0, "real_tool": 0, "trace_complete": 0}, "error": f"{type(exc).__name__}: {exc}"}
        all_rows.append(row)
        write_jsonl(output / "rows.jsonl", all_rows)
        print(json.dumps(row, ensure_ascii=False))
    grouped = defaultdict(list)
    for row in all_rows:
        grouped[row["method"]].append(row)
    summary = {"total": len(all_rows), "pending": 200 - len(all_rows), "methods": {method: {"n": len(items), "passed": sum(bool(item["passed"]) for item in items), "pass_rate": round(sum(bool(item["passed"]) for item in items) / len(items), 4) if items else 0.0, "real_llm_rate": round(sum(int(item.get("metrics", {}).get("real_llm", 0)) for item in items) / len(items), 4) if items else 0.0, "real_tool_rate": round(sum(int(item.get("metrics", {}).get("real_tool", 0)) for item in items) / len(items), 4) if items else 0.0} for method, items in grouped.items()}}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
