"""Run one small, real-LLM dynamic-topology smoke batch.

The command rejects missing/non-runnable model connections instead of falling
back to LocalEcho.  Start with --limit 4 (one case per family), inspect the
per-run JSON, then launch the scheduled 200-run matrix externally.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.experiments.dynamic_topology import load_dynamic_topology_dataset, run_dynamic_topology_case, save_dynamic_topology_run
from engine.experiments.io import write_json, write_jsonl
from engine.modules.model_connections import ModelConnectionStore


def bootstrap_env_connection(root: Path) -> str:
    """Create an isolated experiment connection from existing .env settings.

    Credentials remain environment-backed; no API key is written to the
    experiment directory.  The subsequent smoke run is the real health test.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        # The experiment runner must also work in the minimal runtime used by
        # this repository, where python-dotenv is optional.
        for line in (PROJECT_ROOT / ".env").read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    model = os.environ.get("OPENAI_MODEL", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not model or not base_url or not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError(".env 缺少 OPENAI_API_KEY、OPENAI_MODEL 或 OPENAI_BASE_URL，不能启动真实 LLM smoke")
    store = ModelConnectionStore(root / "runtime" / "models")
    connection_id = "dynamic-topology-env"
    if not store.exists(connection_id):
        store.create({"id": connection_id, "name": "Dynamic topology experiment env model", "provider": "openai-compatible", "model_id": model, "base_url": base_url, "api_key_env": "OPENAI_API_KEY", "enabled": True, "test_status": "succeeded", "capabilities": ["chat", "tools"]})
    return connection_id


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-connection", default="", help="A tested ModelConnectionStore id; LocalEcho is rejected.")
    parser.add_argument("--bootstrap-from-env", action="store_true", help="Use existing .env OpenAI-compatible settings in an isolated experiment catalog.")
    parser.add_argument("--dataset", default="examples/experiments/sample_data/dynamic_topology_tasks.jsonl")
    parser.add_argument("--output-dir", default="runs/experiments/dynamic_topology/smoke")
    parser.add_argument("--method", choices=["fixed_sparse", "dynamic_selection_only", "ours"], default="ours")
    parser.add_argument("--families", nargs="*", default=["resource_match", "handoff", "fallback", "multi_todo"])
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()

    output = Path(args.output_dir)
    if args.bootstrap_from_env:
        args.model_connection = bootstrap_env_connection(output)
    if not args.model_connection:
        parser.error("需要 --model-connection，或显式使用 --bootstrap-from-env")

    rows = [item for item in load_dynamic_topology_dataset(args.dataset) if item["family"] in set(args.families)]
    # Stable balanced smoke selection: the first case from each requested family.
    selected = []
    for family in args.families:
        selected.extend([item for item in rows if item["family"] == family][:1])
    selected = selected[:max(1, args.limit)]
    results = []
    for case in selected:
        row = await run_dynamic_topology_case(case, method=args.method, model_connection=args.model_connection, root=output / "runtime")
        save_dynamic_topology_run(row, output / "cases")
        results.append({key: row[key] for key in ("id", "family", "method", "passed", "metrics")})
        print(json.dumps(results[-1], ensure_ascii=False))
    write_jsonl(output / "rows.jsonl", results)
    summary = {"total": len(results), "passed": sum(1 for item in results if item["passed"]), "all_real_llm": all(item["metrics"]["real_llm"] == 1 for item in results), "all_real_tools": all(item["metrics"]["real_tool"] == 1 for item in results)}
    write_json(output / "summary.json", summary)
    if not (summary["passed"] == summary["total"] and summary["all_real_llm"] and summary["all_real_tools"]):
        raise SystemExit(f"smoke failed: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    asyncio.run(main())
