"""Run the 100-case, real-LLM low-entropy communication experiment.

Examples:
  python examples/experiments/run_communication_error_cascade.py --generate-dataset
  python examples/experiments/run_communication_error_cascade.py --bootstrap-from-env --limit 10
  python examples/experiments/run_communication_error_cascade.py --bootstrap-from-env
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

from engine.experiments.communication_error_cascade import (  # noqa: E402
    METHODS, SUBSETS, build_dataset, dataset_manifest, load_dataset, run_case,
    save_result, summarize,
)
from engine.experiments.io import write_json, write_jsonl  # noqa: E402
from engine.modules.model_connections import ModelConnectionStore  # noqa: E402


def bootstrap_env_connection(root: Path) -> str:
    """Create an isolated real-model catalog entry; credentials stay in env."""
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        env = PROJECT_ROOT / ".env"
        if env.exists():
            for raw in env.read_text(encoding="utf-8").splitlines():
                if "=" not in raw or raw.lstrip().startswith("#"):
                    continue
                key, value = raw.split("=", 1)
                if key.strip() and key.strip() not in os.environ:
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")
    required = {key: os.environ.get(key, "").strip() for key in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL")}
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise RuntimeError("缺少真实模型配置：" + "、".join(missing))
    store = ModelConnectionStore(root / "runtime" / "models")
    ident = "communication-cascade-env"
    if not store.exists(ident):
        store.create({"id": ident, "name": "Communication cascade experiment model", "provider": "openai-compatible", "model_id": required["OPENAI_MODEL"], "base_url": required["OPENAI_BASE_URL"], "api_key_env": "OPENAI_API_KEY", "enabled": True, "test_status": "succeeded", "capabilities": ["chat"]})
    return ident


async def main() -> None:
    parser = argparse.ArgumentParser(description="Real-LLM error cascade experiment; LocalEcho is rejected.")
    parser.add_argument("--dataset", default="examples/experiments/sample_data/communication_low_entropy_100_v3.jsonl")
    parser.add_argument("--output-dir", default="runs/experiments/communication_low_entropy_v3")
    parser.add_argument("--model-connection", default="", help="Tested ModelConnectionStore ID.")
    parser.add_argument("--models-root", default="", help="Directory containing model connection records.")
    parser.add_argument("--bootstrap-from-env", action="store_true")
    parser.add_argument("--generate-dataset", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Use only first N frozen cases (0 means all 100).")
    parser.add_argument("--subsets", nargs="+", choices=list(SUBSETS), default=[], help="Run selected strata only.")
    parser.add_argument("--per-subset", type=int, default=0, help="Cap selected strata to N cases each (requires --subsets).")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()

    dataset_path = PROJECT_ROOT / args.dataset
    if args.generate_dataset:
        rows = build_dataset(dataset_path)
        print(json.dumps(dataset_manifest(rows), ensure_ascii=False, indent=2))
        return
    if not dataset_path.exists():
        raise SystemExit(f"数据集不存在：{dataset_path}；先运行 --generate-dataset")
    output = PROJECT_ROOT / args.output_dir
    if args.bootstrap_from_env:
        args.model_connection = bootstrap_env_connection(output)
        args.models_root = str(output / "runtime" / "models")
    if not args.model_connection or not args.models_root:
        raise SystemExit("必须同时提供 --model-connection 与 --models-root，或显式使用 --bootstrap-from-env；禁止回退模型。")

    rows = load_dataset(dataset_path)
    if args.subsets:
        selected = []
        for subset in args.subsets:
            candidates = [row for row in rows if row["subset"] == subset]
            selected.extend(candidates[:max(1, args.per_subset)] if args.per_subset else candidates)
        rows = selected
    elif args.limit:
        rows = rows[:max(1, args.limit)]
    manifest = dataset_manifest(load_dataset(dataset_path))
    write_json(output / "manifest.json", {**manifest, "selected_cases": len(rows), "methods": args.methods, "model_connection": args.model_connection})
    results = []
    for case in rows:
        for method in args.methods:
            result = await run_case(case, method=method, model_connection=args.model_connection, models_root=args.models_root)
            save_result(result, output / "cases")
            results.append({key: result[key] for key in ("case_id", "subset", "method", "status", "invalid_reason", "real_llm_nodes", "metrics", "duration_ms")})
            print(json.dumps(results[-1], ensure_ascii=False))
    write_jsonl(output / "rows.jsonl", results)
    report = summarize(results)
    write_json(output / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["invalid_runs"]:
        raise SystemExit("存在 invalid run；它们已保留审计证据但未计入指标，不能报告为真实实验结果。")


if __name__ == "__main__":
    asyncio.run(main())
