"""Run the strict BGE-M3 AUTO versus all-cloud routing experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from engine.experiments.routing import EnvironmentRoutingStore, Pricing, RoutingExperimentRunner, load_routing_cases
from engine.modules.model_connections import ModelConnectionStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Frozen RouteCostBench JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-root", default="runs/model_connections")
    parser.add_argument("--env-routing", action="store_true", help="Use device/edge/cloud models and endpoints from .env")
    parser.add_argument("--pricing", required=True, help="JSON: {model_id: {input_per_million_cny, output_per_million_cny}}")
    args = parser.parse_args()
    prices = {
        name: Pricing(**value)
        for name, value in json.loads(Path(args.pricing).read_text(encoding="utf-8")).items()
    }
    if args.env_routing:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        if not (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            raise RuntimeError("DEEPSEEK_API_KEY or OPENAI_API_KEY is required for cloud routing")
        connections = EnvironmentRoutingStore.from_dotenv()
    else:
        connections = ModelConnectionStore(args.model_root)
    runner = RoutingExperimentRunner(connections, prices)
    summary = runner.run(load_routing_cases(args.dataset), args.output_dir, manifest={
        "dataset": str(Path(args.dataset).resolve()), "pricing": str(Path(args.pricing).resolve()),
        "mode": "env_routing_strict_auto_vs_all_cloud" if args.env_routing else "strict_auto_vs_all_cloud",
        "strict_route": True, "env_routing": args.env_routing,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
