"""Run paired AUTO versus all-cloud routing experiments."""

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
    parser.add_argument("--auto-arm", choices=("strict_auto", "quality_auto"), default="strict_auto")
    parser.add_argument("--validated-local-tier", action="append", choices=("device", "edge"), default=[],
                        help="Only admit tiers calibrated on data independent of this test set.")
    parser.add_argument("--accuracy-margin-pp", type=float, default=5.0)
    args = parser.parse_args()
    if args.validated_local_tier and args.auto_arm != "quality_auto":
        parser.error("--validated-local-tier requires --auto-arm quality_auto")
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
    runner = RoutingExperimentRunner(
        connections, prices, auto_arm=args.auto_arm,
        validated_local_tiers=tuple(args.validated_local_tier), accuracy_margin_pp=args.accuracy_margin_pp,
    )
    summary = runner.run(load_routing_cases(args.dataset), args.output_dir, manifest={
        "dataset": str(Path(args.dataset).resolve()), "pricing": str(Path(args.pricing).resolve()),
        "mode": f"{args.auto_arm}_vs_all_cloud",
        "strict_route": args.auto_arm == "strict_auto", "env_routing": args.env_routing,
        "placement_mode": "constrained" if args.auto_arm == "quality_auto" else os.environ.get("PLACEMENT_MODE", "auto"),
        "trained_policy_configured": bool(os.environ.get("PLACEMENT_POLICY_PATH")),
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
