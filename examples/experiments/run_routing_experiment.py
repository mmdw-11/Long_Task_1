"""Run the minimum device/cloud routing comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import PseudoCascadeTeacher, build_route_dataset, default_training_texts
from engine.experiments import RoutingExperimentConfig, run_routing_experiment, save_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/experiments/routing")
    parser.add_argument("--device-latency-ms", type=float, default=40.0)
    parser.add_argument("--cloud-latency-ms", type=float, default=800.0)
    parser.add_argument("--device-cost", type=float, default=0.001)
    parser.add_argument("--cloud-cost", type=float, default=0.02)
    args = parser.parse_args()
    # Formal offline controls must not silently make cloud labeling calls.
    dataset = build_route_dataset(default_training_texts(), teacher=PseudoCascadeTeacher())
    reports = run_routing_experiment(
        dataset,
        RoutingExperimentConfig(
            device_latency_ms=args.device_latency_ms,
            cloud_latency_ms=args.cloud_latency_ms,
            device_cost=args.device_cost,
            cloud_cost=args.cloud_cost,
        ),
    )
    for method, report in reports.items():
        print(method, save_report(report, Path(args.output_dir) / method))


if __name__ == "__main__":
    main()
