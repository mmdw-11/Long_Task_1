"""Benchmark rule-based vs learned routing."""

from __future__ import annotations

import argparse
from pathlib import Path

from engine import (
    PseudoCascadeTeacher,
    benchmark_router,
    build_route_dataset,
    default_training_texts,
    render_experiment_report,
    save_experiment_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/router_learning/benchmark.md")
    args = parser.parse_args()

    dataset = build_route_dataset(default_training_texts(), teacher=PseudoCascadeTeacher())
    rows = benchmark_router(dataset)
    report = render_experiment_report(rows)
    save_experiment_report(rows, Path(args.output))
    print(report)


if __name__ == "__main__":
    main()
