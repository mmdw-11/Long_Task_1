"""Train the learned router from a JSONL routing dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine import (
    RouteDataset,
    benchmark_router,
    render_experiment_report,
    save_experiment_report,
    train_router,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="runs/router_learning/cascade_dataset.jsonl")
    parser.add_argument("--model-out", default="runs/router_learning/router.json")
    parser.add_argument("--report-out", default="runs/router_learning/benchmark.md")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    dataset = RouteDataset.load_jsonl(args.dataset)
    result = train_router(dataset, train_ratio=args.train_ratio)
    model = result["model"]
    model.save(args.model_out)

    rows = benchmark_router(dataset, train_ratio=args.train_ratio)
    report_path = save_experiment_report(rows, args.report_out)
    summary = {
        "dataset": args.dataset,
        "model": str(Path(args.model_out)),
        "report": str(report_path),
        "train_size": result["train_size"],
        "test_size": result["test_size"],
        "metrics": result["metrics"],
        "benchmark": rows,
        "report_text": render_experiment_report(rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
