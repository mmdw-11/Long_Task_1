"""Build a formal experiment table across routing methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from engine import (
    BinaryTextRouterModel,
    HeuristicTaskGate,
    LearnedTaskGate,
    RouteDataset,
    benchmark_router,
    evaluate_gate,
)
from engine.modules.scheduling.advanced_training import (
    benchmark_transformer_router,
    render_extended_experiment_report,
    save_extended_experiment_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="runs/router_learning/cascade_dataset.jsonl")
    parser.add_argument("--nb-model", default="runs/router_learning/router.json")
    parser.add_argument("--transformer-model-dir", default="")
    parser.add_argument("--output", default="runs/router_learning/experiment_table.md")
    args = parser.parse_args()

    dataset = RouteDataset.load_jsonl(args.dataset)
    rows: List[Dict[str, Any]] = []

    rows.append(
        {
            "method": "keyword_rule",
            **evaluate_gate(HeuristicTaskGate(), dataset.examples),
            "latency_ms": 0.1,
            "cost": 0.0,
            "notes": "heuristic baseline",
        }
    )

    if Path(args.nb_model).exists():
        model = BinaryTextRouterModel.load(args.nb_model)
        rows.append(
            {
                "method": "nb_router",
                **evaluate_gate(LearnedTaskGate(model, threshold=0.35), dataset.examples),
                "latency_ms": 0.5,
                "cost": 0.0,
                "notes": "mixed n-gram naive bayes",
            }
        )

    if args.transformer_model_dir and Path(args.transformer_model_dir).exists():
        rows.append(
            benchmark_transformer_router(
                dataset,
                model_dir=args.transformer_model_dir,
                notes="transformer fine-tuned router",
            )
        )

    report_path = save_extended_experiment_report(rows, args.output)
    print(
        json.dumps(
            {
                "output": str(report_path),
                "rows": rows,
                "report_text": render_extended_experiment_report(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
