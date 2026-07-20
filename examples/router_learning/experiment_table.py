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
    benchmark_embedding_router,
    benchmark_transformer_router,
    render_extended_experiment_report,
    save_extended_experiment_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="runs/router_learning/cascade_dataset.jsonl")
    parser.add_argument("--nb-model", default="runs/router_learning/router.json")
    parser.add_argument("--transformer-model-dir", default="")
    parser.add_argument("--transformer-summary", default="")
    parser.add_argument("--lora-model-dir", default="")
    parser.add_argument("--lora-summary", default="")
    parser.add_argument("--bge-artifact-dir", default="")
    parser.add_argument("--bge-summary", default="")
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
        row = benchmark_transformer_router(
            dataset,
            model_dir=args.transformer_model_dir,
            notes="transformer fine-tuned router",
        )
        row["method"] = "bert_full"
        _merge_summary(row, args.transformer_summary)
        rows.append(row)

    if args.lora_model_dir and Path(args.lora_model_dir).exists():
        row = benchmark_transformer_router(
            dataset,
            model_dir=args.lora_model_dir,
            notes="transformer lora router",
        )
        row["method"] = "bert_lora"
        _merge_summary(row, args.lora_summary)
        rows.append(row)

    if args.bge_artifact_dir and Path(args.bge_artifact_dir).exists():
        row = benchmark_embedding_router(
            dataset,
            artifact_dir=args.bge_artifact_dir,
            notes="bge-m3 embedding router",
        )
        row["method"] = "bge_m3"
        _merge_summary(row, args.bge_summary)
        rows.append(row)

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


def _merge_summary(row: Dict[str, Any], summary_path: str) -> None:
    if not summary_path:
        return
    path = Path(summary_path)
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    notes = row.get("notes", "")
    model_name = data.get("model_name")
    train_seconds = data.get("seconds")
    suffix = []
    if model_name:
        suffix.append(str(model_name))
    if train_seconds is not None:
        suffix.append(f"train_s={float(train_seconds):.2f}")
    if suffix:
        row["notes"] = notes + " | " + " | ".join(suffix) if notes else " | ".join(suffix)


if __name__ == "__main__":
    main()
