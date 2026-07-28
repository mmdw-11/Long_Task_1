"""Train a BGE-M3 embedding router with a local sklearn classifier."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from engine import RouteDataset, train_bge_mlp_router, train_bge_router


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="runs/router_learning/expanded_dataset.jsonl")
    parser.add_argument("--model-name", default="D:\\PythonProject\\hf_cache\\models\\BAAI--bge-m3")
    parser.add_argument("--output-dir", default="runs/router_learning/bge_m3")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--hf-home", default="D:\\PythonProject\\hf_cache")
    parser.add_argument("--classifier", choices=["logreg", "mlp"], default="logreg")
    parser.add_argument("--mlp-hidden-sizes", default="512,128")
    parser.add_argument("--mlp-max-iter", type=int, default=300)
    parser.add_argument("--mlp-alpha", type=float, default=0.0001)
    parser.add_argument("--mlp-learning-rate", type=float, default=0.001)
    parser.add_argument("--mlp-random-state", type=int, default=42)
    parser.add_argument("--no-mlp-early-stopping", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(args.hf_home) / "transformers"))

    dataset = RouteDataset.load_jsonl(args.dataset)
    if args.classifier == "mlp":
        hidden_sizes = tuple(int(item.strip()) for item in args.mlp_hidden_sizes.split(",") if item.strip())
        result = train_bge_mlp_router(
            dataset,
            model_name=args.model_name,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            hidden_layer_sizes=hidden_sizes,
            max_iter=args.mlp_max_iter,
            alpha=args.mlp_alpha,
            learning_rate_init=args.mlp_learning_rate,
            random_state=args.mlp_random_state,
            early_stopping=not args.no_mlp_early_stopping,
        )
    else:
        result = train_bge_router(
            dataset,
            model_name=args.model_name,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
        )
    print(
        json.dumps(
            {
                "method": result.method,
                "metrics": result.metrics,
                "artifacts": result.artifacts,
                "seconds": result.seconds,
                "notes": result.notes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
