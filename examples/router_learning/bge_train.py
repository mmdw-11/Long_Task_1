"""Train a BGE-M3 embedding router with a local sklearn classifier."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from engine import RouteDataset, train_bge_router


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="runs/router_learning/expanded_dataset.jsonl")
    parser.add_argument("--model-name", default="D:\\PythonProject\\hf_cache\\models\\BAAI--bge-m3")
    parser.add_argument("--output-dir", default="runs/router_learning/bge_m3")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--hf-home", default="D:\\PythonProject\\hf_cache")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(args.hf_home) / "transformers"))

    dataset = RouteDataset.load_jsonl(args.dataset)
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
