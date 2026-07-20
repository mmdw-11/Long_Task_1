"""Advanced router training with transformer/BGE-M3/LoRA backends."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from engine import RouteDataset
from engine.modules.scheduling.advanced_training import (
    ModelDownloadUnavailableError,
    train_transformer_router,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="runs/router_learning/cascade_dataset.jsonl")
    parser.add_argument("--model-name", default="bert-base-chinese")
    parser.add_argument("--output-dir", default="runs/router_learning/advanced")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--hf-home", default="D:\\PythonProject\\hf_cache")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(args.hf_home) / "transformers"))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "3")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")

    dataset = RouteDataset.load_jsonl(args.dataset)
    try:
        result = train_transformer_router(
            dataset,
            model_name=args.model_name,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            use_lora=args.lora,
            local_files_only=args.local_files_only,
        )
        payload = {
            "ok": True,
            "method": result.method,
            "metrics": result.metrics,
            "artifacts": result.artifacts,
            "seconds": result.seconds,
            "notes": result.notes,
        }
    except ModelDownloadUnavailableError as exc:
        payload = {
            "ok": False,
            "error": str(exc),
            "hint": (
                "Download the model to D:\\PythonProject\\hf_cache first, "
                "or rerun when Hugging Face access is available."
            ),
            "model_name": args.model_name,
            "hf_home": args.hf_home,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
