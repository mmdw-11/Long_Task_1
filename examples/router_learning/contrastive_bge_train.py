"""Contrastively fine-tune BGE-M3, then train an MLP router head."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from engine import RouteDataset, RouteExample, train_bge_mlp_router


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="runs/router_learning/final_tier_labeled_dataset.deduped.jsonl")
    parser.add_argument("--base-model", default="D:\\PythonProject\\hf_cache\\models\\BAAI--bge-m3")
    parser.add_argument("--output-dir", default="runs/router_learning/final_bge_m3_contrastive")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--hf-home", default="D:\\PythonProject\\hf_cache")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-triplets", type=int, default=3000)
    parser.add_argument("--triplets-per-anchor", type=int, default=2)
    parser.add_argument("--triplet-margin", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-contrastive-if-encoder-exists", action="store_true")
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-save-steps", type=int, default=25)
    parser.add_argument("--checkpoint-save-total-limit", type=int, default=3)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--eval-train-limit", type=int, default=600)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--mlp-hidden-sizes", default="512,128")
    parser.add_argument("--mlp-max-iter", type=int, default=300)
    parser.add_argument("--mlp-alpha", type=float, default=0.0001)
    parser.add_argument("--mlp-learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(args.hf_home) / "transformers"))
    os.environ.setdefault("HF_HUB_CACHE", str(Path(args.hf_home) / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(args.hf_home) / "datasets"))
    os.environ.setdefault("TORCH_HOME", str(Path(args.hf_home) / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(args.hf_home) / "xdg"))
    d_tmp = Path("D:\\PythonProject\\long_task_1\\tmp")
    d_tmp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMP", str(d_tmp))
    os.environ.setdefault("TEMP", str(d_tmp))
    os.environ.setdefault("WANDB_DISABLED", "true")

    out_dir = Path(args.output_dir)
    encoder_dir = out_dir / "encoder"
    router_dir = out_dir / "router"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = RouteDataset.load_jsonl(args.dataset)
    train_set, test_set = dataset.split(train_ratio=args.train_ratio, seed=args.seed)
    triplets = build_triplets(
        train_set.examples,
        max_triplets=args.max_triplets,
        triplets_per_anchor=args.triplets_per_anchor,
        seed=args.seed,
    )
    triplet_path = save_triplets_jsonl(triplets, out_dir / "triplets.jsonl")

    start = time.time()
    contrastive_info: Dict[str, object] = {}
    if args.skip_contrastive_if_encoder_exists and (encoder_dir / "modules.json").exists():
        contrastive_seconds = 0.0
        contrastive_info = {"skipped": True}
    else:
        contrastive_info = fine_tune_encoder(
            triplets,
            base_model=args.base_model,
            output_dir=encoder_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            triplet_margin=args.triplet_margin,
            train_examples=train_set.examples,
            validation_examples=test_set.examples,
            early_stop_patience=args.early_stop_patience,
            eval_train_limit=args.eval_train_limit,
            eval_batch_size=args.eval_batch_size,
            resume_from_checkpoint=args.resume_from_checkpoint,
            checkpoint_save_steps=args.checkpoint_save_steps,
            checkpoint_save_total_limit=args.checkpoint_save_total_limit,
            seed=args.seed,
        )
        contrastive_seconds = time.time() - start

    hidden_sizes = tuple(int(item.strip()) for item in args.mlp_hidden_sizes.split(",") if item.strip())
    mlp_result = train_bge_mlp_router(
        dataset,
        model_name=str(encoder_dir),
        output_dir=router_dir,
        train_ratio=args.train_ratio,
        hidden_layer_sizes=hidden_sizes,
        max_iter=args.mlp_max_iter,
        alpha=args.mlp_alpha,
        learning_rate_init=args.mlp_learning_rate,
        random_state=args.seed,
    )

    summary = {
        "method": "bge_m3_contrastive_triplet_mlp",
        "base_model": args.base_model,
        "encoder_dir": str(encoder_dir),
        "router_dir": str(router_dir),
        "dataset": args.dataset,
        "route_label_schema": infer_label_schema(dataset.examples),
        "train_size": len(train_set.examples),
        "test_size": len(test_set.examples),
        "triplet_count": len(triplets),
        "triplets": str(triplet_path),
        "contrastive": {
            "loss": "TripletLoss",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "triplet_margin": args.triplet_margin,
            "early_stop_patience": args.early_stop_patience,
            "eval_train_limit": args.eval_train_limit,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "checkpoint_save_steps": args.checkpoint_save_steps,
            "checkpoint_save_total_limit": args.checkpoint_save_total_limit,
            "validation": contrastive_info,
            "seconds": contrastive_seconds,
        },
        "mlp": {
            "hidden_layer_sizes": list(hidden_sizes),
            "max_iter": args.mlp_max_iter,
            "alpha": args.mlp_alpha,
            "learning_rate_init": args.mlp_learning_rate,
        },
        "metrics": mlp_result.metrics,
        "artifacts": {
            "encoder": str(encoder_dir),
            "classifier": mlp_result.artifacts["classifier"],
            "router_summary": mlp_result.artifacts["summary"],
            "summary": str(out_dir / "summary.json"),
        },
        "seconds": contrastive_seconds + mlp_result.seconds,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_triplets(
    examples: Sequence[RouteExample],
    *,
    max_triplets: int,
    triplets_per_anchor: int,
    seed: int,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    by_label: Dict[int, List[RouteExample]] = defaultdict(list)
    for item in examples:
        by_label[int(item.label)].append(item)

    labels = sorted(by_label)
    if len(labels) < 2:
        raise ValueError("contrastive training needs at least two labels")

    triplets: List[Dict[str, object]] = []
    anchors = list(examples)
    rng.shuffle(anchors)
    for anchor in anchors:
        same = [item for item in by_label[int(anchor.label)] if item is not anchor and item.text != anchor.text]
        if not same:
            continue
        negative_labels = [label for label in labels if label != int(anchor.label) and by_label[label]]
        if not negative_labels:
            continue
        for _ in range(max(1, triplets_per_anchor)):
            pos = rng.choice(same)
            neg_label = rng.choice(negative_labels)
            neg = rng.choice(by_label[neg_label])
            triplets.append(
                {
                    "anchor": anchor.text,
                    "positive": pos.text,
                    "negative": neg.text,
                    "label": int(anchor.label),
                    "negative_label": int(neg.label),
                }
            )
            if max_triplets and len(triplets) >= max_triplets:
                return triplets
    return triplets


def save_triplets_jsonl(triplets: Sequence[Dict[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in triplets:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


def fine_tune_encoder(
    triplets: Sequence[Dict[str, object]],
    *,
    base_model: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    triplet_margin: float,
    train_examples: Sequence[RouteExample],
    validation_examples: Sequence[RouteExample],
    early_stop_patience: int,
    eval_train_limit: int,
    eval_batch_size: int,
    resume_from_checkpoint: bool,
    checkpoint_save_steps: int,
    checkpoint_save_total_limit: int,
    seed: int,
) -> Dict[str, object]:
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    work_dir = output_dir.parent / f"{output_dir.name}_work"
    checkpoint_dir = output_dir.parent / "contrastive_checkpoints"
    state_path = output_dir.parent / "contrastive_state.json"
    state = load_contrastive_state(state_path) if resume_from_checkpoint else {}
    completed_epochs = int(state.get("completed_epochs", 0))
    model_source = str(work_dir) if resume_from_checkpoint and completed_epochs > 0 and (work_dir / "modules.json").exists() else base_model
    model = SentenceTransformer(model_source, local_files_only=True)
    contrastive_examples = [
        InputExample(texts=[str(item["anchor"]), str(item["positive"]), str(item["negative"])])
        for item in triplets
    ]
    dataloader = DataLoader(contrastive_examples, shuffle=True, batch_size=batch_size)
    loss = losses.TripletLoss(model=model, triplet_margin=triplet_margin)
    warmup_steps = int(len(dataloader) * max(1, epochs) * warmup_ratio)
    history: List[Dict[str, object]] = list(state.get("history") or [])
    best_f1 = float(state.get("best_f1", -1.0))
    best_epoch = int(state.get("best_epoch", 0))
    stale_epochs = int(state.get("stale_epochs", 0))
    start_epoch = completed_epochs + 1
    resume_this_fit = resume_from_checkpoint and completed_epochs == 0 and latest_checkpoint(checkpoint_dir) is not None
    for epoch in range(start_epoch, epochs + 1):
        model.fit(
            train_objectives=[(dataloader, loss)],
            epochs=1,
            warmup_steps=warmup_steps if epoch == 1 else 0,
            optimizer_params={"lr": learning_rate},
            output_path=str(work_dir),
            save_best_model=False,
            show_progress_bar=True,
            checkpoint_path=str(checkpoint_dir),
            checkpoint_save_steps=checkpoint_save_steps,
            checkpoint_save_total_limit=checkpoint_save_total_limit,
            resume_from_checkpoint=resume_this_fit,
        )
        resume_this_fit = False
        metrics = evaluate_embedding_centroids(
            model,
            train_examples,
            validation_examples,
            train_limit=eval_train_limit,
            batch_size=eval_batch_size,
            seed=seed,
        )
        row = {"epoch": epoch, **metrics}
        history.append(row)
        print(json.dumps({"contrastive_validation": row}, ensure_ascii=False), flush=True)
        current_f1 = float(metrics["f1"])
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            stale_epochs = 0
            model.save(str(output_dir))
        else:
            stale_epochs += 1
            if stale_epochs >= early_stop_patience:
                save_contrastive_state(
                    state_path,
                    history=history,
                    best_f1=best_f1,
                    best_epoch=best_epoch,
                    completed_epochs=epoch,
                    stale_epochs=stale_epochs,
                    stopped_early=True,
                )
                break
        save_contrastive_state(
            state_path,
            history=history,
            best_f1=best_f1,
            best_epoch=best_epoch,
            completed_epochs=epoch,
            stale_epochs=stale_epochs,
            stopped_early=False,
        )
    return {"best_epoch": best_epoch, "best_f1": best_f1, "history": history}


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    checkpoints = [path for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[-1]) if path.name.rsplit("-", 1)[-1].isdigit() else -1)


def load_contrastive_state(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_contrastive_state(
    path: Path,
    *,
    history: Sequence[Dict[str, object]],
    best_f1: float,
    best_epoch: int,
    completed_epochs: int,
    stale_epochs: int,
    stopped_early: bool,
) -> None:
    payload = {
        "history": list(history),
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "stale_epochs": stale_epochs,
        "stopped_early": stopped_early,
        "updated_at": time.time(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_embedding_centroids(
    model: object,
    train_examples: Sequence[RouteExample],
    validation_examples: Sequence[RouteExample],
    *,
    train_limit: int,
    batch_size: int,
    seed: int,
) -> Dict[str, float]:
    import numpy as np

    rng = random.Random(seed)
    train_items = list(train_examples)
    rng.shuffle(train_items)
    if train_limit:
        train_items = train_items[:train_limit]
    val_items = list(validation_examples)
    train_texts = [item.text for item in train_items]
    train_labels = np.asarray([int(item.label) for item in train_items])
    val_texts = [item.text for item in val_items]
    val_labels = [int(item.label) for item in val_items]

    train_vectors = model.encode(train_texts, normalize_embeddings=True, batch_size=batch_size, convert_to_numpy=True)
    val_vectors = model.encode(val_texts, normalize_embeddings=True, batch_size=batch_size, convert_to_numpy=True)
    labels = sorted(int(label) for label in set(train_labels.tolist()))
    centroids = []
    for label in labels:
        centroid = train_vectors[train_labels == label].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids.append(centroid)
    centroid_matrix = np.vstack(centroids)
    pred_indices = np.argmax(val_vectors @ centroid_matrix.T, axis=1)
    preds = [labels[int(idx)] for idx in pred_indices]
    return evaluate_predictions(preds, val_labels)


def evaluate_predictions(preds: Sequence[int], labels: Sequence[int]) -> Dict[str, float]:
    label_set = sorted(set(int(label) for label in labels) | set(int(pred) for pred in preds))
    total = max(1, len(labels))
    accuracy = sum(1 for pred, label in zip(preds, labels) if int(pred) == int(label)) / total
    precisions = []
    recalls = []
    f1s = []
    for target in label_set:
        tp = sum(1 for pred, label in zip(preds, labels) if int(pred) == target and int(label) == target)
        fp = sum(1 for pred, label in zip(preds, labels) if int(pred) == target and int(label) != target)
        fn = sum(1 for pred, label in zip(preds, labels) if int(pred) != target and int(label) == target)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    classes = max(1, len(label_set))
    return {
        "accuracy": accuracy,
        "precision": sum(precisions) / classes,
        "recall": sum(recalls) / classes,
        "f1": sum(f1s) / classes,
    }


def infer_label_schema(examples: Sequence[RouteExample]) -> str:
    for item in examples:
        schema = str(item.metadata.get("route_label_schema") or "").strip()
        if schema:
            return schema
    return "0=device,1=edge,2=cloud"


if __name__ == "__main__":
    main()
