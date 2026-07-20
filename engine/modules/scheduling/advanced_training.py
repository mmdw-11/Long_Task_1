"""Optional advanced training backends for learned routing."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .learning import RouteDataset, RouteExample


class ModelDownloadUnavailableError(RuntimeError):
    """Raised when advanced training cannot fetch a remote model."""


@dataclass
class AdvancedTrainingResult:
    method: str
    metrics: Dict[str, float]
    artifacts: Dict[str, str]
    seconds: float
    notes: str = ""


class TransformerTextRouter:
    """Inference wrapper around a Hugging Face text-classification model."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = str(model_dir)
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        from transformers import pipeline

        self._pipeline = pipeline(
            "text-classification",
            model=self.model_dir,
            tokenizer=self.model_dir,
            truncation=True,
        )
        return self._pipeline

    def predict(self, text: str) -> int:
        pipe = self._load()
        output = pipe(text, top_k=1)[0]
        label = str(output["label"]).lower()
        if label.endswith("1") or label == "large":
            return 1
        return 0

    def evaluate(self, dataset: Sequence[RouteExample]) -> Dict[str, float]:
        return _evaluate_predictions([self.predict(ex.text) for ex in dataset], [int(ex.label) for ex in dataset])


def train_transformer_router(
    dataset: RouteDataset,
    *,
    model_name: str,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    num_train_epochs: int = 1,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    use_lora: bool = False,
    local_files_only: bool = False,
) -> AdvancedTrainingResult:
    import numpy as np
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    train_set, test_set = dataset.split(train_ratio=train_ratio)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise ModelDownloadUnavailableError(
            f"Unable to load transformer model '{model_name}'. "
            "Check network access to Hugging Face or pre-download the model into the local cache."
        ) from exc
    method = "transformer_lora" if use_lora else "transformer_full"

    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, config)

    def _to_hf(items: Sequence[RouteExample]) -> Dataset:
        return Dataset.from_dict(
            {
                "text": [item.text for item in items],
                "label": [int(item.label) for item in items],
            }
        )

    train_ds = _to_hf(train_set.examples)
    test_ds = _to_hf(test_set.examples)

    def tokenize(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return _evaluate_predictions(list(preds), list(labels))

    args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        report_to=[],
        load_best_model_at_end=False,
    )

    start = time.time()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    save_target = out_dir / "model"
    trainer.save_model(str(save_target))
    tokenizer.save_pretrained(str(save_target))
    seconds = time.time() - start

    clean_metrics = {
        "accuracy": float(metrics.get("eval_accuracy", 0.0)),
        "precision": float(metrics.get("eval_precision", 0.0)),
        "recall": float(metrics.get("eval_recall", 0.0)),
        "f1": float(metrics.get("eval_f1", 0.0)),
    }
    summary = {
        "method": method,
        "model_name": model_name,
        "use_lora": use_lora,
        "train_size": len(train_set.examples),
        "test_size": len(test_set.examples),
        "seconds": seconds,
        "metrics": clean_metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return AdvancedTrainingResult(
        method=method,
        metrics=clean_metrics,
        artifacts={"model_dir": str(save_target), "summary": str(out_dir / "summary.json")},
        seconds=seconds,
        notes=model_name,
    )


def render_extended_experiment_report(rows: Sequence[Dict[str, Any]]) -> str:
    headers = ["method", "accuracy", "precision", "recall", "f1", "latency_ms", "cost", "notes"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("method", "")),
                    _fmt(row.get("accuracy")),
                    _fmt(row.get("precision")),
                    _fmt(row.get("recall")),
                    _fmt(row.get("f1")),
                    _fmt(row.get("latency_ms")),
                    _fmt(row.get("cost")),
                    str(row.get("notes", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def save_extended_experiment_report(rows: Sequence[Dict[str, Any]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_extended_experiment_report(rows), encoding="utf-8")
    return out


def benchmark_transformer_router(
    dataset: RouteDataset,
    *,
    model_dir: str | Path,
    notes: str = "",
) -> Dict[str, Any]:
    router = TransformerTextRouter(model_dir)
    start = time.time()
    metrics = router.evaluate(dataset.examples)
    elapsed = time.time() - start
    per_item_ms = (elapsed / max(1, len(dataset.examples))) * 1000.0
    return {
        "method": "transformer_router",
        **metrics,
        "latency_ms": per_item_ms,
        "cost": 0.0,
        "notes": notes or str(model_dir),
    }


def _evaluate_predictions(preds: Sequence[int], labels: Sequence[int]) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for pred, label in zip(preds, labels):
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4f}"
        return "-"
    return str(value)
