"""Optional advanced training backends for learned routing."""

from __future__ import annotations

import json
import math
import os
import inspect
import pickle
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
        import json
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )

        model_path = Path(self.model_dir)
        tokenizer_source = self.model_dir
        model_source: Any = self.model_dir

        adapter_config = model_path / "adapter_config.json"
        if adapter_config.exists():
            from peft import PeftModel

            config = json.loads(adapter_config.read_text(encoding="utf-8"))
            base_model_name = str(config.get("base_model_name_or_path") or "").strip()
            if not base_model_name:
                raise RuntimeError("LoRA adapter is missing base_model_name_or_path")
            tokenizer_source = base_model_name
            base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
            model_source = PeftModel.from_pretrained(base_model, self.model_dir)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        self._pipeline = pipeline(
            "text-classification",
            model=model_source,
            tokenizer=tokenizer,
            truncation=True,
        )
        return self._pipeline

    def predict(self, text: str) -> int:
        probs = self.predict_proba(text)
        return 1 if probs[1] >= probs[0] else 0

    def predict_proba(self, text: str) -> Dict[int, float]:
        pipe = self._load()
        output = pipe(text, top_k=2)
        if isinstance(output, list) and output and isinstance(output[0], list):
            output = output[0]
        scores = {0: 0.0, 1: 0.0}
        for item in output:
            label = str(item["label"]).lower()
            score = float(item.get("score", 0.0))
            if label.endswith("1") or label == "large":
                scores[1] = score
            else:
                scores[0] = score
        if scores[0] == 0.0 and scores[1] == 0.0:
            scores[self.predict(text)] = 1.0
        total = scores[0] + scores[1]
        if total <= 0:
            return {0: 0.5, 1: 0.5}
        return {0: scores[0] / total, 1: scores[1] / total}

    def evaluate(self, dataset: Sequence[RouteExample]) -> Dict[str, float]:
        return _evaluate_predictions([self.predict(ex.text) for ex in dataset], [int(ex.label) for ex in dataset])


class EmbeddingClassifierRouter:
    """Embedding model + sklearn classifier router."""

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self._encoder = None
        self._classifier = None
        self._config = None

    def _load(self) -> None:
        if self._classifier is not None and self._encoder is not None and self._config is not None:
            return

        self._config = json.loads((self.artifact_dir / "summary.json").read_text(encoding="utf-8"))
        with (self.artifact_dir / "classifier.pkl").open("rb") as fh:
            self._classifier = pickle.load(fh)

        encoder_name = str(self._config["model_name"])
        from sentence_transformers import SentenceTransformer

        self._encoder = SentenceTransformer(encoder_name, local_files_only=True)

    def predict(self, text: str) -> int:
        probs = self.predict_proba(text)
        return max(probs, key=probs.get)

    def predict_proba(self, text: str) -> Dict[int, float]:
        self._load()
        vector = self._encoder.encode([text], normalize_embeddings=True)
        if hasattr(self._classifier, "predict_proba"):
            raw = self._classifier.predict_proba(vector)[0]
            classes = [int(label) for label in getattr(self._classifier, "classes_", range(len(raw)))]
            return {label: float(score) for label, score in zip(classes, raw)}
        pred = int(self._classifier.predict(vector)[0])
        labels = [int(label) for label in getattr(self._classifier, "classes_", [0, 1, 2])]
        return {label: 1.0 if label == pred else 0.0 for label in labels}

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
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": test_ds,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
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


def train_bge_router(
    dataset: RouteDataset,
    *,
    model_name: str,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    max_iter: int = 1000,
) -> AdvancedTrainingResult:
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    train_set, test_set = dataset.split(train_ratio=train_ratio)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        encoder = SentenceTransformer(model_name, local_files_only=True)
    except Exception as exc:
        raise ModelDownloadUnavailableError(
            f"Unable to load BGE model '{model_name}'. Make sure it is available in the local cache."
        ) from exc

    start = time.time()
    train_vectors = encoder.encode([item.text for item in train_set.examples], normalize_embeddings=True)
    test_vectors = encoder.encode([item.text for item in test_set.examples], normalize_embeddings=True)
    classifier = LogisticRegression(max_iter=max_iter, class_weight="balanced")
    classifier.fit(train_vectors, [int(item.label) for item in train_set.examples])
    preds = classifier.predict(test_vectors)
    metrics = _evaluate_predictions(list(preds), [int(item.label) for item in test_set.examples])
    seconds = time.time() - start

    with (out_dir / "classifier.pkl").open("wb") as fh:
        pickle.dump(classifier, fh)
    summary = {
        "method": "bge_m3_logreg",
        "model_name": model_name,
        "labels": sorted({int(item.label) for item in dataset.examples}),
        "route_label_schema": _infer_label_schema(dataset.examples),
        "train_size": len(train_set.examples),
        "test_size": len(test_set.examples),
        "seconds": seconds,
        "metrics": metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return AdvancedTrainingResult(
        method="bge_m3_logreg",
        metrics=metrics,
        artifacts={
            "artifact_dir": str(out_dir),
            "classifier": str(out_dir / "classifier.pkl"),
            "summary": str(out_dir / "summary.json"),
        },
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


def benchmark_embedding_router(
    dataset: RouteDataset,
    *,
    artifact_dir: str | Path,
    notes: str = "",
) -> Dict[str, Any]:
    router = EmbeddingClassifierRouter(artifact_dir)
    start = time.time()
    metrics = router.evaluate(dataset.examples)
    elapsed = time.time() - start
    per_item_ms = (elapsed / max(1, len(dataset.examples))) * 1000.0
    return {
        "method": "embedding_router",
        **metrics,
        "latency_ms": per_item_ms,
        "cost": 0.0,
        "notes": notes or str(artifact_dir),
    }


def _evaluate_predictions(preds: Sequence[int], labels: Sequence[int]) -> Dict[str, float]:
    label_set = sorted(set(int(label) for label in labels) | set(int(pred) for pred in preds))
    if any(label not in (0, 1) for label in label_set):
        return _evaluate_multiclass_predictions(preds, labels, label_set)

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


def _evaluate_multiclass_predictions(
    preds: Sequence[int],
    labels: Sequence[int],
    label_set: Sequence[int],
) -> Dict[str, float]:
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


def _infer_label_schema(dataset: Sequence[RouteExample]) -> str:
    for item in dataset:
        schema = str(item.metadata.get("route_label_schema") or "").strip()
        if schema:
            return schema
    labels = sorted({int(item.label) for item in dataset})
    if labels == [0, 1, 2]:
        return "0=device,1=edge,2=cloud"
    return "0=small,1=large"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4f}"
        return "-"
    return str(value)
