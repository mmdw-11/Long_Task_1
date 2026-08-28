"""Device/cloud routing controls with quality and resource proxy metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict

from engine.modules.scheduling import BinaryTextRouterModel, HeuristicTaskGate, RouteDataset
from engine.modules.scheduling._types import ResourceRequest, SensitivityLevel, TaskComplexity

from .reports import ExperimentReport
from .types import ExperimentRow


@dataclass
class RoutingExperimentConfig:
    train_ratio: float = 0.8
    device_latency_ms: float = 40.0
    cloud_latency_ms: float = 800.0
    device_cost: float = 0.001
    cloud_cost: float = 0.02


def run_routing_experiment(
    dataset: RouteDataset,
    config: RoutingExperimentConfig | None = None,
) -> Dict[str, ExperimentReport]:
    """Evaluate All Device/Cloud, heuristic, and lightweight learned routing.

    Latency and cost are configurable proxy values; classification and sensitive
    cloud-send metrics are measured from actual routing decisions.
    """
    cfg = config or RoutingExperimentConfig()
    train, test = dataset.split(train_ratio=cfg.train_ratio)
    if not train.examples or not test.examples:
        raise ValueError("routing dataset must produce non-empty train and test splits")
    model = BinaryTextRouterModel().fit(train.examples)
    heuristic = HeuristicTaskGate()

    def heuristic_predict(text: str) -> int:
        profile = heuristic.evaluate(ResourceRequest(node="router", state={"input": text}))
        return int(profile.complexity in {TaskComplexity.HIGH, TaskComplexity.EXTREME})

    predictors: Dict[str, Callable[[str], int]] = {
        "all_device": lambda _text: 0,
        "all_cloud": lambda _text: 1,
        "heuristic": heuristic_predict,
        "learned": model.predict,
    }
    return {
        name: _evaluate_predictor(name, predictor, test, heuristic, cfg)
        for name, predictor in predictors.items()
    }


def _evaluate_predictor(
    name: str,
    predictor: Callable[[str], int],
    dataset: RouteDataset,
    heuristic: HeuristicTaskGate,
    cfg: RoutingExperimentConfig,
) -> ExperimentReport:
    rows = []
    for index, example in enumerate(dataset.examples):
        started = time.perf_counter()
        prediction = int(predictor(example.text))
        decision_ms = (time.perf_counter() - started) * 1000
        profile = heuristic.evaluate(
            ResourceRequest(node="router", state={"input": example.text})
        )
        sensitive = profile.sensitivity in {
            SensitivityLevel.CONFIDENTIAL,
            SensitivityLevel.SECRET,
        }
        correct = prediction == int(example.label)
        rows.append(
            ExperimentRow(
                id=str(example.metadata.get("id") or f"route-{index:05d}"),
                passed=correct,
                score=1.0 if correct else 0.0,
                prediction="cloud" if prediction else "device",
                expected="cloud" if example.label else "device",
                metrics={
                    "tp": int(prediction == 1 and example.label == 1),
                    "fp": int(prediction == 1 and example.label == 0),
                    "tn": int(prediction == 0 and example.label == 0),
                    "fn": int(prediction == 0 and example.label == 1),
                    "cloud_call": prediction,
                    "latency_ms": cfg.cloud_latency_ms if prediction else cfg.device_latency_ms,
                    "cost": cfg.cloud_cost if prediction else cfg.device_cost,
                    "sensitive_cloud_send": int(sensitive and prediction == 1),
                    "human_approval": int(sensitive and prediction == 1),
                    "fallback": 0,
                    "decision_ms": decision_ms,
                },
                metadata={"method": name, "sensitive": sensitive},
            )
        )
    classification = _classification_metrics(rows)
    return ExperimentReport(
        name=f"routing-{name}",
        rows=rows,
        metadata={
            "method": name,
            **classification,
            "latency_cost_note": "configurable proxy, replace with trace measurements for final paper",
        },
    )


def _classification_metrics(rows: list[ExperimentRow]) -> Dict[str, float]:
    totals = {
        key: sum(int(row.metrics[key]) for row in rows)
        for key in ("tp", "fp", "tn", "fn")
    }
    tp, fp, tn, fn = (totals[key] for key in ("tp", "fp", "tn", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / max(1, tp + fp + tn + fn),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }
