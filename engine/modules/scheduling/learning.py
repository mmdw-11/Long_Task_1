"""Learned binary router for small-model vs large-model routing.

This module keeps the current rule-based scheduler intact while adding:
- dataset generation from route traces or a teacher callback
- a lightweight pure-Python text classifier
- a TaskGate implementation that plugs into AdaptiveResourceScheduler

The initial implementation is intentionally dependency-free so it can run in
the current environment. Later, the text encoder can be swapped to BGE-M3 or
LoRA-backed embeddings without changing the gate interface.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

from ..bge_local import resolve_bge_m3_cache_dir, resolve_bge_m3_model_path
from ._types import RealtimeRequirement, ResourceRequest, SensitivityLevel, TaskComplexity, TaskProfile
from .gate import HeuristicTaskGate, TaskGate


class RouteTeacher(Protocol):
    """Assign a 0/1 routing label to a prompt."""

    def label(self, text: str, request: Optional[ResourceRequest] = None) -> int:
        raise NotImplementedError


@dataclass
class RouteExample:
    """A single binary routing example."""

    text: str
    label: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "label": int(self.label), "metadata": self.metadata}


@dataclass
class CascadeRouteDecision:
    """A true cascade routing decision produced from model execution."""

    label: int
    promoted: bool
    small_output: str = ""
    large_output: str = ""
    judge_reason: str = ""
    judge_confidence: Optional[float] = None
    models: Dict[str, str] = field(default_factory=dict)

    def to_example(self, text: str, **metadata: Any) -> RouteExample:
        return RouteExample(
            text=text,
            label=int(self.label),
            metadata={
                **metadata,
                "cascade": {
                    "promoted": self.promoted,
                    "small_output": self.small_output,
                    "large_output": self.large_output,
                    "judge_reason": self.judge_reason,
                    "judge_confidence": self.judge_confidence,
                    "models": dict(self.models),
                },
            },
        )


@dataclass
class RouteDataset:
    """JSONL-compatible dataset container."""

    examples: List[RouteExample] = field(default_factory=list)

    def add(self, text: str, label: int, **metadata: Any) -> None:
        self.examples.append(RouteExample(text=text, label=int(label), metadata=dict(metadata)))

    def extend(self, items: Iterable[RouteExample]) -> None:
        self.examples.extend(list(items))

    def save_jsonl(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for ex in self.examples:
                fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
        return out

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "RouteDataset":
        items: List[RouteExample] = []
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                items.append(
                    RouteExample(
                        text=str(data["text"]),
                        label=int(data["label"]),
                        metadata=dict(data.get("metadata") or {}),
                    )
                )
        return cls(items)

    def split(
        self, train_ratio: float = 0.8, *, seed: int = 42
    ) -> tuple["RouteDataset", "RouteDataset"]:
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        if len(self.examples) < 2:
            raise ValueError("dataset needs at least 2 examples")
        import random

        buckets: Dict[int, List[RouteExample]] = defaultdict(list)
        for item in self.examples:
            buckets[int(item.label)].append(item)
        train_items: List[RouteExample] = []
        test_items: List[RouteExample] = []
        rng = random.Random(seed)
        for label, items in sorted(buckets.items()):
            shuffled = list(items)
            rng.shuffle(shuffled)
            cut = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_ratio)))
            train_items.extend(shuffled[:cut])
            test_items.extend(shuffled[cut:])
        rng.shuffle(train_items)
        rng.shuffle(test_items)
        return RouteDataset(train_items), RouteDataset(test_items)


class PseudoCascadeTeacher:
    """Pseudo-labels from the current rule-based scheduler.

    Label convention:
    - 0: small model is sufficient
    - 1: large model is preferred
    """

    def __init__(self, gate: Optional[TaskGate] = None) -> None:
        self.gate = gate or HeuristicTaskGate()

    def label(self, text: str, request: Optional[ResourceRequest] = None) -> int:
        req = request or ResourceRequest(node="route_teacher", state={"input": text})
        profile = self.gate.evaluate(req)
        score = _route_complexity_score(text)
        if _needs_large_model(profile):
            score += 1.5
        return 1 if score >= 1.5 else 0

    def annotate(self, text: str, request: Optional[ResourceRequest] = None) -> RouteExample:
        req = request or ResourceRequest(node="route_teacher", state={"input": text})
        label = self.label(text, req)
        return RouteExample(
            text=text,
            label=label,
            metadata={
                "node": req.node,
                "route_source": type(self).__name__,
                "cascade": {
                    "promoted": bool(label),
                    "mode": "pseudo",
                },
            },
        )


class RealCascadeTeacher:
    """True cascade teacher: small model -> judge -> optional large model.

    Label convention:
    - 0: keep the small-model path
    - 1: promote to the large-model path
    """

    def __init__(
        self,
        *,
        small_model: Optional[str] = None,
        large_model: Optional[str] = None,
        judge_model: Optional[str] = None,
        client: Optional[Any] = None,
        run_large_on_promote: bool = True,
        temperature: float = 0.0,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.small_model = small_model
        self.large_model = large_model
        self.judge_model = judge_model
        self.client = client
        self.run_large_on_promote = run_large_on_promote
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def label(self, text: str, request: Optional[ResourceRequest] = None) -> int:
        return self.annotate(text, request).label

    def annotate(self, text: str, request: Optional[ResourceRequest] = None) -> RouteExample:
        req = request or ResourceRequest(node="route_teacher", state={"input": text})
        decision = self._route(req, text)
        return decision.to_example(
            text,
            node=req.node,
            route_source=type(self).__name__,
            task_type=str(req.metadata.get("task_type") or "general"),
        )

    def _route(self, request: ResourceRequest, text: str) -> CascadeRouteDecision:
        client = self._get_client()
        small_model, judge_model, large_model = self._resolve_models()

        small_output = self._chat(
            client,
            model=small_model,
            system_prompt=(
                "You are the small-model stage in a cascade router. "
                "Answer quickly and concisely, but do not fabricate certainty."
            ),
            user_prompt=_build_small_stage_prompt(request, text),
        )
        judge_data = self._judge(
            client,
            model=judge_model,
            request=request,
            text=text,
            small_output=small_output,
        )
        promoted = bool(judge_data.get("use_large", False))
        large_output = ""
        if promoted and self.run_large_on_promote:
            large_output = self._chat(
                client,
                model=large_model,
                system_prompt=(
                    "You are the large-model stage in a cascade router. "
                    "Produce the stronger answer for the task."
                ),
                user_prompt=_build_large_stage_prompt(request, text, small_output, judge_data),
            )
        return CascadeRouteDecision(
            label=1 if promoted else 0,
            promoted=promoted,
            small_output=small_output,
            large_output=large_output,
            judge_reason=str(judge_data.get("reason", "")),
            judge_confidence=_coerce_optional_float(judge_data.get("confidence")),
            models={
                "small": small_model,
                "judge": judge_model,
                "large": large_model,
            },
        )

    def _judge(
        self,
        client: Any,
        *,
        model: str,
        request: ResourceRequest,
        text: str,
        small_output: str,
    ) -> Dict[str, Any]:
        raw = self._chat(
            client,
            model=model,
            system_prompt=(
                "You are a cascade routing judge. "
                "Return strict JSON only."
            ),
            user_prompt=_build_judge_prompt(request, text, small_output),
        )
        data = _parse_json_object(raw)
        return {
            "use_large": bool(data.get("use_large", False)),
            "confidence": data.get("confidence"),
            "reason": str(data.get("reason", "")),
        }

    def _resolve_models(self) -> tuple[str, str, str]:
        settings = _load_openai_settings()
        fallback_model = settings.model
        small_model = self.small_model or _get_env("OPENAI_SMALL_MODEL") or fallback_model
        large_model = self.large_model or _get_env("OPENAI_LARGE_MODEL") or fallback_model
        judge_model = self.judge_model or _get_env("OPENAI_JUDGE_MODEL") or small_model
        return small_model, judge_model, large_model

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise ImportError(
                "RealCascadeTeacher requires the openai package in the active environment."
            ) from exc
        settings = _load_openai_settings()
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            organization=settings.organization,
            timeout=self.timeout_seconds,
        )
        return self.client

    def _chat(
        self,
        client: Any,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        try:
            response = client.chat.completions.create(
                **payload,
                timeout=self.timeout_seconds,
            )
        except TypeError:
            response = client.chat.completions.create(**payload)
        return str(response.choices[0].message.content or "").strip()


class FallbackCascadeTeacher:
    """Prefer the real cascade teacher and fall back to pseudo labels."""

    def __init__(
        self,
        primary: Optional[RealCascadeTeacher] = None,
        fallback: Optional[PseudoCascadeTeacher] = None,
        disable_primary_after_error: bool = True,
    ) -> None:
        self.primary = primary or RealCascadeTeacher()
        self.fallback = fallback or PseudoCascadeTeacher()
        self.disable_primary_after_error = disable_primary_after_error
        self._primary_disabled = False

    def label(self, text: str, request: Optional[ResourceRequest] = None) -> int:
        return self.annotate(text, request).label

    def annotate(self, text: str, request: Optional[ResourceRequest] = None) -> RouteExample:
        req = request or ResourceRequest(node="route_teacher", state={"input": text})
        if self._primary_disabled:
            example = self.fallback.annotate(text, req)
            example.metadata["teacher_mode"] = "pseudo_fallback_disabled"
            return example
        try:
            example = self.primary.annotate(text, req)
            example.metadata["teacher_mode"] = "real"
            return example
        except Exception as exc:
            if self.disable_primary_after_error:
                self._primary_disabled = True
            example = self.fallback.annotate(text, req)
            example.metadata["teacher_mode"] = "pseudo_fallback"
            example.metadata["teacher_error"] = str(exc)
            return example


class TextEncoder(Protocol):
    """Feature extractor protocol."""

    def encode(self, text: str) -> Dict[str, float]:
        raise NotImplementedError


class BagOfWordsEncoder:
    """Simple token + bigram encoder."""

    def encode(self, text: str) -> Dict[str, float]:
        tokens = _tokenize(text)
        feats: Dict[str, float] = {}
        for token in tokens:
            feats[token] = feats.get(token, 0.0) + 1.0
        for left, right in zip(tokens, tokens[1:]):
            key = f"{left}__{right}"
            feats[key] = feats.get(key, 0.0) + 1.0
        feats["__len_bucket_short__" if len(text) < 200 else "__len_bucket_long__"] = 1.0
        return feats


class MixedNgramEncoder:
    """Token + character n-gram encoder for Chinese/English mixed text.

    This tends to work much better than whitespace tokenization for Chinese
    prompts while remaining dependency-free.
    """

    def __init__(self, char_ngram_min: int = 2, char_ngram_max: int = 4) -> None:
        self.char_ngram_min = char_ngram_min
        self.char_ngram_max = char_ngram_max

    def encode(self, text: str) -> Dict[str, float]:
        feats = BagOfWordsEncoder().encode(text)
        compact = re.sub(r"\s+", "", text.lower())
        for n in range(self.char_ngram_min, self.char_ngram_max + 1):
            if len(compact) < n:
                continue
            for idx in range(len(compact) - n + 1):
                key = f"char:{compact[idx:idx+n]}"
                feats[key] = feats.get(key, 0.0) + 1.0
        if any(ch.isdigit() for ch in text):
            feats["__has_digit__"] = 1.0
        if any(ch.isalpha() for ch in text):
            feats["__has_alpha__"] = 1.0
        if any("\u4e00" <= ch <= "\u9fff" for ch in text):
            feats["__has_cjk__"] = 1.0
        return feats


class BgeM3Encoder:
    """Optional BGE-M3 embedding adapter.

    This is a placeholder for the next stage. It intentionally raises a clear
    error if the dependency is not installed so the current project can run
    without extra packages.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", cache_dir: Optional[str] = None) -> None:
        self.model_name = resolve_bge_m3_model_path(model_name)
        self.cache_dir = resolve_bge_m3_cache_dir(cache_dir)
        self._model = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "BGE-M3 encoder requires FlagEmbedding. Install it before use."
            ) from exc
        try:
            self._model = BGEM3FlagModel(self.model_name, cache_dir=self.cache_dir)
        except Exception as exc:  # pragma: no cover - depends on model cache/network
            raise RuntimeError(
                f"Unable to load BGE-M3 encoder '{self.model_name}'. "
                f"Ensure the model is available and the cache directory is writable: {self.cache_dir}"
            ) from exc
        return self._model

    def encode(self, text: str) -> Dict[str, float]:
        model = self._load()
        result = model.encode(text, return_dense=False, return_sparse=True, return_colbert_vecs=False)
        sparse = result.get("lexical_weights") or {}
        feats: Dict[str, float] = {}
        for token, weight in sparse.items():
            feats[f"bge:{token}"] = float(weight)
        feats["__bge_m3__"] = 1.0
        return feats


def make_default_encoder() -> TextEncoder:
    return MixedNgramEncoder()


class BinaryTextRouterModel:
    """Pure-Python multinomial Naive Bayes router."""

    def __init__(self, encoder: Optional[TextEncoder] = None) -> None:
        self.encoder = encoder or make_default_encoder()
        self.class_doc_counts: Dict[int, int] = {0: 0, 1: 0}
        self.class_token_counts: Dict[int, Counter[str]] = {0: Counter(), 1: Counter()}
        self.class_totals: Dict[int, float] = {0: 0.0, 1: 0.0}
        self.vocab: set[str] = set()
        self.fitted = False

    def fit(self, dataset: Sequence[RouteExample]) -> "BinaryTextRouterModel":
        if not dataset:
            raise ValueError("dataset is empty")
        for ex in dataset:
            label = int(ex.label)
            feats = self.encoder.encode(ex.text)
            self.class_doc_counts[label] += 1
            for token, weight in feats.items():
                self.class_token_counts[label][token] += weight
                self.class_totals[label] += weight
                self.vocab.add(token)
        self.fitted = True
        return self

    def predict_proba(self, text: str) -> Dict[int, float]:
        if not self.fitted:
            raise RuntimeError("model is not fitted")
        feats = self.encoder.encode(text)
        scores = {label: self._log_score(label, feats) for label in (0, 1)}
        return _softmax(scores)

    def predict(self, text: str) -> int:
        probs = self.predict_proba(text)
        return 1 if probs[1] >= probs[0] else 0

    def evaluate(self, dataset: Sequence[RouteExample]) -> Dict[str, float]:
        tp = fp = tn = fn = 0
        for ex in dataset:
            pred = self.predict(ex.text)
            if pred == 1 and ex.label == 1:
                tp += 1
            elif pred == 1 and ex.label == 0:
                fp += 1
            elif pred == 0 and ex.label == 0:
                tn += 1
            else:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        acc = (tp + tn) / max(1, tp + tn + fp + fn)
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    def to_dict(self) -> Dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("model is not fitted")
        return {
            "encoder": "bag_of_words_v1",
            "class_doc_counts": self.class_doc_counts,
            "class_token_counts": {
                str(label): dict(counter) for label, counter in self.class_token_counts.items()
            },
            "class_totals": self.class_totals,
            "vocab": sorted(self.vocab),
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path, encoder: Optional[TextEncoder] = None) -> "BinaryTextRouterModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(encoder=encoder)
        model.class_doc_counts = {int(k): int(v) for k, v in data["class_doc_counts"].items()}
        model.class_token_counts = {
            int(label): Counter({k: float(v) for k, v in counts.items()})
            for label, counts in data["class_token_counts"].items()
        }
        model.class_totals = {int(k): float(v) for k, v in data["class_totals"].items()}
        model.vocab = set(data.get("vocab") or [])
        model.fitted = True
        return model

    def _log_score(self, label: int, feats: Dict[str, float]) -> float:
        vocab_size = max(1, len(self.vocab))
        total_docs = sum(self.class_doc_counts.values())
        prior = math.log((self.class_doc_counts[label] + 1) / (total_docs + 2))
        total = self.class_totals[label] + vocab_size
        score = prior
        for token, weight in feats.items():
            token_count = self.class_token_counts[label].get(token, 0.0) + 1.0
            score += weight * math.log(token_count / total)
        return score


class LearnedTaskGate(TaskGate):
    """Task gate backed by a trained binary router."""

    def __init__(
        self,
        model: BinaryTextRouterModel,
        *,
        threshold: float = 0.5,
        fallback: Optional[TaskGate] = None,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.fallback = fallback or HeuristicTaskGate()

    def evaluate(self, request: ResourceRequest) -> TaskProfile:
        text = _collect_request_text(request)
        try:
            probs = self.model.predict_proba(text)
            label = 1 if probs[1] >= self.threshold else 0
            return self._profile_from_label(label, probs[1], request)
        except Exception:
            profile = self.fallback.evaluate(request)
            profile.metadata = {**profile.metadata, "gate": "learned_fallback"}
            return profile

    def _profile_from_label(
        self, label: int, score: float, request: ResourceRequest
    ) -> TaskProfile:
        if label == 1:
            return TaskProfile(
                realtime=RealtimeRequirement.NORMAL,
                sensitivity=SensitivityLevel.INTERNAL,
                complexity=TaskComplexity.HIGH,
                task_type=str(request.metadata.get("task_type") or "general"),
                requires_trusted_workspace=bool(request.metadata.get("requires_trusted_workspace")),
                human_approved=bool(
                    request.metadata.get("human_approved")
                    or request.state.get("human_approved")
                    or request.state.get("cloud_audit_approved")
                ),
                metadata={
                    "node": request.node,
                    "router": "learned",
                    "route": "large",
                    "score": score,
                },
            )
        return TaskProfile(
            realtime=RealtimeRequirement.INTERACTIVE,
            sensitivity=SensitivityLevel.INTERNAL,
            complexity=TaskComplexity.LOW,
            task_type=str(request.metadata.get("task_type") or "general"),
            requires_trusted_workspace=False,
            human_approved=bool(
                request.metadata.get("human_approved")
                or request.state.get("human_approved")
                or request.state.get("cloud_audit_approved")
            ),
            metadata={
                "node": request.node,
                "router": "learned",
                "route": "small",
                "score": score,
            },
        )


def evaluate_gate(
    gate: TaskGate,
    dataset: Sequence[RouteExample],
    *,
    request_factory: Optional[Callable[[str], ResourceRequest]] = None,
) -> Dict[str, float]:
    request_factory = request_factory or (lambda text: ResourceRequest(node="router", state={"input": text}))
    tp = fp = tn = fn = 0
    for ex in dataset:
        req = request_factory(ex.text)
        profile = gate.evaluate(req)
        pred = 1 if _needs_large_model(profile) else 0
        if pred == 1 and ex.label == 1:
            tp += 1
        elif pred == 1 and ex.label == 0:
            fp += 1
        elif pred == 0 and ex.label == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def build_route_dataset(
    texts: Iterable[str],
    *,
    teacher: Optional[RouteTeacher] = None,
    request_factory: Optional[Callable[[str], ResourceRequest]] = None,
) -> RouteDataset:
    teacher = teacher or FallbackCascadeTeacher()
    request_factory = request_factory or (lambda text: ResourceRequest(node="router", state={"input": text}))
    dataset = RouteDataset()
    for text in texts:
        req = request_factory(text)
        if hasattr(teacher, "annotate"):
            dataset.extend([teacher.annotate(text, req)])  # type: ignore[attr-defined]
            continue
        label = teacher.label(text, req)
        dataset.add(text, label, node=req.node, route_source=type(teacher).__name__)
    return dataset


def build_route_dataset_from_requests(
    requests: Iterable[ResourceRequest],
    *,
    teacher: Optional[RouteTeacher] = None,
) -> RouteDataset:
    teacher = teacher or FallbackCascadeTeacher()
    dataset = RouteDataset()
    for req in requests:
        text = _collect_request_text(req)
        if hasattr(teacher, "annotate"):
            dataset.extend([teacher.annotate(text, req)])  # type: ignore[attr-defined]
            continue
        label = teacher.label(text, req)
        dataset.add(text, label, node=req.node, route_source=type(teacher).__name__)
    return dataset


def train_router(
    dataset: RouteDataset,
    *,
    encoder: Optional[TextEncoder] = None,
    train_ratio: float = 0.8,
) -> Dict[str, Any]:
    train_set, test_set = dataset.split(train_ratio=train_ratio)
    model = BinaryTextRouterModel(encoder=encoder).fit(train_set.examples)
    metrics = model.evaluate(test_set.examples) if test_set.examples else {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {"model": model, "metrics": metrics, "train_size": len(train_set.examples), "test_size": len(test_set.examples)}


def benchmark_router(
    dataset: RouteDataset,
    *,
    encoder: Optional[TextEncoder] = None,
    train_ratio: float = 0.8,
) -> List[Dict[str, Any]]:
    train_set, test_set = dataset.split(train_ratio=train_ratio)
    model = BinaryTextRouterModel(encoder=encoder).fit(train_set.examples)
    rule_gate = HeuristicTaskGate()
    learned_gate = LearnedTaskGate(model, threshold=0.35)
    rows = [
        {"method": "rule_gate", **evaluate_gate(rule_gate, test_set.examples), "notes": "heuristic gate"},
        {"method": "learned_gate", **evaluate_gate(learned_gate, test_set.examples), "notes": "mixed n-gram NB"},
    ]
    return rows


def render_experiment_report(rows: Sequence[Dict[str, Any]]) -> str:
    headers = ["method", "accuracy", "precision", "recall", "f1", "notes"]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table.append(
            "| "
            + " | ".join(
                [
                    str(row.get("method", "")),
                    _fmt_metric(row.get("accuracy")),
                    _fmt_metric(row.get("precision")),
                    _fmt_metric(row.get("recall")),
                    _fmt_metric(row.get("f1")),
                    str(row.get("notes", "")),
                ]
            )
            + " |"
        )
    return "\n".join(table)


def save_experiment_report(rows: Sequence[Dict[str, Any]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_experiment_report(rows), encoding="utf-8")
    return out


def default_training_texts() -> List[str]:
    return [
        "总结一下这封邮件的要点",
        "帮我写一个 Python 排序算法并加测试",
        "创建明天下午三点的项目会议日程",
        "请分析这份 20 页的复杂技术文档",
        "把这段日志里的 token 和 key 脱敏",
        "解释一下这个数学证明",
        "给我写一个简单的 TODO 列表",
        "把这张图里的文字识别出来并总结",
        "将客户投诉邮件提炼为三条行动建议",
        "生成一个适合初学者的登录页面代码",
        "请比较两个架构方案并给出优劣分析",
        "帮我安排下周一上午的评审会议",
        "识别截图中的地址和手机号并隐藏",
        "针对这段长文档输出结构化摘要",
        "根据预算表生成采购建议",
        "写一个复杂的并发爬虫并考虑错误重试",
        "把这封会议邀请转成日程事件",
        "解释这段数据库慢查询的原因",
        "生成一份简短的日报模板",
        "根据这段代码找出潜在的性能瓶颈",
        "把这段对话记录总结成要点",
        "请对这份合同做风险分析",
    ]


def _collect_request_text(request: ResourceRequest) -> str:
    parts = [request.node]
    for key in ("input", "task", "goal", "query", "messages"):
        value = request.state.get(key)
        if value:
            parts.append(str(value))
    for key in ("description", "sys_prompt"):
        value = request.metadata.get(key)
        if value:
            parts.append(str(value))
    return "\n".join(parts)


def _needs_large_model(profile: TaskProfile) -> bool:
    return (
        profile.complexity in (TaskComplexity.HIGH, TaskComplexity.EXTREME)
        or profile.sensitivity in (SensitivityLevel.CONFIDENTIAL, SensitivityLevel.SECRET)
        or profile.realtime == RealtimeRequirement.HARD
    )


def _route_complexity_score(text: str) -> float:
    lowered = text.lower()
    score = 0.0
    hard_keywords = [
        "复杂",
        "重构",
        "性能",
        "架构",
        "调试",
        "debug",
        "refactor",
        "optimize",
        "分析",
        "比较",
        "评估",
        "风险",
        "推理",
        "代码",
        "程序",
        "算法",
        "文档",
        "合同",
        "图像",
        "ocr",
        "视频",
        "表格",
    ]
    easy_keywords = [
        "总结",
        "摘要",
        "日程",
        "待办",
        "提醒",
        "calendar",
        "会议",
        "翻译",
        "列出",
        "简化",
    ]
    if any(word in lowered for word in hard_keywords):
        score += 1.0
    if any(word in lowered for word in easy_keywords):
        score -= 0.25
    if len(text) > 120:
        score += 0.25
    if len(text) > 300:
        score += 0.5
    if len(text) > 800:
        score += 0.75
    return score


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def _softmax(scores: Dict[int, float]) -> Dict[int, float]:
    max_score = max(scores.values())
    exps = {label: math.exp(score - max_score) for label, score in scores.items()}
    total = sum(exps.values()) or 1.0
    return {label: value / total for label, value in exps.items()}


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _build_small_stage_prompt(request: ResourceRequest, text: str) -> str:
    return (
        f"Node: {request.node}\n"
        f"Task:\n{text}\n\n"
        "Provide your best concise answer. If the task seems underspecified, "
        "say what is missing instead of fabricating."
    )


def _build_judge_prompt(request: ResourceRequest, text: str, small_output: str) -> str:
    return (
        "Decide whether the cascade should escalate from the small model to the large model.\n"
        "Return JSON with keys: use_large (bool), confidence (0-1), reason (string).\n\n"
        f"Node: {request.node}\n"
        f"Task:\n{text}\n\n"
        f"Small model answer:\n{small_output}\n\n"
        "Escalate when the answer is weak, incomplete, hallucinated, or the task needs deep reasoning, "
        "long-context synthesis, multimodal ability, or stronger reliability."
    )


def _build_large_stage_prompt(
    request: ResourceRequest,
    text: str,
    small_output: str,
    judge_data: Dict[str, Any],
) -> str:
    return (
        f"Node: {request.node}\n"
        f"Task:\n{text}\n\n"
        f"Small model draft:\n{small_output}\n\n"
        f"Why escalation was requested:\n{judge_data.get('reason', '')}\n\n"
        "Produce the stronger final answer."
    )


def _parse_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    return json.loads(stripped or "{}")


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_openai_settings() -> Any:
    from engine.config import load_settings

    return load_settings()


def _get_env(name: str) -> Optional[str]:
    try:
        import os

        value = os.environ.get(name)
        return value or None
    except Exception:
        return None
