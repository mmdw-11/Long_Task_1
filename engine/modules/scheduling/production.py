"""Production helpers for loading trained routing gates."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from ._types import RealtimeRequirement, ResourceRequest, SensitivityLevel, TaskComplexity, TaskProfile
from .advanced_training import EmbeddingClassifierRouter, TransformerTextRouter
from .gate import HeuristicTaskGate, TaskGate
from .learning import BinaryTextRouterModel, LearnedTaskGate


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROUTER_BACKEND = "bge_m3"
DEFAULT_ROUTER_PATHS: Dict[str, Path] = {
    "nb": PACKAGE_ROOT / "runs/router_learning/balanced_router.json",
    "bert_full": PACKAGE_ROOT / "runs/router_learning/balanced_bert_full/model",
    "bert_lora": PACKAGE_ROOT / "runs/router_learning/balanced_bert_lora/model",
    "bge_m3": PACKAGE_ROOT / "runs/router_learning/balanced_bge_m3",
}
LEGACY_DEFAULT_ROUTER_PATH = PACKAGE_ROOT / "runs/router_learning/router.json"


class ProbabilisticRouter(Protocol):
    def predict_proba(self, text: str) -> Dict[int, float]:
        raise NotImplementedError


class AdvancedLearnedTaskGate(TaskGate):
    """Task gate adapter for transformer and embedding routers."""

    def __init__(
        self,
        model: ProbabilisticRouter,
        *,
        threshold: float = 0.5,
        fallback: Optional[TaskGate] = None,
        router_name: str = "advanced",
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.fallback = fallback or HeuristicTaskGate()
        self.router_name = router_name

    def evaluate(self, request: ResourceRequest) -> TaskProfile:
        text = _collect_request_text(request)
        try:
            probs = self.model.predict_proba(text)
            score = float(probs.get(1, 0.0))
            label = 1 if score >= self.threshold else 0
            return _profile_from_label(
                label=label,
                score=score,
                request=request,
                router_name=self.router_name,
            )
        except Exception:
            profile = self.fallback.evaluate(request)
            profile.metadata = {**profile.metadata, "gate": f"{self.router_name}_fallback"}
            return profile


def resolve_router_backend(explicit_backend: Optional[str] = None) -> str:
    backend = explicit_backend or os.environ.get("ROUTER_MODEL_BACKEND") or DEFAULT_ROUTER_BACKEND
    return str(backend).strip().lower()


def infer_router_backend_from_path(path: str | Path) -> Optional[str]:
    resolved = Path(path)
    if resolved.is_file() and resolved.suffix.lower() == ".json":
        return "nb"
    name = resolved.name.lower()
    if "bert_lora" in name:
        return "bert_lora"
    if "bert" in name:
        return "bert_full"
    if "bge" in name:
        return "bge_m3"
    return None


def resolve_router_path(
    explicit_path: Optional[str | Path] = None,
    *,
    backend: Optional[str] = None,
) -> Optional[Path]:
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None

    env_path = os.environ.get("ROUTER_MODEL_PATH")
    if env_path:
        path = Path(env_path)
        return path if path.exists() else None

    backend_name = resolve_router_backend(backend)
    default_path = DEFAULT_ROUTER_PATHS.get(backend_name)
    if default_path and default_path.exists():
        return default_path
    if LEGACY_DEFAULT_ROUTER_PATH.exists():
        return LEGACY_DEFAULT_ROUTER_PATH
    return None


def resolve_default_router_path(*, backend: Optional[str] = None) -> Optional[Path]:
    return resolve_router_path(backend=backend)


def load_production_gate(
    *,
    router_path: Optional[str | Path] = None,
    threshold: float = 0.35,
    fallback: Optional[TaskGate] = None,
    backend: Optional[str] = None,
) -> TaskGate:
    fallback = fallback or HeuristicTaskGate()
    inferred_backend = infer_router_backend_from_path(router_path) if router_path else None
    resolved_backend = resolve_router_backend(backend or inferred_backend)
    resolved_path = resolve_router_path(router_path, backend=resolved_backend)
    if resolved_path is None:
        return fallback

    try:
        if resolved_backend == "nb":
            model = BinaryTextRouterModel.load(resolved_path)
            return LearnedTaskGate(model, threshold=threshold, fallback=fallback)
        if resolved_backend in {"bert_full", "bert_lora"}:
            model = TransformerTextRouter(resolved_path)
            return AdvancedLearnedTaskGate(
                model,
                threshold=threshold,
                fallback=fallback,
                router_name=resolved_backend,
            )
        if resolved_backend == "bge_m3":
            model = EmbeddingClassifierRouter(resolved_path)
            return AdvancedLearnedTaskGate(
                model,
                threshold=threshold,
                fallback=fallback,
                router_name=resolved_backend,
            )
        return fallback
    except Exception:
        if resolved_backend != "nb" and LEGACY_DEFAULT_ROUTER_PATH.exists():
            try:
                model = BinaryTextRouterModel.load(LEGACY_DEFAULT_ROUTER_PATH)
                return LearnedTaskGate(model, threshold=threshold, fallback=fallback)
            except Exception:
                return fallback
        return fallback


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


def _profile_from_label(
    *,
    label: int,
    score: float,
    request: ResourceRequest,
    router_name: str,
) -> TaskProfile:
    human_approved = bool(
        request.metadata.get("human_approved")
        or request.state.get("human_approved")
        or request.state.get("cloud_audit_approved")
    )
    task_type = str(request.metadata.get("task_type") or "general")
    if label == 1:
        return TaskProfile(
            realtime=RealtimeRequirement.NORMAL,
            sensitivity=SensitivityLevel.INTERNAL,
            complexity=TaskComplexity.HIGH,
            task_type=task_type,
            requires_trusted_workspace=bool(request.metadata.get("requires_trusted_workspace")),
            human_approved=human_approved,
            metadata={
                "node": request.node,
                "router": "learned",
                "router_backend": router_name,
                "route": "large",
                "score": score,
            },
        )
    return TaskProfile(
        realtime=RealtimeRequirement.INTERACTIVE,
        sensitivity=SensitivityLevel.INTERNAL,
        complexity=TaskComplexity.LOW,
        task_type=task_type,
        requires_trusted_workspace=False,
        human_approved=human_approved,
        metadata={
            "node": request.node,
            "router": "learned",
            "router_backend": router_name,
            "route": "small",
            "score": score,
        },
    )


def read_router_summary(router_path: str | Path) -> Dict[str, Any]:
    path = Path(router_path)
    summary_path = path / "summary.json" if path.is_dir() else path.with_name("summary.json")
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
