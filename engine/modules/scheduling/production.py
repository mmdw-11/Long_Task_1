"""Production helpers for loading trained routing gates."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .gate import HeuristicTaskGate, TaskGate
from .learning import BinaryTextRouterModel, LearnedTaskGate


DEFAULT_ROUTER_PATH = Path("runs/router_learning/router.json")


def resolve_router_path(explicit_path: Optional[str | Path] = None) -> Optional[Path]:
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None

    env_path = os.environ.get("ROUTER_MODEL_PATH")
    if env_path:
        path = Path(env_path)
        return path if path.exists() else None
    return None


def resolve_default_router_path() -> Optional[Path]:
    return DEFAULT_ROUTER_PATH if DEFAULT_ROUTER_PATH.exists() else None


def load_production_gate(
    *,
    router_path: Optional[str | Path] = None,
    threshold: float = 0.35,
    fallback: Optional[TaskGate] = None,
) -> TaskGate:
    fallback = fallback or HeuristicTaskGate()
    resolved = resolve_router_path(router_path)
    if resolved is None:
        return fallback

    try:
        model = BinaryTextRouterModel.load(resolved)
        return LearnedTaskGate(model, threshold=threshold, fallback=fallback)
    except Exception:
        return fallback
