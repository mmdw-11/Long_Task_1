"""Local BGE-M3 model path resolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def resolve_bge_m3_model_path(model_name: str = "BAAI/bge-m3") -> str:
    """Prefer local BGE-M3 artifacts over HuggingFace repo ids."""
    if model_name != "BAAI/bge-m3":
        return model_name

    for raw in (
        os.environ.get("BGE_M3_MODEL_PATH"),
        _project_snapshot(),
        Path("D:/PythonProject/hf_cache/models/BAAI--bge-m3"),
    ):
        if not raw:
            continue
        path = Path(raw)
        if _looks_like_bge_model(path):
            return str(path)
    return model_name


def resolve_bge_m3_cache_dir(cache_dir: Optional[str] = None) -> str:
    if cache_dir:
        return cache_dir
    env_cache = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    if env_cache:
        return env_cache
    return str(Path(__file__).resolve().parents[2] / ".hf_cache" / "huggingface")


def _project_snapshot() -> Optional[Path]:
    root = Path(__file__).resolve().parents[2]
    snapshots = root / ".hf_cache" / "huggingface" / "models--BAAI--bge-m3" / "snapshots"
    if not snapshots.exists():
        return None
    candidates = [path for path in snapshots.iterdir() if path.is_dir()]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _looks_like_bge_model(path: Path) -> bool:
    return (
        path.exists()
        and (path / "config.json").exists()
        and (
            (path / "pytorch_model.bin").exists()
            or (path / "model.safetensors").exists()
        )
    )
