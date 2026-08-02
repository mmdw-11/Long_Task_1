"""Graph execution checkpoint and resume support."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GraphCheckpoint:
    """Serializable graph execution checkpoint."""

    run_id: str
    checkpoint_id: str
    step: int
    frontier: List[str]
    state: Dict[str, Any]
    status: str = "running"
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "checkpoint_id": self.checkpoint_id,
            "step": self.step,
            "frontier": list(self.frontier),
            "state": self.state,
            "status": self.status,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphCheckpoint":
        return cls(
            run_id=str(data["run_id"]),
            checkpoint_id=str(data["checkpoint_id"]),
            step=int(data["step"]),
            frontier=[str(item) for item in data.get("frontier", [])],
            state=dict(data.get("state") or {}),
            status=str(data.get("status", "running")),
            created_at=float(data.get("created_at", time.time())),
            metadata=dict(data.get("metadata") or {}),
        )


class GraphCheckpointStore:
    """File-backed graph checkpoint store."""

    def __init__(self, root_dir: str | Path = "runs/graph_checkpoints") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        run_id: str,
        step: int,
        frontier: List[str],
        state: Dict[str, Any],
        status: str = "running",
        checkpoint_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphCheckpoint:
        checkpoint_id = checkpoint_id or f"step_{step:04d}"
        checkpoint = GraphCheckpoint(
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            step=step,
            frontier=list(frontier),
            state=state,
            status=status,
            metadata=dict(metadata or {}),
        )
        path = self.path_for(run_id, checkpoint_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        latest = self.root_dir / _safe_name(run_id) / "latest.json"
        latest.write_text(str(path), encoding="utf-8")
        return checkpoint

    def load(self, run_id: str, checkpoint_id: str = "latest") -> GraphCheckpoint:
        if checkpoint_id == "latest":
            latest = self.root_dir / _safe_name(run_id) / "latest.json"
            path = Path(latest.read_text(encoding="utf-8"))
        else:
            path = self.path_for(run_id, checkpoint_id)
        return GraphCheckpoint.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def path_for(self, run_id: str, checkpoint_id: str) -> Path:
        return self.root_dir / _safe_name(run_id) / f"{_safe_name(checkpoint_id)}.json"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value) or "item"
