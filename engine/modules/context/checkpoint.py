"""Context checkpoint snapshots."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .ledger import ContextLedgerStore


@dataclass
class ContextCheckpoint:
    """A recoverable snapshot of run-scoped context artifacts."""

    run_id: str
    checkpoint_id: str
    created_at: float
    ledger_path: str
    memory_path: str
    raw_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at,
            "ledger_path": self.ledger_path,
            "memory_path": self.memory_path,
            "raw_refs": list(self.raw_refs),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextCheckpoint":
        return cls(
            run_id=str(data["run_id"]),
            checkpoint_id=str(data["checkpoint_id"]),
            created_at=float(data["created_at"]),
            ledger_path=str(data["ledger_path"]),
            memory_path=str(data["memory_path"]),
            raw_refs=[str(item) for item in data.get("raw_refs", [])],
            metadata=dict(data.get("metadata") or {}),
        )


class ContextCheckpointStore:
    """Create and restore context-only checkpoints."""

    def __init__(
        self,
        ledger_store: ContextLedgerStore,
        *,
        root_dir: str | Path | None = None,
    ) -> None:
        self.ledger_store = ledger_store
        self.root_dir = Path(root_dir) if root_dir else ledger_store.root_dir / "checkpoints"
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        run_id: str,
        *,
        checkpoint_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> ContextCheckpoint:
        checkpoint_id = checkpoint_id or f"ctx_{int(time.time() * 1000)}"
        target_dir = self.root_dir / _safe_name(run_id) / _safe_name(checkpoint_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        ledger_path = self.ledger_store.path_for(run_id)
        memory_path = self.ledger_store.memory_path_for(run_id)
        copied_ledger = target_dir / "ledger.json"
        copied_memory = target_dir / "MEMORY.md"
        shutil.copy2(ledger_path, copied_ledger)
        if memory_path.exists():
            shutil.copy2(memory_path, copied_memory)

        raw_refs = self._copy_raw_refs(run_id, target_dir)
        checkpoint = ContextCheckpoint(
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            created_at=time.time(),
            ledger_path=str(copied_ledger),
            memory_path=str(copied_memory),
            raw_refs=raw_refs,
            metadata=dict(metadata or {}),
        )
        (target_dir / "checkpoint.json").write_text(
            json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return checkpoint

    def restore(self, checkpoint: ContextCheckpoint | str | Path) -> None:
        if not isinstance(checkpoint, ContextCheckpoint):
            data = json.loads(Path(checkpoint).read_text(encoding="utf-8"))
            checkpoint = ContextCheckpoint.from_dict(data)
        shutil.copy2(checkpoint.ledger_path, self.ledger_store.path_for(checkpoint.run_id))
        memory_src = Path(checkpoint.memory_path)
        if memory_src.exists():
            shutil.copy2(memory_src, self.ledger_store.memory_path_for(checkpoint.run_id))
        checkpoint_dir = Path(checkpoint.ledger_path).parent
        raw_dir = checkpoint_dir / "raw"
        if raw_dir.exists():
            target_raw = self.ledger_store.root_dir / "raw"
            for path in raw_dir.rglob("*"):
                if path.is_file():
                    rel = path.relative_to(raw_dir)
                    target = target_raw / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)

    def _copy_raw_refs(self, run_id: str, target_dir: Path) -> List[str]:
        ledger = self.ledger_store.load_or_create(run_id)
        refs = [
            summary.raw_ref
            for summary in ledger.tool_summaries
            if summary.raw_ref
        ]
        copied: List[str] = []
        raw_target = target_dir / "raw"
        for ref in refs:
            source = self.ledger_store.root_dir / ref
            if not source.exists():
                continue
            target = raw_target / Path(ref).relative_to("raw")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(ref)
        return copied


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value) or "item"
