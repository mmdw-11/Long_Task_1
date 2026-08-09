"""Append-only skill trace store.

Execution hooks write compact node events here so skill evolution can inspect
what actually happened without scraping logs or frontend state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SkillTraceEvent:
    run_id: str
    node: str
    event: str
    payload: Dict[str, Any] = field(default_factory=dict)
    step: int = 0
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "node": self.node,
            "event": self.event,
            "payload": self.payload,
            "step": self.step,
            "ts": self.ts,
        }


class SkillTraceStore:
    """Write and read per-run JSONL traces for skill generation."""

    def __init__(self, root_dir: str | Path = "runs/skill_traces") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def append(self, event: SkillTraceEvent) -> None:
        path = self.path_for(event.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.to_dict(), ensure_ascii=False, default=str) + "\n")

    def list(self, run_id: str) -> List[SkillTraceEvent]:
        path = self.path_for(run_id)
        if not path.exists():
            return []
        events: List[SkillTraceEvent] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                data = json.loads(line)
                events.append(
                    SkillTraceEvent(
                        run_id=str(data.get("run_id") or run_id),
                        node=str(data.get("node") or ""),
                        event=str(data.get("event") or ""),
                        payload=dict(data.get("payload") or {}),
                        step=int(data.get("step") or 0),
                        ts=str(data.get("ts") or ""),
                    )
                )
        return events

    def path_for(self, run_id: str) -> Path:
        clean = "".join(ch for ch in run_id.strip() if ch.isalnum() or ch in {"-", "_"})
        if not clean:
            raise ValueError("run_id is required")
        return self.root_dir / f"{clean}.jsonl"
