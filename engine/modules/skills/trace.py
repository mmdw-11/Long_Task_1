"""技能演化轨迹存储：脱敏、追加写 JSONL，并保留技能版本与评测证据。"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..security import SensitiveDataRedactor
from .retrieval import SKILL_CONTEXT_KEY


class SkillTraceStore:
    def __init__(self, root_dir: str | Path, *, redactor: Optional[SensitiveDataRedactor] = None) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.redactor = redactor or SensitiveDataRedactor()

    def record(
        self,
        event_type: str,
        *,
        state: Dict[str, Any],
        node: str = "",
        step: int = 0,
        payload: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not event_type.strip():
            raise ValueError("event_type cannot be empty")
        run_id = str(state.get("run_id") or state.get("task_id") or "default-run")
        safe_run = _safe_name(run_id)
        redacted = self.redactor.redact(payload).redacted
        event = {
            "event_id": uuid.uuid4().hex,
            "event_type": event_type,
            "run_id": run_id,
            "task_type": str(state.get("task_type") or ""),
            "node": node,
            "step": step,
            "timestamp": time.time(),
            "skills": list(state.get(SKILL_CONTEXT_KEY) or []),
            "evaluation": dict(state.get("__evaluation__") or {}),
            "payload": redacted,
            "metadata": self.redactor.redact(metadata or {}).redacted,
        }
        path = self.root_dir / safe_run / "trace.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
        return event

    def read(self, run_id: str) -> List[Dict[str, Any]]:
        path = self.root_dir / _safe_name(run_id) / "trace.jsonl"
        if not path.exists():
            return []
        events = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid trace JSON at line {number}: {exc}") from exc
            if isinstance(item, dict):
                events.append(item)
        return events


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in value.strip())
    return safe or "default-run"

