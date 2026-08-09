"""技能运行轨迹追加日志。

执行钩子把节点事件写入这里，技能演化服务可以基于真实轨迹生成候选技能。
append/list 是当前接口；record/read 用于兼容早期调用。
"""

from __future__ import annotations

import json
import re
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

    def record(
        self,
        event_type: str,
        *,
        state: Dict[str, Any],
        node: str = "",
        step: int = 0,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """兼容旧接口：从 state.run_id 提取运行号并写入事件。"""
        run_id = str(state.get("run_id") or "default")
        self.append(
            SkillTraceEvent(
                run_id=run_id,
                node=node,
                event=event_type,
                step=step,
                payload={"state": dict(state), **(payload or {})},
            )
        )

    def read(self, run_id: str) -> List[Dict[str, Any]]:
        """兼容旧接口：返回字典形式，并补 event_type 字段。"""
        records = []
        for event in self.list(run_id):
            payload = _redact_value(event.to_dict())
            payload["event_type"] = event.event
            payload["skills"] = _normalize_skills(event.payload.get("skills") or event.payload.get("matches"))
            records.append(payload)
            # 新版 skill_retrieved 是节点开始时的检索事件；旧版前端按 node_start/step_start 展示。
            if event.event == "skill_retrieved":
                node_start = dict(payload)
                node_start["event_type"] = "node_start"
                records.append(node_start)
                step_start = dict(payload)
                step_start["event_type"] = "step_start"
                records.append(step_start)
        return records

    def path_for(self, run_id: str) -> Path:
        clean = "".join(ch for ch in run_id.strip() if ch.isalnum() or ch in {"-", "_"})
        if not clean:
            raise ValueError("run_id is required")
        return self.root_dir / f"{clean}.jsonl"


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"(api[_-]?key\s*[=:]\s*)[A-Za-z0-9_\-]{8,}", r"\1REDACTED", value)
    return value


def _normalize_skills(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, str):
        matches = re.findall(r"\(([^(),\s]+),\s*score=([0-9.]+)\)", raw)
        return [
            {"skill_id": skill_id, "score": float(score)}
            for skill_id, score in matches
        ]
    return []
