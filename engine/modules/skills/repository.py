"""File-backed repository for Markdown procedural skills.

Each skill has a JSON metadata record and a Markdown body. Keeping the body as
Markdown makes review, approval, rollback, and future migration straightforward.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._types import SkillRecord, SkillStatus


class SkillRepository:
    """Persist and transition skill records with simple lifecycle guards."""

    def __init__(self, root_dir: str | Path = "runs/skills") -> None:
        self.root_dir = Path(root_dir)
        self.meta_dir = self.root_dir / "metadata"
        self.body_dir = self.root_dir / "markdown"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.body_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        *,
        name: str,
        content: str,
        status: SkillStatus = SkillStatus.DRAFT,
        description: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_run_ids: Optional[List[str]] = None,
        skill_id: Optional[str] = None,
    ) -> SkillRecord:
        record = SkillRecord.from_dict(
            {
                "id": self._clean_id(skill_id or "") or f"skill-{uuid.uuid4().hex[:12]}",
                "name": name,
                "content": content,
                "status": status.value,
                "description": description,
                "tags": tags or [],
                "metadata": metadata or {},
                "source_run_ids": source_run_ids or [],
            }
        )
        return self.save(record)

    def save(self, record: SkillRecord) -> SkillRecord:
        now = _utc_now()
        existing = self.get(record.id) if record.id and self.exists(record.id) else None
        if existing is not None:
            record.created_at = existing.created_at
            record.version = existing.version + 1
        else:
            record.created_at = record.created_at or now
            record.version = max(1, record.version)
        record.updated_at = now

        self._body_path(record.id).write_text(record.content, encoding="utf-8")
        payload = record.to_dict()
        payload["content_path"] = str(self._body_path(record.id))
        payload.pop("content", None)
        self._meta_path(record.id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return record

    def get(self, skill_id: str) -> SkillRecord:
        meta_path = self._meta_path(skill_id)
        if not meta_path.exists():
            raise KeyError(f"skill {skill_id!r} not found")
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        body_path = Path(payload.get("content_path") or self._body_path(skill_id))
        payload["content"] = body_path.read_text(encoding="utf-8")
        return SkillRecord.from_dict(payload)

    def list(self, *, status: Optional[SkillStatus | str] = None) -> List[SkillRecord]:
        records = [self.get(path.stem) for path in sorted(self.meta_dir.glob("*.json"))]
        if status is not None:
            wanted = status if isinstance(status, SkillStatus) else SkillStatus(str(status))
            records = [item for item in records if item.status == wanted]
        return sorted(records, key=lambda item: item.updated_at, reverse=True)

    def exists(self, skill_id: str) -> bool:
        return self._meta_path(skill_id).exists()

    def transition(
        self,
        skill_id: str,
        status: SkillStatus,
        *,
        approved_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SkillRecord:
        record = self.get(skill_id)
        self._validate_transition(record.status, status)
        record.status = status
        if approved_by:
            record.approved_by = approved_by
        if metadata:
            record.metadata = {**record.metadata, **metadata}
        return self.save(record)

    def delete(self, skill_id: str) -> None:
        if not self.exists(skill_id):
            raise KeyError(f"skill {skill_id!r} not found")
        self._meta_path(skill_id).unlink()
        body = self._body_path(skill_id)
        if body.exists():
            body.unlink()

    def _validate_transition(self, current: SkillStatus, target: SkillStatus) -> None:
        allowed = {
            SkillStatus.DRAFT: {SkillStatus.CANDIDATE, SkillStatus.REJECTED},
            SkillStatus.CANDIDATE: {SkillStatus.VALIDATED, SkillStatus.REJECTED},
            SkillStatus.VALIDATED: {SkillStatus.PUBLISHED, SkillStatus.REJECTED},
            SkillStatus.PUBLISHED: {SkillStatus.RETIRED},
            SkillStatus.REJECTED: {SkillStatus.CANDIDATE},
            SkillStatus.RETIRED: {SkillStatus.PUBLISHED},
        }
        if current == target:
            return
        if target not in allowed.get(current, set()):
            raise ValueError(f"cannot transition skill from {current.value} to {target.value}")

    def _meta_path(self, skill_id: str) -> Path:
        clean = self._clean_id(skill_id)
        if not clean:
            raise ValueError("skill_id is required")
        return self.meta_dir / f"{clean}.json"

    def _body_path(self, skill_id: str) -> Path:
        clean = self._clean_id(skill_id)
        if not clean:
            raise ValueError("skill_id is required")
        return self.body_dir / f"{clean}.md"

    @staticmethod
    def _clean_id(raw: str) -> str:
        return "".join(ch for ch in raw.strip() if ch.isalnum() or ch in {"-", "_"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
