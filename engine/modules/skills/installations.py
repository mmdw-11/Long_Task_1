"""Per-user installation state for shared skills."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class SkillInstallationStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def list(self, user_id: str) -> List[Dict[str, Any]]:
        path = self._path(user_id)
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        return list(payload) if isinstance(payload, list) else []

    def installed_ids(self, user_id: str) -> set[str]:
        return {str(item.get("skill_id")) for item in self.list(user_id) if item.get("enabled", True)}

    def install(self, user_id: str, skill_id: str) -> tuple[Dict[str, Any], bool]:
        records = self.list(user_id)
        for item in records:
            if item.get("skill_id") == skill_id:
                changed = not item.get("enabled", True)
                item["enabled"] = True
                if changed:
                    self._save(user_id, records)
                return item, changed
        item = {"user_id": user_id, "skill_id": skill_id, "installed_at": _utc_now(), "enabled": True}
        records.append(item)
        self._save(user_id, records)
        return item, True

    def uninstall(self, user_id: str, skill_id: str) -> bool:
        records = self.list(user_id)
        remaining = [item for item in records if item.get("skill_id") != skill_id]
        if len(remaining) == len(records):
            return False
        self._save(user_id, remaining)
        return True

    def _save(self, user_id: str, records: List[Dict[str, Any]]) -> None:
        self._path(user_id).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    def _path(self, user_id: str) -> Path:
        safe = "".join(ch for ch in user_id if ch.isalnum() or ch in {"-", "_"}) or "local-user"
        return self.root_dir / f"{safe}.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
