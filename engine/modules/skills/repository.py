"""文件型 SkillRepo：以状态分区、版本目录和原子索引切换保证可审计发布。"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from ._types import Skill, SkillManifest, SkillStatus


class SkillRepository:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        for status in SkillStatus:
            (self.root_dir / status.value).mkdir(exist_ok=True)
        self.index_path = self.root_dir / "index.json"
        if not self.index_path.exists():
            self._write_json(self.index_path, {"published": {}})

    def save(self, skill: Skill, *, overwrite: bool = False) -> Path:
        manifest = skill.manifest
        target = self._version_dir(manifest.status, manifest.skill_id, manifest.version)
        if target.exists() and not overwrite:
            raise FileExistsError(f"skill version already exists: {manifest.skill_id}@{manifest.version}")
        target.mkdir(parents=True, exist_ok=True)
        manifest.updated_at = time.time()
        self._write_text(target / "SKILL.md", skill.content.rstrip() + "\n")
        self._write_json(target / "manifest.json", manifest.to_dict())
        return target

    def get(
        self,
        skill_id: str,
        version: Optional[str] = None,
        *,
        status: SkillStatus = SkillStatus.PUBLISHED,
    ) -> Skill:
        if version is None:
            if status != SkillStatus.PUBLISHED:
                raise ValueError("version is required for non-published skills")
            version = self._read_index().get("published", {}).get(skill_id)
            if not version:
                raise KeyError(f"published skill not found: {skill_id}")
        path = self._version_dir(status, skill_id, version)
        if not path.exists():
            raise KeyError(f"skill not found: {skill_id}@{version} ({status.value})")
        manifest = SkillManifest.from_dict(json.loads((path / "manifest.json").read_text(encoding="utf-8")))
        return Skill(manifest, (path / "SKILL.md").read_text(encoding="utf-8"))

    def list(self, *, status: SkillStatus = SkillStatus.PUBLISHED) -> List[Skill]:
        if status == SkillStatus.PUBLISHED:
            result = []
            for skill_id, version in sorted(self._read_index().get("published", {}).items()):
                result.append(self.get(skill_id, version, status=status))
            return result
        result: List[Skill] = []
        base = self.root_dir / status.value
        for manifest_path in sorted(base.glob("*/*/manifest.json")):
            manifest = SkillManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
            result.append(Skill(manifest, (manifest_path.parent / "SKILL.md").read_text(encoding="utf-8")))
        return result

    def publish(self, skill_id: str, version: str, *, approved_by: str) -> Skill:
        if not approved_by.strip():
            raise ValueError("approved_by is required")
        candidate = self.get(skill_id, version, status=SkillStatus.VALIDATED)
        manifest = SkillManifest.from_dict(candidate.manifest.to_dict())
        manifest.status = SkillStatus.PUBLISHED
        manifest.approved_by = approved_by.strip()
        published = Skill(manifest, candidate.content)
        self.save(published, overwrite=True)
        index = self._read_index()
        index.setdefault("published", {})[skill_id] = version
        self._write_json(self.index_path, index)
        return published

    def retire(self, skill_id: str, *, reason: str = "") -> Skill:
        current = self.get(skill_id)
        retired_manifest = SkillManifest.from_dict(current.manifest.to_dict())
        retired_manifest.status = SkillStatus.RETIRED
        retired_manifest.metrics = {**retired_manifest.metrics, "retire_reason": reason}
        retired = Skill(retired_manifest, current.content)
        self.save(retired, overwrite=True)
        index = self._read_index()
        index.get("published", {}).pop(skill_id, None)
        self._write_json(self.index_path, index)
        return retired

    def rollback(self, skill_id: str, version: str, *, approved_by: str) -> Skill:
        """Atomically point the published index back to an existing published version."""
        if not approved_by.strip():
            raise ValueError("approved_by is required")
        target = self.get(skill_id, version, status=SkillStatus.PUBLISHED)
        index = self._read_index()
        index.setdefault("published", {})[skill_id] = version
        self._write_json(self.index_path, index)
        return target

    def _version_dir(self, status: SkillStatus, skill_id: str, version: str) -> Path:
        # SkillManifest validates identifiers; version validation is repeated by construction.
        SkillManifest(skill_id=skill_id, name=skill_id, version=version, status=status)
        return self.root_dir / status.value / skill_id / version

    def _read_index(self) -> Dict[str, object]:
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read skill index: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError("skill index must be a JSON object")
        return data

    @staticmethod
    def _write_json(path: Path, data: object) -> None:
        SkillRepository._write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
            os.replace(temp_name, path)
        except Exception:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
            raise
