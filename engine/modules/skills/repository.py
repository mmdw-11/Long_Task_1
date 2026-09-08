"""过程性技能文件仓库。

本模块负责把 Markdown 技能、生命周期状态、版本历史和灰度发布策略落到本地文件。
当前版本保持原有 metadata/markdown 结构，历史版本额外归档到 versions/，方便审计与回滚。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._types import Skill, SkillRecord, SkillStatus


class SkillRepository:
    """持久化技能记录，并提供状态流转、版本回滚和灰度发布能力。"""

    def __init__(self, root_dir: str | Path = "runs/skills") -> None:
        self.root_dir = Path(root_dir)
        self.meta_dir = self.root_dir / "metadata"
        self.body_dir = self.root_dir / "markdown"
        self.version_dir = self.root_dir / "versions"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.body_dir.mkdir(parents=True, exist_ok=True)
        self.version_dir.mkdir(parents=True, exist_ok=True)

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
        owner_user_id: Optional[str] = None,
        visibility: str = "private",
        source_type: str = "manual",
        validation_status: str = "pending",
        package_sha256: str = "",
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
                "owner_user_id": owner_user_id,
                "visibility": visibility,
                "source_type": source_type,
                "validation_status": validation_status,
                "package_sha256": package_sha256,
            }
        )
        return self.save(record, versioned=False)

    def save(self, record: SkillRecord | Skill, *, versioned: bool = True) -> SkillRecord:
        if isinstance(record, Skill):
            record = self._from_legacy_skill(record)
        now = _utc_now()
        existing = self.get(record.id) if record.id and self.exists(record.id) else None
        if existing is not None:
            if versioned:
                self._archive_version(existing)
            record.created_at = existing.created_at
            record.version = existing.version + 1 if versioned else existing.version
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

    def get(
        self,
        skill_id: str,
        version: Optional[str | int] = None,
        *,
        status: Optional[SkillStatus | str] = None,
    ) -> SkillRecord:
        meta_path = self._meta_path(skill_id)
        if not meta_path.exists():
            raise KeyError(f"skill {skill_id!r} not found")
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        body_path = Path(payload.get("content_path") or self._body_path(skill_id))
        payload["content"] = body_path.read_text(encoding="utf-8")
        record = SkillRecord.from_dict(payload)
        if version is not None and str(record.manifest.version) != str(version) and record.version != _parse_int(version):
            raise KeyError(f"skill {skill_id!r} version {version!r} not found")
        if status is not None:
            wanted = status if isinstance(status, SkillStatus) else SkillStatus(str(status))
            if record.status != wanted:
                raise KeyError(f"skill {skill_id!r} with status {wanted.value!r} not found")
        return record

    def list(self, *, status: Optional[SkillStatus | str] = None) -> List[SkillRecord]:
        records = [self.get(path.stem) for path in sorted(self.meta_dir.glob("*.json"))]
        if status is not None:
            wanted = status if isinstance(status, SkillStatus) else SkillStatus(str(status))
            records = [item for item in records if item.status == wanted]
        return sorted(records, key=lambda item: item.updated_at, reverse=True)

    def list_versions(self, skill_id: str) -> List[SkillRecord]:
        """列出某个技能的全部版本，包含当前版本和已归档历史版本。"""
        if not self.exists(skill_id):
            raise KeyError(f"skill {skill_id!r} not found")
        records = []
        for path in sorted(self._version_root(skill_id).glob("v*.json")):
            records.append(SkillRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        records.append(self.get(skill_id))
        return sorted(records, key=lambda item: item.version, reverse=True)

    def get_version(self, skill_id: str, version: int | str) -> SkillRecord:
        """读取指定技能版本；当前版本和历史版本使用同一个返回结构。"""
        version_int = _parse_int(version)
        current = self.get(skill_id)
        if current.version == version_int or str(current.manifest.version) == str(version):
            return current
        path = self._version_path(skill_id, version_int)
        if not path.exists():
            raise KeyError(f"skill {skill_id!r} version {version!r} not found")
        return SkillRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def rollback(
        self,
        skill_id: str,
        version: int | str,
        *,
        approved_by: str,
        reason: str = "",
    ) -> SkillRecord:
        """把指定历史版本恢复为新的当前版本，保留回滚操作审计信息。"""
        if not approved_by.strip():
            raise ValueError("approved_by is required")
        version_int = _parse_int(version)
        target = self.get_version(skill_id, version)
        current = self.get(skill_id)
        restored = SkillRecord.from_dict(
            {
                **target.to_dict(),
                "version": current.version,
                "approved_by": approved_by,
                "metadata": {
                    **target.metadata,
                    "rollback_from_version": current.version,
                    "rollback_to_version": version_int,
                    "rollback_by": approved_by,
                    **({"rollback_reason": reason} if reason else {}),
                },
            }
        )
        return self.save(restored)

    def set_rollout(self, skill_id: str, percent: int, *, approved_by: str) -> SkillRecord:
        """设置技能灰度比例，0 表示不参与检索，100 表示全量生效。"""
        if not approved_by.strip():
            raise ValueError("approved_by is required")
        if percent < 0 or percent > 100:
            raise ValueError("rollout_percent must be between 0 and 100")
        record = self.get(skill_id)
        if record.status != SkillStatus.PUBLISHED:
            raise ValueError("only published skills can change rollout percent")
        record.metadata = {
            **record.metadata,
            "rollout_percent": percent,
            "rollout_updated_by": approved_by,
            "rollout_updated_at": _utc_now(),
        }
        return self.save(record, versioned=False)

    def publish(
        self,
        skill_id: str,
        version: Optional[str | int] = None,
        *,
        approved_by: str,
    ) -> SkillRecord:
        """兼容旧调用：仓库层直接发布已验证技能。"""
        if not approved_by.strip():
            raise ValueError("approved_by is required")
        record = self.get(skill_id, version)
        if record.status != SkillStatus.VALIDATED:
            raise ValueError("only validated skills can be published")
        return self.transition(
            skill_id,
            SkillStatus.PUBLISHED,
            approved_by=approved_by,
            metadata={"published_by": approved_by},
        )

    def retire(self, skill_id: str, *, reason: str = "") -> SkillRecord:
        """兼容旧调用：仓库层直接退役技能。"""
        metadata = {"retired_reason": reason} if reason else None
        retired = self.transition(skill_id, SkillStatus.RETIRED, metadata=metadata)
        self.delete(skill_id)
        return retired

    def is_rollout_enabled(self, skill: SkillRecord, *, query: str = "", node: str = "") -> bool:
        """按技能、节点和查询稳定散列，实现无状态、可复现的灰度命中判断。"""
        raw_percent = skill.metadata.get("rollout_percent", 100)
        try:
            percent = int(raw_percent)
        except (TypeError, ValueError):
            percent = 100
        if percent <= 0:
            return False
        if percent >= 100:
            return True
        seed = f"{skill.id}:{skill.version}:{node}:{query}".encode("utf-8", errors="ignore")
        bucket = int(hashlib.sha256(seed).hexdigest()[:8], 16) % 100
        return bucket < percent

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
        # Lifecycle/metadata changes do not create a content release. Publishing is
        # the sole transition that advances the public version number.
        return self.save(record, versioned=status == SkillStatus.PUBLISHED)

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

    def _archive_version(self, record: SkillRecord) -> None:
        # 历史版本以完整 JSON 保存，避免回滚时依赖当前 markdown 文件。
        path = self._version_path(record.id, record.version)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        path.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _version_root(self, skill_id: str) -> Path:
        clean = self._clean_id(skill_id)
        if not clean:
            raise ValueError("skill_id is required")
        return self.version_dir / clean

    def _version_path(self, skill_id: str, version: int) -> Path:
        if version <= 0:
            raise ValueError("version must be positive")
        return self._version_root(skill_id) / f"v{version}.json"

    def _from_legacy_skill(self, skill: Skill) -> SkillRecord:
        # 旧版 SkillManifest 的筛选字段放入 metadata，避免改变当前 API 返回结构。
        manifest = skill.manifest
        legacy_metadata = {
            "legacy_manifest": {
                "version": manifest.version,
                "task_types": list(manifest.task_types),
                "applicable_nodes": list(manifest.applicable_nodes),
            }
        }
        return SkillRecord.from_dict(
            {
                "id": manifest.skill_id,
                "name": manifest.name,
                "status": manifest.status.value,
                "content": skill.content,
                "description": manifest.description,
                "tags": manifest.tags,
                "metadata": {**manifest.metadata, **legacy_metadata},
                "approved_by": manifest.approved_by,
            }
        )

    @staticmethod
    def _clean_id(raw: str) -> str:
        return "".join(ch for ch in raw.strip() if ch.isalnum() or ch in {"-", "_"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_int(value: Any) -> int:
    raw = str(value).strip()
    if not raw:
        raise ValueError("version must be positive")
    head = raw.split(".", 1)[0]
    parsed = int(head)
    if parsed <= 0:
        raise ValueError("version must be positive")
    return parsed
