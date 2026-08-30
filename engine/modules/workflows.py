"""工作流与运行记录的文件型持久化模块。

该模块把面向产品的状态从内存 Orchestrator 中拆出来，先用 JSON 文件提供稳定
存储接口，后续迁移数据库时可以尽量不改 REST 和业务调用层。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


WORKFLOW_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1


@dataclass
class WorkflowRecord:
    """A named graph orchestration snapshot that can be loaded by the backend."""

    id: str
    name: str
    graph: Dict[str, Any]
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    version: int = WORKFLOW_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "graph": self.graph,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowRecord":
        if not isinstance(data, dict):
            raise ValueError("workflow payload must be an object")
        graph = data.get("graph")
        if not isinstance(graph, dict):
            raise ValueError("workflow.graph must be an object")
        workflow_id = _clean_id(str(data.get("id") or ""))
        name = str(data.get("name") or "").strip()
        if not name:
            raise ValueError("workflow.name is required")
        return cls(
            id=workflow_id or f"workflow-{uuid.uuid4().hex[:12]}",
            name=name,
            description=str(data.get("description") or ""),
            tags=[str(item) for item in data.get("tags") or []],
            metadata=dict(data.get("metadata") or {}),
            graph=graph,
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
            version=int(data.get("version") or WORKFLOW_SCHEMA_VERSION),
        )


@dataclass
class RunRecord:
    """持久化的运行生命周期记录，用于轮询、重试、取消和评估。"""

    id: str
    workflow_id: Optional[str]
    status: str
    input: Dict[str, Any]
    recursion_limit: int
    state: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    parent_run_id: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    finished_at: Optional[str] = None
    canceled_at: Optional[str] = None
    version: int = RUN_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "status": self.status,
            "input": self.input,
            "recursion_limit": self.recursion_limit,
            "state": self.state,
            "events": list(self.events),
            "error": self.error,
            "parent_run_id": self.parent_run_id,
            "retry_count": self.retry_count,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "canceled_at": self.canceled_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunRecord":
        if not isinstance(data, dict):
            raise ValueError("run payload must be an object")
        run_id = _clean_id(str(data.get("id") or ""))
        if not run_id:
            raise ValueError("run.id is required")
        return cls(
            id=run_id,
            workflow_id=data.get("workflow_id"),
            status=str(data.get("status") or "created"),
            input=dict(data.get("input") or {}),
            recursion_limit=int(data.get("recursion_limit") or 50),
            state=dict(data.get("state") or {}),
            events=list(data.get("events") or []),
            error=data.get("error"),
            parent_run_id=data.get("parent_run_id"),
            retry_count=int(data.get("retry_count") or 0),
            metadata=dict(data.get("metadata") or {}),
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
            finished_at=data.get("finished_at"),
            canceled_at=data.get("canceled_at"),
            version=int(data.get("version") or RUN_SCHEMA_VERSION),
        )


class WorkflowStore:
    """文件型工作流仓库，支持主版本与历史版本归档。"""

    def __init__(self, root_dir: str | Path = "runs/workflows") -> None:
        self.root_dir = Path(root_dir)
        self.version_dir = self.root_dir / "versions"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.version_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: WorkflowRecord) -> WorkflowRecord:
        existing = self.get(record.id) if self.exists(record.id) else None
        now = _utc_now()
        if existing is not None:
            self._archive_version(existing)
            record.created_at = existing.created_at
            record.version = existing.version + 1
        else:
            record.version = max(1, record.version)
        record.updated_at = now
        self._path(record.id).write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return record

    def save_draft(self, record: WorkflowRecord) -> WorkflowRecord:
        """保存应用工作流草稿，不创建或递增发布版本。"""
        existing = self.get(record.id) if self.exists(record.id) else None
        if existing is not None:
            record.created_at = existing.created_at
            record.version = existing.version
        record.updated_at = _utc_now()
        self._write_current(record)
        return record

    def publish(self, workflow_id: str) -> WorkflowRecord:
        """将当前应用工作流草稿保存为一个新的不可变发布快照。"""
        current = self.get(workflow_id)
        published = [item.version for item in self._published_versions(workflow_id)]
        next_version = max(published, default=0) + 1
        now = _utc_now()
        snapshot = WorkflowRecord.from_dict(
            {
                **current.to_dict(),
                "version": next_version,
                "updated_at": now,
                "metadata": {
                    **current.metadata,
                    "published_snapshot": True,
                    "published_at": now,
                },
            }
        )
        # 旧版本逻辑可能已经留下同编号的“草稿历史”文件。这类文件没有
        # published_snapshot 标记，应由新的正式发布快照替换，否则发布成功
        # 后版本列表会因为过滤旧草稿而显示为空。
        self._archive_version(snapshot, overwrite_unpublished=True)
        current.version = next_version
        current.updated_at = now
        current.metadata = {
            **current.metadata,
            "published_version": next_version,
            "published_at": now,
        }
        self._write_current(current)
        return current

    def activate_version(self, workflow_id: str, version: int) -> WorkflowRecord:
        """激活既有发布快照，不创建新版本。"""
        path = self._version_path(workflow_id, version)
        if not path.exists():
            raise KeyError(f"workflow {workflow_id!r} version {version!r} not found")
        target = WorkflowRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if not target.metadata.get("published_snapshot"):
            raise ValueError("只能激活已发布的工作流版本")
        current = self.get(workflow_id)
        activated = WorkflowRecord.from_dict(
            {
                **target.to_dict(),
                "created_at": current.created_at,
                "updated_at": _utc_now(),
                "metadata": {
                    **target.metadata,
                    "application_id": current.metadata.get("application_id", target.metadata.get("application_id")),
                    "published_version": version,
                    "active_version": version,
                },
            }
        )
        self._write_current(activated)
        return activated

    def create(
        self,
        *,
        name: str,
        graph: Dict[str, Any],
        description: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
    ) -> WorkflowRecord:
        record = WorkflowRecord.from_dict(
            {
                "id": workflow_id or f"workflow-{uuid.uuid4().hex[:12]}",
                "name": name,
                "description": description,
                "tags": tags or [],
                "metadata": metadata or {},
                "graph": graph,
            }
        )
        return self.save(record)

    def get(self, workflow_id: str) -> WorkflowRecord:
        path = self._path(workflow_id)
        if not path.exists():
            raise KeyError(f"workflow {workflow_id!r} not found")
        return WorkflowRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def exists(self, workflow_id: str) -> bool:
        return self._path(workflow_id).exists()

    def list(self) -> List[WorkflowRecord]:
        records = [self.get(path.stem) for path in sorted(self.root_dir.glob("*.json"))]
        return sorted(records, key=lambda item: item.updated_at, reverse=True)

    def delete(self, workflow_id: str) -> None:
        path = self._path(workflow_id)
        if not path.exists():
            raise KeyError(f"workflow {workflow_id!r} not found")
        path.unlink()

    def list_versions(self, workflow_id: str) -> List[WorkflowRecord]:
        if not self.exists(workflow_id):
            raise KeyError(f"workflow {workflow_id!r} not found")
        version_root = self._version_root(workflow_id)
        current = self.get(workflow_id)
        if "application" in current.tags:
            return sorted(self._published_versions(workflow_id), key=lambda item: item.version, reverse=True)
        records = []
        for path in sorted(version_root.glob("v*.json")):
            records.append(WorkflowRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        records.append(self.get(workflow_id))
        return sorted(records, key=lambda item: item.version, reverse=True)

    def get_version(self, workflow_id: str, version: int) -> WorkflowRecord:
        path = self._version_path(workflow_id, version)
        if path.exists():
            return WorkflowRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        current = self.get(workflow_id)
        if current.version == version:
            return current
        raise KeyError(f"workflow {workflow_id!r} version {version!r} not found")

    def rollback(self, workflow_id: str, version: int) -> WorkflowRecord:
        target = self.get_version(workflow_id, version)
        current = self.get(workflow_id)
        restored = WorkflowRecord.from_dict(
            {
                **target.to_dict(),
                "version": current.version,
                "metadata": {
                    **target.metadata,
                    "rollback_from_version": current.version,
                    "rollback_to_version": version,
                },
            }
        )
        return self.save(restored)

    def _archive_version(self, record: WorkflowRecord, *, overwrite_unpublished: bool = False) -> None:
        path = self._version_path(record.id, record.version)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not overwrite_unpublished:
                return
            existing = WorkflowRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if existing.metadata.get("published_snapshot"):
                return
        path.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _published_versions(self, workflow_id: str) -> List[WorkflowRecord]:
        version_root = self._version_root(workflow_id)
        records = [
            WorkflowRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(version_root.glob("v*.json"))
        ]
        return [item for item in records if item.metadata.get("published_snapshot")]

    def _write_current(self, record: WorkflowRecord) -> None:
        self._path(record.id).write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _path(self, workflow_id: str) -> Path:
        clean = _clean_id(workflow_id)
        if not clean:
            raise ValueError("workflow_id is required")
        return self.root_dir / f"{clean}.json"

    def _version_root(self, workflow_id: str) -> Path:
        clean = _clean_id(workflow_id)
        if not clean:
            raise ValueError("workflow_id is required")
        return self.version_dir / clean

    def _version_path(self, workflow_id: str, version: int) -> Path:
        if version <= 0:
            raise ValueError("version must be positive")
        return self._version_root(workflow_id) / f"v{version}.json"


class RunStore:
    """File-backed run repository for synchronous and background executions."""

    def __init__(self, root_dir: str | Path = "runs/executions") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        *,
        input: Dict[str, Any],
        recursion_limit: int,
        workflow_id: Optional[str] = None,
        run_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        retry_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RunRecord:
        record = RunRecord(
            id=_clean_id(run_id or "") or f"run-{uuid.uuid4().hex[:12]}",
            workflow_id=workflow_id,
            status="created",
            input=dict(input or {}),
            recursion_limit=recursion_limit,
            parent_run_id=parent_run_id,
            retry_count=retry_count,
            metadata=dict(metadata or {}),
        )
        return self.save(record)

    def save(self, record: RunRecord) -> RunRecord:
        record.updated_at = _utc_now()
        self._path(record.id).write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return record

    def get(self, run_id: str) -> RunRecord:
        path = self._path(run_id)
        if not path.exists():
            raise KeyError(f"run {run_id!r} not found")
        return RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self, *, workflow_id: Optional[str] = None) -> List[RunRecord]:
        records = [self.get(path.stem) for path in sorted(self.root_dir.glob("*.json"))]
        if workflow_id is not None:
            records = [item for item in records if item.workflow_id == workflow_id]
        return sorted(records, key=lambda item: item.updated_at, reverse=True)

    def mark_cancel_requested(self, run_id: str, *, reason: str = "") -> RunRecord:
        record = self.get(run_id)
        if record.status in {"succeeded", "failed", "canceled"}:
            return record
        record.status = "cancel_requested" if record.status == "running" else "canceled"
        record.canceled_at = _utc_now()
        record.finished_at = record.finished_at or record.canceled_at
        record.metadata = {**record.metadata, "cancel_reason": reason}
        return self.save(record)

    def retry(self, run_id: str) -> RunRecord:
        source = self.get(run_id)
        return self.create(
            input=source.input,
            recursion_limit=source.recursion_limit,
            workflow_id=source.workflow_id,
            parent_run_id=source.id,
            retry_count=source.retry_count + 1,
            metadata={"retry_from": source.id},
        )

    def metrics(self) -> Dict[str, Any]:
        records = self.list()
        status_counts: Dict[str, int] = {}
        for record in records:
            status_counts[record.status] = status_counts.get(record.status, 0) + 1
        return {
            "total": len(records),
            "status_counts": status_counts,
            "latest_run_id": records[0].id if records else None,
        }

    def _path(self, run_id: str) -> Path:
        clean = _clean_id(run_id)
        if not clean:
            raise ValueError("run_id is required")
        return self.root_dir / f"{clean}.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_id(raw: str) -> str:
    return "".join(ch for ch in raw.strip() if ch.isalnum() or ch in {"-", "_"})
