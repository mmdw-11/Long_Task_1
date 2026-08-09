"""Backend persistence primitives for saved workflows and execution runs.

This module keeps product-facing state outside the in-memory Orchestrator so
the REST service can support save/load/list flows before a database is added.
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
    """Persisted execution lifecycle data for polling and later evaluation."""

    id: str
    workflow_id: Optional[str]
    status: str
    input: Dict[str, Any]
    recursion_limit: int
    state: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    finished_at: Optional[str] = None
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
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
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
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
            finished_at=data.get("finished_at"),
            version=int(data.get("version") or RUN_SCHEMA_VERSION),
        )


class WorkflowStore:
    """File-backed workflow repository with deterministic JSON records."""

    def __init__(self, root_dir: str | Path = "runs/workflows") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: WorkflowRecord) -> WorkflowRecord:
        existing = self.get(record.id) if self.exists(record.id) else None
        now = _utc_now()
        if existing is not None:
            record.created_at = existing.created_at
        record.updated_at = now
        self._path(record.id).write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return record

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

    def _path(self, workflow_id: str) -> Path:
        clean = _clean_id(workflow_id)
        if not clean:
            raise ValueError("workflow_id is required")
        return self.root_dir / f"{clean}.json"


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
    ) -> RunRecord:
        record = RunRecord(
            id=_clean_id(run_id or "") or f"run-{uuid.uuid4().hex[:12]}",
            workflow_id=workflow_id,
            status="created",
            input=dict(input or {}),
            recursion_limit=recursion_limit,
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

    def _path(self, run_id: str) -> Path:
        clean = _clean_id(run_id)
        if not clean:
            raise ValueError("run_id is required")
        return self.root_dir / f"{clean}.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_id(raw: str) -> str:
    return "".join(ch for ch in raw.strip() if ch.isalnum() or ch in {"-", "_"})
