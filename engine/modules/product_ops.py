"""产品化运维辅助模块。

该模块给前端提供两类后端能力：
1. 系统状态摘要：集中暴露模型、运行目录、能力开关与时间信息。
2. 工具目录仓库：把可被 Agent 使用的工具元信息持久化，避免散落在前端配置里。
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .skills import SkillRepository
from .workflows import RunStore, WorkflowStore


@dataclass
class ToolRecord:
    """工具目录中的一条工具元数据。"""

    id: str
    name: str
    display_name: str
    description: str = ""
    category: str = "general"
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category,
            "enabled": self.enabled,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolRecord":
        if not isinstance(data, dict):
            raise ValueError("tool payload must be an object")
        name = str(data.get("name") or "").strip()
        display_name = str(data.get("display_name") or "").strip()
        if not name:
            raise ValueError("tool.name is required")
        if not display_name:
            raise ValueError("tool.display_name is required")
        return cls(
            id=_clean_id(str(data.get("id") or "")) or f"tool-{uuid.uuid4().hex[:12]}",
            name=name,
            display_name=display_name,
            description=str(data.get("description") or ""),
            category=str(data.get("category") or "general"),
            enabled=bool(data.get("enabled", True)),
            tags=[str(item) for item in data.get("tags") or []],
            metadata=dict(data.get("metadata") or {}),
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
        )


class ToolCatalogStore:
    """文件型工具目录仓库。"""

    def __init__(self, root_dir: str | Path = "runs/tool_catalog") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        *,
        name: str,
        display_name: str,
        description: str = "",
        category: str = "general",
        enabled: bool = True,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tool_id: Optional[str] = None,
    ) -> ToolRecord:
        record = ToolRecord.from_dict(
            {
                "id": tool_id,
                "name": name,
                "display_name": display_name,
                "description": description,
                "category": category,
                "enabled": enabled,
                "tags": tags or [],
                "metadata": metadata or {},
            }
        )
        return self.save(record)

    def save(self, record: ToolRecord) -> ToolRecord:
        now = _utc_now()
        if self.exists(record.id):
            existing = self.get(record.id)
            record.created_at = existing.created_at
        else:
            record.created_at = record.created_at or now
        record.updated_at = now
        self._path(record.id).write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return record

    def get(self, tool_id: str) -> ToolRecord:
        path = self._path(tool_id)
        if not path.exists():
            raise KeyError(f"tool {tool_id!r} not found")
        return ToolRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self, *, enabled: Optional[bool] = None) -> List[ToolRecord]:
        records = [self.get(path.stem) for path in sorted(self.root_dir.glob("*.json"))]
        if enabled is not None:
            records = [item for item in records if item.enabled is enabled]
        return sorted(records, key=lambda item: item.updated_at, reverse=True)

    def delete(self, tool_id: str) -> None:
        path = self._path(tool_id)
        if not path.exists():
            raise KeyError(f"tool {tool_id!r} not found")
        path.unlink()

    def exists(self, tool_id: str) -> bool:
        return self._path(tool_id).exists()

    def _path(self, tool_id: str) -> Path:
        clean = _clean_id(tool_id)
        if not clean:
            raise ValueError("tool_id is required")
        return self.root_dir / f"{clean}.json"


class ProductStatusService:
    """汇总后端能力、配置和健康摘要。"""

    def __init__(
        self,
        *,
        workflow_root: str,
        run_root: str,
        skill_root: str,
        tool_root: str,
    ) -> None:
        self.workflow_root = workflow_root
        self.run_root = run_root
        self.skill_root = skill_root
        self.tool_root = tool_root

    def snapshot(self) -> Dict[str, Any]:
        device_base_url = os.environ.get("DEVICE_BASE_URL") or os.environ.get("DEVICE_ENDPOINT") or ""
        device_model = os.environ.get("DEVICE_MODEL") or ""
        edge_base_url = os.environ.get("EDGE_OLLAMA_BASE_URL") or os.environ.get("EDGE_ENDPOINT") or ""
        edge_model = os.environ.get("EDGE_MODEL") or ""
        openai_base_url = os.environ.get("OPENAI_BASE_URL") or ""
        openai_model = os.environ.get("OPENAI_MODEL") or ""
        return {
            "now": _utc_now(),
            "capabilities": {
                "workflow_persistence": True,
                "background_runs": True,
                "skill_lifecycle": True,
                "agent_runtime": True,
                "tool_catalog": True,
                "runtime_tool_execution": True,
                "run_todo_stream": True,
                "approval_events": True,
            },
            "models": {
                "device": {
                    "base_url": device_base_url,
                    "model": device_model,
                    "configured": bool(device_base_url and device_model),
                },
                "edge": {
                    "base_url": edge_base_url,
                    "model": edge_model,
                    "configured": bool(edge_base_url and edge_model),
                },
                "cloud": {
                    "base_url": openai_base_url,
                    "model": openai_model,
                    "configured": bool(openai_base_url and openai_model),
                },
            },
            "storage": {
                "workflow_root": self.workflow_root,
                "run_root": self.run_root,
                "skill_root": self.skill_root,
                "tool_root": self.tool_root,
            },
        }


class ProjectSnapshotService:
    """聚合导出后端核心数据，供备份、迁移预检查和问题排查使用。"""

    def __init__(
        self,
        *,
        workflows: WorkflowStore,
        runs: RunStore,
        skills: SkillRepository,
        tools: ToolCatalogStore,
        status_service: ProductStatusService,
    ) -> None:
        self.workflows = workflows
        self.runs = runs
        self.skills = skills
        self.tools = tools
        self.status_service = status_service

    def export(
        self,
        *,
        include_runs: bool = True,
        include_skill_content: bool = True,
    ) -> Dict[str, Any]:
        """生成只读快照；默认包含运行记录和技能正文，方便完整排障。"""
        workflow_records = self.workflows.list()
        run_records = self.runs.list() if include_runs else []
        skill_records = self.skills.list()
        tool_records = self.tools.list()
        exported_skills = []
        for skill in skill_records:
            # 技能正文可能较长，接口允许前端按需关闭正文导出。
            payload = skill.to_dict()
            if not include_skill_content:
                payload.pop("content", None)
            exported_skills.append(payload)
        return {
            "format_version": 1,
            "generated_at": _utc_now(),
            "system": self.status_service.snapshot(),
            "summary": {
                "workflows": len(workflow_records),
                "runs": len(run_records),
                "skills": len(skill_records),
                "tools": len(tool_records),
            },
            "workflows": [item.to_dict() for item in workflow_records],
            "runs": [item.to_dict() for item in run_records],
            "skills": exported_skills,
            "tools": [item.to_dict() for item in tool_records],
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_id(raw: str) -> str:
    return "".join(ch for ch in raw.strip() if ch.isalnum() or ch in {"-", "_"})
