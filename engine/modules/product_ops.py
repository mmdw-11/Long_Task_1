"""产品化运维辅助模块，集中保存控制台需要展示和管理的后端资源。"""

from __future__ import annotations

import json
import os
import secrets
import hashlib
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


@dataclass
class ApiKeyRecord:
    """控制台 API Key 记录，只持久化哈希和前缀，不保存完整密钥。"""

    id: str
    name: str
    prefix: str
    key_hash: str
    scope: str = "workspace"
    enabled: bool = True
    created_by: str = ""
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    last_used_at: str = ""

    def to_dict(self, *, include_secret: Optional[str] = None) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "name": self.name,
            "prefix": self.prefix,
            "scope": self.scope,
            "enabled": self.enabled,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_used_at": self.last_used_at,
        }
        if include_secret:
            data["secret"] = include_secret
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApiKeyRecord":
        if not isinstance(data, dict):
            raise ValueError("api key payload must be an object")
        name = str(data.get("name") or "").strip()
        if not name:
            raise ValueError("api key name is required")
        return cls(
            id=_clean_id(str(data.get("id") or "")) or f"ak-{uuid.uuid4().hex[:12]}",
            name=name,
            prefix=str(data.get("prefix") or ""),
            key_hash=str(data.get("key_hash") or ""),
            scope=str(data.get("scope") or "workspace"),
            enabled=bool(data.get("enabled", True)),
            created_by=str(data.get("created_by") or ""),
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
            last_used_at=str(data.get("last_used_at") or ""),
        )


class ApiKeyStore:
    """文件型 API Key 仓库，提供类似百炼控制台的密钥创建和禁用能力。"""

    def __init__(self, root_dir: str | Path = "runs/api_keys") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create(self, *, name: str, scope: str = "workspace", created_by: str = "") -> tuple[ApiKeyRecord, str]:
        secret = f"af-{secrets.token_urlsafe(32)}"
        record = ApiKeyRecord.from_dict(
            {
                "name": name,
                "scope": scope,
                "prefix": secret[:10],
                "key_hash": hashlib.sha256(secret.encode("utf-8")).hexdigest(),
                "created_by": created_by,
            }
        )
        self.save(record)
        return record, secret

    def save(self, record: ApiKeyRecord) -> ApiKeyRecord:
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

    def list(self) -> List[ApiKeyRecord]:
        return sorted([self.get(path.stem) for path in self.root_dir.glob("*.json")], key=lambda item: item.updated_at, reverse=True)

    def get(self, key_id: str) -> ApiKeyRecord:
        path = self._path(key_id)
        if not path.exists():
            raise KeyError(f"api key {key_id!r} not found")
        return ApiKeyRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def update_enabled(self, key_id: str, enabled: bool) -> ApiKeyRecord:
        record = self.get(key_id)
        record.enabled = enabled
        return self.save(record)

    def delete(self, key_id: str) -> None:
        path = self._path(key_id)
        if not path.exists():
            raise KeyError(f"api key {key_id!r} not found")
        path.unlink()

    def exists(self, key_id: str) -> bool:
        return self._path(key_id).exists()

    def _path(self, key_id: str) -> Path:
        clean = _clean_id(key_id)
        if not clean:
            raise ValueError("api key id is required")
        return self.root_dir / f"{clean}.json"


@dataclass
class ApplicationRecord:
    """应用中心记录，承接百炼式创建应用入口并绑定现有工作流。"""

    id: str
    name: str
    app_type: str = "agent"
    description: str = ""
    status: str = "draft"
    workflow_id: str = ""
    entry_agent_id: str = ""
    model: str = ""
    system_prompt: str = ""
    avatar_url: str = ""
    tool_ids: List[str] = field(default_factory=list)
    skill_ids: List[str] = field(default_factory=list)
    knowledge_base_ids: List[str] = field(default_factory=list)
    memory_bank_ids: List[str] = field(default_factory=list)
    primary_memory_bank_id: Optional[str] = None
    memory_config: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_MEMORY_CONFIG))
    prompt_variables: List[Dict[str, Any]] = field(default_factory=list)
    owner_user_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "app_type": self.app_type,
            "description": self.description,
            "status": self.status,
            "workflow_id": self.workflow_id,
            "entry_agent_id": self.entry_agent_id,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "avatar_url": self.avatar_url,
            "tool_ids": list(self.tool_ids),
            "skill_ids": list(self.skill_ids),
            "knowledge_base_ids": list(self.knowledge_base_ids),
            "memory_bank_ids": list(self.memory_bank_ids),
            "primary_memory_bank_id": self.primary_memory_bank_id,
            "memory_config": normalize_memory_config(self.memory_config),
            "prompt_variables": [dict(item) for item in self.prompt_variables],
            "owner_user_id": self.owner_user_id,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApplicationRecord":
        if not isinstance(data, dict):
            raise ValueError("application payload must be an object")
        name = str(data.get("name") or "").strip()
        if not name:
            raise ValueError("application.name is required")
        return cls(
            id=_clean_id(str(data.get("id") or "")) or f"app-{uuid.uuid4().hex[:12]}",
            name=name,
            app_type=str(data.get("app_type") or "agent"),
            description=str(data.get("description") or ""),
            status=str(data.get("status") or "draft"),
            workflow_id=str(data.get("workflow_id") or ""),
            entry_agent_id=str(data.get("entry_agent_id") or ""),
            model=str(data.get("model") or ""),
            system_prompt=str(data.get("system_prompt") or ""),
            avatar_url=str(data.get("avatar_url") or ""),
            tool_ids=[str(item) for item in data.get("tool_ids") or []],
            skill_ids=[str(item) for item in data.get("skill_ids") or []],
            knowledge_base_ids=[str(item) for item in data.get("knowledge_base_ids") or []],
            memory_bank_ids=[str(item) for item in data.get("memory_bank_ids") or []],
            primary_memory_bank_id=(str(data.get("primary_memory_bank_id")) if data.get("primary_memory_bank_id") else None),
            memory_config=normalize_memory_config(data.get("memory_config")),
            prompt_variables=[dict(item) for item in data.get("prompt_variables") or [] if isinstance(item, dict)],
            owner_user_id=str(data.get("owner_user_id") or ""),
            metadata=dict(data.get("metadata") or {}),
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
        )


class ApplicationStore:
    """文件型应用仓库，保存用户在控制台创建的应用入口。"""

    def __init__(self, root_dir: str | Path = "runs/applications") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        *,
        name: str,
        app_type: str = "agent",
        description: str = "",
        model: str = "",
        system_prompt: str = "",
        avatar_url: str = "",
        tool_ids: Optional[List[str]] = None,
        skill_ids: Optional[List[str]] = None,
        knowledge_base_ids: Optional[List[str]] = None,
        memory_bank_ids: Optional[List[str]] = None,
        primary_memory_bank_id: Optional[str] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        prompt_variables: Optional[List[Dict[str, Any]]] = None,
        owner_user_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApplicationRecord:
        record = ApplicationRecord.from_dict(
            {
                "name": name,
                "app_type": app_type,
                "description": description,
                "model": model,
                "system_prompt": system_prompt,
                "avatar_url": avatar_url,
                "tool_ids": tool_ids or [],
                "skill_ids": skill_ids or [],
                "knowledge_base_ids": knowledge_base_ids or [],
                "memory_bank_ids": memory_bank_ids or [],
                "primary_memory_bank_id": primary_memory_bank_id or ((memory_bank_ids or [None])[0]),
                "memory_config": normalize_memory_config(memory_config),
                "prompt_variables": prompt_variables or [],
                "owner_user_id": owner_user_id,
                "metadata": metadata or {},
            }
        )
        return self.save(record)

    def save(self, record: ApplicationRecord) -> ApplicationRecord:
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

    def list(self) -> List[ApplicationRecord]:
        records = [self.get(path.stem) for path in sorted(self.root_dir.glob("*.json"))]
        return sorted(records, key=lambda item: item.updated_at, reverse=True)

    def get(self, app_id: str) -> ApplicationRecord:
        path = self._path(app_id)
        if not path.exists():
            raise KeyError(f"application {app_id!r} not found")
        return ApplicationRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def delete(self, app_id: str) -> None:
        path = self._path(app_id)
        if not path.exists():
            raise KeyError(f"application {app_id!r} not found")
        path.unlink()

    def exists(self, app_id: str) -> bool:
        return self._path(app_id).exists()

    def _path(self, app_id: str) -> Path:
        clean = _clean_id(app_id)
        if not clean:
            raise ValueError("application id is required")
        return self.root_dir / f"{clean}.json"


DEFAULT_MEMORY_RETRIEVAL_CONFIG: Dict[str, Any] = {"top_k":5,"similarity_threshold":0.3,"mode":"hybrid","temporal_weight":0.15,"scopes":["working","task","project","global"],"expand_project":True}


def normalize_memory_retrieval_config(value: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config={**DEFAULT_MEMORY_RETRIEVAL_CONFIG,**dict(value or {})}
    config["top_k"]=max(1,min(20,int(config.get("top_k") or 5)))
    config["similarity_threshold"]=max(0.0,min(1.0,float(config.get("similarity_threshold") or 0)))
    config["temporal_weight"]=max(0.0,min(1.0,float(config.get("temporal_weight") or 0)))
    config["mode"]=config["mode"] if config.get("mode") in {"dense","sparse","hybrid","hybrid_temporal"} else "hybrid"
    config["scopes"]=[scope for scope in config.get("scopes",[]) if scope in {"working","task","project","global"}] or list(DEFAULT_MEMORY_RETRIEVAL_CONFIG["scopes"])
    config["expand_project"]=bool(config.get("expand_project",True))
    return config


def default_memory_rules() -> List[Dict[str, Any]]:
    return [
        {"id":f"rule-{uuid.uuid4().hex[:12]}","type":"fragment","name":"默认记忆片段规则","description":"提取稳定偏好、长期事实、项目决策和可复用经验","instruction":"仅提取对未来任务仍有价值的稳定信息，忽略临时请求和工具原始输出。","source_types":["user","assistant","trusted_tool"],"update_policy":"merge","retention_days":180,"target_scope":"project","enabled":True},
        {"id":f"rule-{uuid.uuid4().hex[:12]}","type":"profile","name":"默认用户画像规则","description":"提取语言、格式偏好、专业背景、长期目标和项目角色","instruction":"提取非敏感用户画像；禁止密码、密钥、身份凭据和敏感属性。","source_types":["user"],"update_policy":"merge","retention_days":0,"target_scope":"project","enabled":True},
    ]


@dataclass
class MemoryBankRecord:
    """控制台记忆库资源；不替代运行时上下文账本，只保存其可配置入口。"""

    id: str
    name: str
    description: str = ""
    status: str = "ready"
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_config: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "description": self.description,
            "status": self.status, "metadata": dict(self.metadata),
            "retrieval_config": normalize_memory_retrieval_config(self.retrieval_config),
            "created_at": self.created_at, "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryBankRecord":
        name = str(data.get("name") or "").strip()
        if not name:
            raise ValueError("memory bank name is required")
        return cls(
            id=_clean_id(str(data.get("id") or "")) or f"memory-{uuid.uuid4().hex[:12]}",
            name=name, description=str(data.get("description") or ""),
            status=str(data.get("status") or "ready"), metadata=dict(data.get("metadata") or {}),
            retrieval_config=normalize_memory_retrieval_config(data.get("retrieval_config")),
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
        )


class MemoryBankStore:
    """文件型记忆库目录，供应用挂载和控制台浏览。"""

    def __init__(self, root_dir: str | Path = "runs/memory_banks") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create(self, *, name: str, description: str = "", metadata: Optional[Dict[str, Any]] = None) -> MemoryBankRecord:
        effective_metadata={**(metadata or {}),"rules":default_memory_rules()}
        return self.save(MemoryBankRecord.from_dict({"name": name, "description": description, "metadata": effective_metadata}))

    def save(self, record: MemoryBankRecord) -> MemoryBankRecord:
        now = _utc_now()
        if self.exists(record.id):
            record.created_at = self.get(record.id).created_at
        record.updated_at = now
        self._path(record.id).write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return record

    def list(self) -> List[MemoryBankRecord]:
        return sorted([self.get(p.stem) for p in self.root_dir.glob("*.json")], key=lambda item: item.updated_at, reverse=True)

    def get(self, bank_id: str) -> MemoryBankRecord:
        path = self._path(bank_id)
        if not path.exists():
            raise KeyError(f"memory bank {bank_id!r} not found")
        return MemoryBankRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def delete(self, bank_id: str) -> None:
        path = self._path(bank_id)
        if not path.exists():
            raise KeyError(f"memory bank {bank_id!r} not found")
        path.unlink()

    def exists(self, bank_id: str) -> bool:
        return self._path(bank_id).exists()

    def _path(self, bank_id: str) -> Path:
        clean = _clean_id(bank_id)
        if not clean:
            raise ValueError("memory bank id is required")
        return self.root_dir / f"{clean}.json"


@dataclass
class ConsoleResourceRecord:
    """组件、知识库、连接、评测等控制台资源的通用文件记录。"""

    id: str
    kind: str
    name: str
    description: str = ""
    status: str = "ready"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "name": self.name, "description": self.description, "status": self.status, "metadata": dict(self.metadata), "created_at": self.created_at, "updated_at": self.updated_at}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsoleResourceRecord":
        kind = _clean_id(str(data.get("kind") or ""))
        name = str(data.get("name") or "").strip()
        if not kind or not name:
            raise ValueError("resource kind and name are required")
        return cls(id=_clean_id(str(data.get("id") or "")) or f"{kind}-{uuid.uuid4().hex[:12]}", kind=kind, name=name, description=str(data.get("description") or ""), status=str(data.get("status") or "ready"), metadata=dict(data.get("metadata") or {}), created_at=str(data.get("created_at") or _utc_now()), updated_at=str(data.get("updated_at") or _utc_now()))


class ConsoleResourceStore:
    """通用控制台资源仓库，保持产品页数据可操作且不侵入运行时核心状态。"""

    def __init__(self, root_dir: str | Path = "runs/console_resources") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create(self, *, kind: str, name: str, description: str = "", metadata: Optional[Dict[str, Any]] = None) -> ConsoleResourceRecord:
        return self.save(ConsoleResourceRecord.from_dict({"kind": kind, "name": name, "description": description, "metadata": metadata or {}}))

    def save(self, record: ConsoleResourceRecord) -> ConsoleResourceRecord:
        now = _utc_now(); path = self._path(record.kind, record.id)
        if path.exists(): record.created_at = self.get(record.kind, record.id).created_at
        record.updated_at = now
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return record

    def list(self, kind: str) -> List[ConsoleResourceRecord]:
        clean = _clean_id(kind)
        return sorted([ConsoleResourceRecord.from_dict(json.loads(p.read_text(encoding="utf-8"))) for p in (self.root_dir / clean).glob("*.json")], key=lambda item: item.updated_at, reverse=True)

    def get(self, kind: str, resource_id: str) -> ConsoleResourceRecord:
        path = self._path(kind, resource_id)
        if not path.exists(): raise KeyError(f"resource {resource_id!r} not found")
        return ConsoleResourceRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def delete(self, kind: str, resource_id: str) -> None:
        path = self._path(kind, resource_id)
        if not path.exists(): raise KeyError(f"resource {resource_id!r} not found")
        path.unlink()

    def _path(self, kind: str, resource_id: str) -> Path:
        clean_kind, clean_id = _clean_id(kind), _clean_id(resource_id)
        if not clean_kind or not clean_id: raise ValueError("resource kind and id are required")
        return self.root_dir / clean_kind / f"{clean_id}.json"


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
DEFAULT_MEMORY_CONFIG: Dict[str, Any] = {
    "short_term_enabled": True,
    "context_rounds": 8,
    "context_token_budget": 0,
    "rolling_summary_enabled": True,
    "long_term_enabled": True,
    "memory_top_k": 5,
    "wakeup_level": 1,
    "auto_write": True,
    "deduplicate": True,
    "sensitive_filter": True,
    "retrieval_override_enabled": False,
}


def normalize_memory_config(value: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = {**DEFAULT_MEMORY_CONFIG, **dict(value or {})}
    config["context_rounds"] = max(0, min(30, int(config["context_rounds"])))
    config["context_token_budget"] = max(0, int(config["context_token_budget"]))
    config["memory_top_k"] = max(1, min(20, int(config["memory_top_k"])))
    level = config.get("wakeup_level", 1)
    aliases = {"silent":0,"standard":1,"deep":2,"recovery":3}
    config["wakeup_level"] = max(0, min(3, int(aliases.get(str(level).lower(), level))))
    for key in ("short_term_enabled","rolling_summary_enabled","long_term_enabled","auto_write","deduplicate","sensitive_filter","retrieval_override_enabled"):
        config[key] = bool(config[key])
    return config
