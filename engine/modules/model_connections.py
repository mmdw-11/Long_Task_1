"""File-backed model connection catalog with write-only credential references."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ModelConnection:
    id: str
    name: str
    provider: str
    model_id: str
    base_url: str
    api_key_env: str = ""
    tier: str = "cloud"
    auto_default: bool = False
    enabled: bool = True
    test_status: str = "untested"
    capabilities: List[str] = field(default_factory=lambda: ["chat"])
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {"id":self.id,"name":self.name,"provider":self.provider,"model_id":self.model_id,"base_url":self.base_url,"api_key_env":self.api_key_env,"tier":self.tier,"auto_default":self.auto_default,"enabled":self.enabled,"test_status":self.test_status,"configured":self.configured,"capabilities":list(self.capabilities),"created_at":self.created_at,"updated_at":self.updated_at}

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model_id and (not self.api_key_env or os.environ.get(self.api_key_env)))

    @property
    def runnable(self) -> bool:
        return self.enabled and self.configured and self.test_status == "succeeded"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConnection":
        name, model_id = str(data.get("name") or "").strip(), str(data.get("model_id") or "").strip()
        if not name or not model_id:
            raise ValueError("model connection name and model_id are required")
        tier = str(data.get("tier") or "cloud")
        if tier not in {"device","edge","cloud"}:
            raise ValueError("model connection tier must be device, edge or cloud")
        return cls(id=str(data.get("id") or f"model-{uuid.uuid4().hex[:12]}"),name=name,provider=str(data.get("provider") or "openai-compatible"),model_id=model_id,base_url=str(data.get("base_url") or "").rstrip("/"),api_key_env=str(data.get("api_key_env") or ""),tier=tier,auto_default=bool(data.get("auto_default",False)),enabled=bool(data.get("enabled",True)),test_status=str(data.get("test_status") or "untested"),capabilities=[str(x) for x in data.get("capabilities") or ["chat"]],created_at=str(data.get("created_at") or _now()),updated_at=str(data.get("updated_at") or _now()))


class ModelConnectionStore:
    def __init__(self, root: str | Path = "runs/model_connections") -> None:
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
    def save(self, item: ModelConnection) -> ModelConnection:
        if self.exists(item.id): item.created_at=self.get(item.id).created_at
        if item.auto_default:
            for other in self.list():
                if other.id != item.id and other.tier == item.tier and other.auto_default:
                    other.auto_default = False
                    self._write(other)
        # Changing a tested connection invalidates its previous health result.
        if self.exists(item.id):
            previous = self.get(item.id)
            if (previous.base_url, previous.model_id, previous.api_key_env) != (item.base_url, item.model_id, item.api_key_env):
                item.test_status = "untested"
        return self._write(item)
    def create(self, data: Dict[str, Any]) -> ModelConnection: return self.save(ModelConnection.from_dict(data))
    def get(self, item_id: str) -> ModelConnection:
        path=self._path(item_id)
        if not path.exists(): raise KeyError(f"model connection {item_id!r} not found")
        return ModelConnection.from_dict(json.loads(path.read_text(encoding="utf-8")))
    def list(self) -> List[ModelConnection]: return sorted([self.get(p.stem) for p in self.root.glob("*.json")],key=lambda x:x.updated_at,reverse=True)
    def exists(self,item_id:str)->bool:return self._path(item_id).exists()
    def delete(self, item_id: str) -> None:
        path = self._path(item_id)
        if not path.exists():
            raise KeyError(f"model connection {item_id!r} not found")
        path.unlink()
    def default_for_tier(self, tier: str, *, runnable: bool = True) -> Optional[ModelConnection]:
        matches = [item for item in self.list() if item.tier == tier and item.auto_default]
        if runnable:
            matches = [item for item in matches if item.runnable]
        return matches[0] if matches else None
    def auto_status(self) -> Dict[str, Any]:
        tiers: Dict[str, Any] = {}
        for tier in ("device", "edge", "cloud"):
            configured_default = self.default_for_tier(tier, runnable=False)
            runnable_default = self.default_for_tier(tier, runnable=True)
            tiers[tier] = {
                "ready": runnable_default is not None,
                "connection": runnable_default.to_dict() if runnable_default else (configured_default.to_dict() if configured_default else None),
                "reason": "" if runnable_default else ("默认模型未启用、未配置或尚未测试成功" if configured_default else "尚未设置 AUTO 默认模型"),
            }
        return {"ready": all(item["ready"] for item in tiers.values()), "tiers": tiers}
    def _write(self, item: ModelConnection) -> ModelConnection:
        item.updated_at=_now(); self._path(item.id).write_text(json.dumps(item.to_dict(),ensure_ascii=False,indent=2),encoding="utf-8"); return item
    def _path(self,item_id:str)->Path:return self.root/f"{''.join(c for c in item_id if c.isalnum() or c in '-_')}.json"


MODEL_PRESETS = [
 {"provider":"deepseek","name":"DeepSeek","base_url":"https://api.deepseek.com/v1"},
 {"provider":"qwen","name":"通义千问 Qwen","base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1"},
 {"provider":"glm","name":"智谱 GLM","base_url":"https://open.bigmodel.cn/api/paas/v4"},
 {"provider":"doubao","name":"豆包 Doubao","base_url":"https://ark.cn-beijing.volces.com/api/v3"},
 {"provider":"openai","name":"OpenAI GPT","base_url":"https://api.openai.com/v1"},
 {"provider":"anthropic","name":"Claude（兼容网关）","base_url":""},
 {"provider":"google","name":"Gemini（兼容网关）","base_url":""},
]
