"""File-backed model connection catalog with write-only credential references."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


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
    enabled: bool = True
    test_status: str = "untested"
    capabilities: List[str] = field(default_factory=lambda: ["chat"])
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {"id":self.id,"name":self.name,"provider":self.provider,"model_id":self.model_id,"base_url":self.base_url,"api_key_env":self.api_key_env,"tier":self.tier,"enabled":self.enabled,"test_status":self.test_status,"configured":bool(self.base_url and self.model_id and (not self.api_key_env or os.environ.get(self.api_key_env))),"capabilities":list(self.capabilities),"created_at":self.created_at,"updated_at":self.updated_at}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConnection":
        name, model_id = str(data.get("name") or "").strip(), str(data.get("model_id") or "").strip()
        if not name or not model_id:
            raise ValueError("model connection name and model_id are required")
        tier = str(data.get("tier") or "cloud")
        if tier not in {"device","edge","cloud"}:
            raise ValueError("model connection tier must be device, edge or cloud")
        return cls(id=str(data.get("id") or f"model-{uuid.uuid4().hex[:12]}"),name=name,provider=str(data.get("provider") or "openai-compatible"),model_id=model_id,base_url=str(data.get("base_url") or "").rstrip("/"),api_key_env=str(data.get("api_key_env") or ""),tier=tier,enabled=bool(data.get("enabled",True)),test_status=str(data.get("test_status") or "untested"),capabilities=[str(x) for x in data.get("capabilities") or ["chat"]],created_at=str(data.get("created_at") or _now()),updated_at=str(data.get("updated_at") or _now()))


class ModelConnectionStore:
    def __init__(self, root: str | Path = "runs/model_connections") -> None:
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
    def save(self, item: ModelConnection) -> ModelConnection:
        if self.exists(item.id): item.created_at=self.get(item.id).created_at
        item.updated_at=_now(); self._path(item.id).write_text(json.dumps(item.to_dict(),ensure_ascii=False,indent=2),encoding="utf-8"); return item
    def create(self, data: Dict[str, Any]) -> ModelConnection: return self.save(ModelConnection.from_dict(data))
    def get(self, item_id: str) -> ModelConnection:
        path=self._path(item_id)
        if not path.exists(): raise KeyError(f"model connection {item_id!r} not found")
        return ModelConnection.from_dict(json.loads(path.read_text(encoding="utf-8")))
    def list(self) -> List[ModelConnection]: return sorted([self.get(p.stem) for p in self.root.glob("*.json")],key=lambda x:x.updated_at,reverse=True)
    def exists(self,item_id:str)->bool:return self._path(item_id).exists()
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
