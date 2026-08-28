"""Persistent short-term conversations and memory-governance audit records."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ConversationRecord:
    id: str
    application_id: str
    owner_user_id: str = ""
    title: str = "新会话"
    messages: List[Dict[str, Any]] = field(default_factory=list)
    rolling_summary: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {"id":self.id,"application_id":self.application_id,"owner_user_id":self.owner_user_id,"title":self.title,"messages":list(self.messages),"rolling_summary":self.rolling_summary,"token_estimate":estimate_tokens(self.rolling_summary)+sum(estimate_tokens(str(x.get("content") or "")) for x in self.messages),"created_at":self.created_at,"updated_at":self.updated_at}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationRecord":
        return cls(id=str(data["id"]),application_id=str(data["application_id"]),owner_user_id=str(data.get("owner_user_id") or ""),title=str(data.get("title") or "新会话"),messages=list(data.get("messages") or []),rolling_summary=str(data.get("rolling_summary") or ""),created_at=str(data.get("created_at") or _now()),updated_at=str(data.get("updated_at") or _now()))


class ConversationStore:
    def __init__(self, root_dir: str | Path = "runs/conversations") -> None:
        self.root_dir=Path(root_dir);self.root_dir.mkdir(parents=True,exist_ok=True)

    def create(self, application_id: str, owner_user_id: str = "") -> ConversationRecord:
        return self.save(ConversationRecord(id=f"conversation-{uuid.uuid4().hex[:12]}",application_id=application_id,owner_user_id=owner_user_id))

    def save(self, record: ConversationRecord) -> ConversationRecord:
        record.updated_at=_now();self._path(record.id).write_text(json.dumps(record.to_dict(),ensure_ascii=False,indent=2),encoding="utf-8");return record

    def get(self, conversation_id: str) -> ConversationRecord:
        path=self._path(conversation_id)
        if not path.exists():raise KeyError(f"conversation {conversation_id!r} not found")
        return ConversationRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self, application_id: str, owner_user_id: str = "") -> List[ConversationRecord]:
        items=[ConversationRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))) for path in self.root_dir.glob("*.json")]
        return sorted([x for x in items if x.application_id==application_id and (not owner_user_id or x.owner_user_id in {"",owner_user_id})],key=lambda x:x.updated_at,reverse=True)

    def clear(self, conversation_id: str) -> ConversationRecord:
        record=self.get(conversation_id);record.messages=[];record.rolling_summary="";return self.save(record)

    def _path(self, conversation_id: str) -> Path:
        clean=re.sub(r"[^A-Za-z0-9_.-]","",conversation_id)
        if not clean:raise ValueError("conversation id is required")
        return self.root_dir/f"{clean}.json"


class MemoryAuditStore:
    def __init__(self, root_dir: str | Path = "runs/memory_audit") -> None:
        self.root_dir=Path(root_dir);self.root_dir.mkdir(parents=True,exist_ok=True)

    def append(self, application_id: str, event: Dict[str, Any]) -> Dict[str, Any]:
        item={"id":f"audit-{uuid.uuid4().hex[:12]}","application_id":application_id,"timestamp":_now(),**event}
        with self._path(application_id).open("a",encoding="utf-8") as handle:handle.write(json.dumps(item,ensure_ascii=False)+"\n")
        return item

    def list(self, application_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        path=self._path(application_id)
        if not path.exists():return []
        rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return list(reversed(rows[-max(1,min(limit,500)):]))

    def _path(self, application_id: str) -> Path:
        clean=re.sub(r"[^A-Za-z0-9_.-]","",application_id)
        return self.root_dir/f"{clean}.jsonl"


def estimate_tokens(text: str) -> int:
    return max(0,(len(text)+3)//4)


def build_conversation_context(record: ConversationRecord, config: Dict[str, Any], current_input: str = "") -> Dict[str, Any]:
    if not config.get("short_term_enabled") or int(config.get("context_rounds") or 0)==0:
        return {"text":"","messages":[],"summary":"","rounds":0,"estimated_tokens":0,"compressed":False}
    messages=record.messages[-int(config["context_rounds"])*2:]
    budget=int(config.get("context_token_budget") or 0) or 12000
    reserved=estimate_tokens(current_input)+1200
    available=max(500,budget-reserved)
    selected=list(messages);older=[]
    while len(selected)>4 and sum(estimate_tokens(str(x.get("content") or "")) for x in selected)>available:
        older.append(selected.pop(0))
    summary=record.rolling_summary
    compressed=bool(older)
    if older and config.get("rolling_summary_enabled"):
        fragments=[f"{x.get('role','message')}: {str(x.get('content') or '')[:300]}" for x in older]
        summary=(summary+"\n"+"\n".join(fragments)).strip()[-3000:]
    lines=[]
    if summary:lines.append("较早对话摘要：\n"+summary)
    if selected:lines.append("最近对话：\n"+"\n".join(f"{x.get('role')}: {x.get('content')}" for x in selected))
    text="\n\n".join(lines)
    return {"text":text,"messages":selected,"summary":summary,"rounds":len(selected)//2,"estimated_tokens":estimate_tokens(text),"compressed":compressed}


SENSITIVE_PATTERN=re.compile(r"(?i)(api[_ -]?key|access[_ -]?token|password|密码|密钥)\s*[:=]\s*\S+")


def durable_memory_candidates(text: str) -> List[str]:
    candidates=[]
    for sentence in re.split(r"[\n。！？!?]+",text):
        value=sentence.strip()
        if len(value)<6:continue
        if re.search(r"(我喜欢|我偏好|请始终|以后都|项目采用|项目使用|我们决定|记住|长期)",value):candidates.append(value[:1000])
    return list(dict.fromkeys(candidates))[:5]
