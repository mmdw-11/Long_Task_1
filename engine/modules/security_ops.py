"""后端请求保护与审计模块。

这个模块提供两项最小产品化能力：
1. 可选的管理密钥校验：配置后保护所有写接口。
2. 审计日志落盘：记录谁在什么时间访问了什么写接口，以及结果如何。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ApiAuditRecord:
    """一条接口审计记录。"""

    ts: str
    method: str
    path: str
    status_code: int
    actor: str = ""
    authorized: bool = True
    detail: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "method": self.method,
            "path": self.path,
            "status_code": self.status_code,
            "actor": self.actor,
            "authorized": self.authorized,
            "detail": self.detail,
            "metadata": dict(self.metadata),
        }


class ApiAuditStore:
    """简单 JSONL 审计仓库。"""

    def __init__(self, path: str | Path = "runs/audit/api_audit.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ApiAuditRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False, default=str) + "\n")

    def list(self, *, limit: int = 100) -> List[ApiAuditRecord]:
        if not self.path.exists():
            return []
        rows: List[ApiAuditRecord] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                data = json.loads(line)
                rows.append(
                    ApiAuditRecord(
                        ts=str(data.get("ts") or ""),
                        method=str(data.get("method") or ""),
                        path=str(data.get("path") or ""),
                        status_code=int(data.get("status_code") or 0),
                        actor=str(data.get("actor") or ""),
                        authorized=bool(data.get("authorized", True)),
                        detail=str(data.get("detail") or ""),
                        metadata=dict(data.get("metadata") or {}),
                    )
                )
        return rows[-limit:][::-1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
