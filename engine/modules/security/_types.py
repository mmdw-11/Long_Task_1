"""安全与审计数据结构。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SensitiveFinding:
    """一次敏感信息命中记录。

    不保存原始命中值，避免审计包本身泄露敏感数据。
    """

    kind: str
    path: str
    replacement: str
    start: int = -1
    end: int = -1
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "replacement": self.replacement,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }


@dataclass
class RedactionResult:
    """脱敏结果。"""

    original: Any
    redacted: Any
    findings: List[SensitiveFinding] = field(default_factory=list)

    @property
    def redacted_count(self) -> int:
        return len(self.findings)

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for finding in self.findings:
            counts[finding.kind] = counts.get(finding.kind, 0) + 1
        return {
            "redacted_count": self.redacted_count,
            "kinds": counts,
            "paths": sorted({finding.path for finding in self.findings}),
        }

    def to_dict(self, *, include_payload: bool = True) -> Dict[str, Any]:
        data = {
            "summary": self.summary(),
            "findings": [finding.to_dict() for finding in self.findings],
        }
        if include_payload:
            data["redacted"] = self.redacted
        return data


@dataclass
class AuditPack:
    """敏感数据外传前的审计包。"""

    node: str
    original_hash: str
    redacted_hash: str
    findings_summary: Dict[str, Any]
    approved: bool = False
    reviewer: str = ""
    reason: str = ""
    ts: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "original_hash": self.original_hash,
            "redacted_hash": self.redacted_hash,
            "findings_summary": self.findings_summary,
            "approved": self.approved,
            "reviewer": self.reviewer,
            "reason": self.reason,
            "ts": self.ts,
            "metadata": self.metadata,
        }
