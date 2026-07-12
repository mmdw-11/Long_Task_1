"""审计包生成。"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from ._types import AuditPack, RedactionResult


def stable_hash(value: Any) -> str:
    """对任意可 JSON 化对象计算稳定 SHA256。"""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditPackBuilder:
    """生成不含原始敏感值的审计包。"""

    def build(
        self,
        *,
        node: str,
        result: RedactionResult,
        approved: bool = False,
        reviewer: str = "",
        reason: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> AuditPack:
        return AuditPack(
            node=node,
            original_hash=stable_hash(result.original),
            redacted_hash=stable_hash(result.redacted),
            findings_summary=result.summary(),
            approved=approved,
            reviewer=reviewer,
            reason=reason,
            metadata=dict(metadata or {}),
        )
