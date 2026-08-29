"""失败轨迹（共享数据基础）。

记录图执行过程中各节点的失败信息，作为动态路由、恢复策略、流控等模块的
上下文输入。失败轨迹不单独成模块，而是作为引擎与各扩展模块共享的数据结构。

约定：``GraphState`` 中以 ``__failures__`` 字段（使用 append reducer）承载
失败记录列表，各模块可从状态快照中读取该字段获取失败上下文。
"""

from __future__ import annotations

import time
import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 状态中承载失败轨迹的约定字段名。
FAILURES_KEY = "__failures__"
RECOVERY_TRACE_KEY = "__recovery_trace__"
SIDE_EFFECT_JOURNAL_KEY = "__side_effect_journal__"


class FailureKind(str, enum.Enum):
    AGENT="agent"; TOOL="tool"; TIMEOUT="timeout"; MESSAGE_MISSING="message_missing"
    EVIDENCE_MISSING="evidence_missing"; PLAN_CONFLICT="plan_conflict"
    STEP_FAILED="step_failed"; RESOURCE_UNAVAILABLE="resource_unavailable"
    ROLE_UNAVAILABLE="role_unavailable"; VALIDATION_FAILED="validation_failed"


class FailureSeverity(str, enum.Enum):
    TRANSIENT="transient"; DEGRADED="degraded"; CRITICAL="critical"; FATAL="fatal"


@dataclass
class FailureContext:
    kind: FailureKind; node: str; message: str; step: int=0
    severity: FailureSeverity=FailureSeverity.TRANSIENT
    attempt: int=1; retryable: bool=True; side_effect_class: str="idempotent"
    metadata: Dict[str,Any]=field(default_factory=dict); timestamp: float=field(default_factory=time.time)
    def to_dict(self)->Dict[str,Any]:
        return {"kind":self.kind.value,"node":self.node,"message":self.message,"step":self.step,"severity":self.severity.value,"attempt":self.attempt,"retryable":self.retryable,"side_effect_class":self.side_effect_class,"metadata":self.metadata,"timestamp":self.timestamp}
    @classmethod
    def from_error(cls,error:BaseException,*,node:str,step:int=0,metadata:Optional[Dict[str,Any]]=None,attempt:int=1)->"FailureContext":
        md=dict(metadata or {}); text=str(error).lower()
        declared=getattr(error,"failure_kind",None)
        kind=FailureKind(declared) if declared else FailureKind.TIMEOUT if isinstance(error,TimeoutError) or "timeout" in text else FailureKind.RESOURCE_UNAVAILABLE if any(x in text for x in ("resource","rate limit","429")) else FailureKind.TOOL if md.get("tool") or "tool" in text else FailureKind.AGENT
        return cls(kind,node,str(error),step,FailureSeverity.TRANSIENT if kind in {FailureKind.TIMEOUT,FailureKind.RESOURCE_UNAVAILABLE} else FailureSeverity.CRITICAL,attempt,True,str(md.get("idempotency") or "idempotent"),md)


@dataclass
class FailureRecord:
    """一条节点失败记录。"""

    node: str
    error_type: str
    message: str
    step: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    kind: str = FailureKind.AGENT.value
    severity: str = FailureSeverity.CRITICAL.value
    attempt: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "error_type": self.error_type,
            "message": self.message,
            "step": self.step,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "kind": self.kind,
            "severity": self.severity,
            "attempt": self.attempt,
        }


class FailureTrace:
    """失败轨迹容器：维护按时间顺序追加的失败记录。"""

    def __init__(self, records: Optional[List[FailureRecord]] = None) -> None:
        self.records: List[FailureRecord] = list(records or [])

    def record(
        self,
        node: str,
        error: BaseException,
        *,
        step: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FailureRecord:
        """依据一个异常构造并追加一条失败记录。"""
        rec = FailureRecord(
            node=node,
            error_type=type(error).__name__,
            message=str(error),
            step=step,
            metadata=dict(metadata or {}),
        )
        self.records.append(rec)
        return rec

    def recent(self, n: int = 1) -> List[FailureRecord]:
        """返回最近 n 条失败记录。"""
        return self.records[-n:] if n > 0 else []

    def by_node(self, name: str) -> List[FailureRecord]:
        """返回某个节点的所有失败记录。"""
        return [r for r in self.records if r.node == name]

    def to_list(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.records]

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "FailureTrace":
        """从状态快照的 ``__failures__`` 字段重建失败轨迹（供模块读取上下文）。"""
        raw = state.get(FAILURES_KEY) or []
        records: List[FailureRecord] = []
        for item in raw:
            if isinstance(item, FailureRecord):
                records.append(item)
            elif isinstance(item, dict):
                records.append(
                    FailureRecord(
                        node=item.get("node", ""),
                        error_type=item.get("error_type", ""),
                        message=item.get("message", ""),
                        step=item.get("step", 0),
                        timestamp=item.get("timestamp", time.time()),
                        metadata=item.get("metadata", {}),
                        kind=item.get("kind", FailureKind.AGENT.value),
                        severity=item.get("severity", FailureSeverity.CRITICAL.value),
                        attempt=int(item.get("attempt", 1)),
                    )
                )
        return cls(records)

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"FailureTrace(records={len(self.records)})"
