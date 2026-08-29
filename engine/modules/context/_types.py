"""Runtime context ledger data structures.

The context ledger is run-scoped state for long-running agent workflows. It is
separate from long-term memory: memory stores reusable knowledge, while the
ledger preserves the current run's goal, constraints, verified facts, progress,
and recent failures in a compact, auditable form.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


CONTEXT_LEDGER_KEY = "__context_ledger__"


@dataclass
class ContextFact:
    """A candidate or verified fact extracted from a node update."""

    text: str
    source: str
    confidence: float = 0.5
    verified: bool = False
    node: str = ""
    step: int = 0
    fact_id: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    valid_from: Optional[float] = None
    valid_to: Optional[float] = None
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "confidence": self.confidence,
            "verified": self.verified,
            "node": self.node,
            "step": self.step,
            "fact_id": self.fact_id,
            "evidence_ids": list(self.evidence_ids),
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextFact":
        return cls(
            text=str(data.get("text", "")),
            source=str(data.get("source", "")),
            confidence=float(data.get("confidence", 0.5)),
            verified=bool(data.get("verified", False)),
            node=str(data.get("node", "")),
            step=int(data.get("step", 0)),
            fact_id=str(data.get("fact_id", "")),
            evidence_ids=[str(item) for item in data.get("evidence_ids", [])],
            valid_from=float(data["valid_from"]) if data.get("valid_from") is not None else None,
            valid_to=float(data["valid_to"]) if data.get("valid_to") is not None else None,
            status=str(data.get("status", "active")),
        )


@dataclass
class ToolSummary:
    """Compact record for tool or node output."""

    tool_name: str
    raw_ref: str = ""
    short_summary: str = ""
    extracted_facts: List[ContextFact] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    token_count_raw: int = 0
    token_count_summary: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "raw_ref": self.raw_ref,
            "short_summary": self.short_summary,
            "extracted_facts": [fact.to_dict() for fact in self.extracted_facts],
            "errors": list(self.errors),
            "token_count_raw": self.token_count_raw,
            "token_count_summary": self.token_count_summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolSummary":
        return cls(
            tool_name=str(data.get("tool_name", "")),
            raw_ref=str(data.get("raw_ref", "")),
            short_summary=str(data.get("short_summary", "")),
            extracted_facts=[
                ContextFact.from_dict(item)
                for item in data.get("extracted_facts", [])
                if isinstance(item, dict)
            ],
            errors=[str(item) for item in data.get("errors", [])],
            token_count_raw=int(data.get("token_count_raw", 0)),
            token_count_summary=int(data.get("token_count_summary", 0)),
        )


@dataclass
class FailureSummary:
    """Compact run-context failure record."""

    node: str
    step: int
    error_type: str
    message: str
    recoverable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "step": self.step,
            "error_type": self.error_type,
            "message": self.message,
            "recoverable": self.recoverable,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureSummary":
        return cls(
            node=str(data.get("node", "")),
            step=int(data.get("step", 0)),
            error_type=str(data.get("error_type", "")),
            message=str(data.get("message", "")),
            recoverable=bool(data.get("recoverable", False)),
        )


TODO_STATUSES = {"pending", "in_progress", "completed", "blocked", "cancelled"}


@dataclass
class TodoItem:
    """Structured task-plan item tracked during a run."""

    id: str
    content: str
    status: str = "pending"
    source: str = "current_plan"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    evidence: str = ""
    revision: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "evidence": self.evidence,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TodoItem":
        status = str(data.get("status", "pending"))
        if status not in TODO_STATUSES:
            status = "pending"
        return cls(
            id=str(data.get("id", "")),
            content=str(data.get("content", "")),
            status=status,
            source=str(data.get("source", "current_plan")),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            evidence=str(data.get("evidence", "")),
            revision=int(data.get("revision", 0)),
        )


@dataclass
class TodoEvent:
    """Auditable todo-list mutation record."""

    revision: int
    action: str
    todo_id: str = ""
    before: Dict[str, Any] | None = None
    after: Dict[str, Any] | None = None
    reason: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision": self.revision,
            "action": self.action,
            "todo_id": self.todo_id,
            "before": dict(self.before or {}),
            "after": dict(self.after or {}),
            "reason": self.reason,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TodoEvent":
        before = data.get("before")
        after = data.get("after")
        return cls(
            revision=int(data.get("revision", 0)),
            action=str(data.get("action", "")),
            todo_id=str(data.get("todo_id", "")),
            before=dict(before) if isinstance(before, dict) else {},
            after=dict(after) if isinstance(after, dict) else {},
            reason=str(data.get("reason", "")),
            ts=float(data.get("ts", time.time())),
        )


@dataclass
class ContextBudget:
    """Observed budget fields for the first ledger implementation.

    Full pause and token-budget enforcement is implemented in the next planned
    phase. This object is included now so checkpoints and ledger JSON have a
    stable schema.
    """

    max_context_tokens: int = 0
    reserved_output_tokens: int = 0
    max_memory_tokens: int = 0
    max_tool_summary_tokens: int = 0
    used_context_tokens: int = 0
    compression_count: int = 0
    paused: bool = False
    pause_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_context_tokens": self.max_context_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "max_memory_tokens": self.max_memory_tokens,
            "max_tool_summary_tokens": self.max_tool_summary_tokens,
            "used_context_tokens": self.used_context_tokens,
            "compression_count": self.compression_count,
            "paused": self.paused,
            "pause_reason": self.pause_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "ContextBudget":
        data = data or {}
        return cls(
            max_context_tokens=int(data.get("max_context_tokens", 0)),
            reserved_output_tokens=int(data.get("reserved_output_tokens", 0)),
            max_memory_tokens=int(data.get("max_memory_tokens", 0)),
            max_tool_summary_tokens=int(data.get("max_tool_summary_tokens", 0)),
            used_context_tokens=int(data.get("used_context_tokens", 0)),
            compression_count=int(data.get("compression_count", 0)),
            paused=bool(data.get("paused", False)),
            pause_reason=str(data.get("pause_reason", "")),
        )


@dataclass
class ContextLedger:
    """Run-scoped context skeleton for long-running workflows."""

    run_id: str
    original_goal: str = ""
    hard_constraints: List[str] = field(default_factory=list)
    current_plan: List[str] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)
    pending_steps: List[str] = field(default_factory=list)
    todo_items: List[TodoItem] = field(default_factory=list)
    active_todo_id: str = ""
    todo_revision: int = 0
    todo_events: List[TodoEvent] = field(default_factory=list)
    key_facts: List[ContextFact] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    tool_summaries: List[ToolSummary] = field(default_factory=list)
    failure_summaries: List[FailureSummary] = field(default_factory=list)
    budget: ContextBudget = field(default_factory=ContextBudget)
    next_action: str = ""
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "original_goal": self.original_goal,
            "hard_constraints": list(self.hard_constraints),
            "current_plan": list(self.current_plan),
            "completed_steps": list(self.completed_steps),
            "pending_steps": list(self.pending_steps),
            "todo_items": [item.to_dict() for item in self.todo_items],
            "active_todo_id": self.active_todo_id,
            "todo_revision": self.todo_revision,
            "todo_events": [event.to_dict() for event in self.todo_events],
            "key_facts": [fact.to_dict() for fact in self.key_facts],
            "open_questions": list(self.open_questions),
            "tool_summaries": [summary.to_dict() for summary in self.tool_summaries],
            "failure_summaries": [
                summary.to_dict() for summary in self.failure_summaries
            ],
            "budget": self.budget.to_dict(),
            "next_action": self.next_action,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextLedger":
        return cls(
            run_id=str(data.get("run_id", "")),
            original_goal=str(data.get("original_goal", "")),
            hard_constraints=[str(item) for item in data.get("hard_constraints", [])],
            current_plan=[str(item) for item in data.get("current_plan", [])],
            completed_steps=[str(item) for item in data.get("completed_steps", [])],
            pending_steps=[str(item) for item in data.get("pending_steps", [])],
            todo_items=[
                TodoItem.from_dict(item)
                for item in data.get("todo_items", [])
                if isinstance(item, dict)
            ],
            active_todo_id=str(data.get("active_todo_id", "")),
            todo_revision=int(data.get("todo_revision", 0)),
            todo_events=[
                TodoEvent.from_dict(item)
                for item in data.get("todo_events", [])
                if isinstance(item, dict)
            ],
            key_facts=[
                ContextFact.from_dict(item)
                for item in data.get("key_facts", [])
                if isinstance(item, dict)
            ],
            open_questions=[str(item) for item in data.get("open_questions", [])],
            tool_summaries=[
                ToolSummary.from_dict(item)
                for item in data.get("tool_summaries", [])
                if isinstance(item, dict)
            ],
            failure_summaries=[
                FailureSummary.from_dict(item)
                for item in data.get("failure_summaries", [])
                if isinstance(item, dict)
            ],
            budget=ContextBudget.from_dict(data.get("budget")),
            next_action=str(data.get("next_action", "")),
            updated_at=float(data.get("updated_at", time.time())),
        )
