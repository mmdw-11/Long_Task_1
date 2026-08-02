"""Runtime context ledger data structures.

The context ledger is run-scoped state for long-running agent workflows. It is
separate from long-term memory: memory stores reusable knowledge, while the
ledger preserves the current run's goal, constraints, verified facts, progress,
and recent failures in a compact, auditable form.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "confidence": self.confidence,
            "verified": self.verified,
            "node": self.node,
            "step": self.step,
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
