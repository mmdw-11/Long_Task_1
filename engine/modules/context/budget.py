"""Context budget accounting and pause decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

from ._types import ContextBudget


RUN_STATUS_KEY = "__run_status__"
PAUSE_REASON_KEY = "__pause_reason__"
BUDGET_PAUSED = "budget_paused"


@dataclass
class BudgetDecision:
    """Result of checking context budget for a step."""

    allowed: bool
    used_context_tokens: int
    pause_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "used_context_tokens": self.used_context_tokens,
            "pause_reason": self.pause_reason,
        }


class ContextBudgetController:
    """Estimate and enforce context budget.

    This controller is intentionally model-free. It uses a conservative rough
    token estimate so the graph can enforce guardrails even when tokenizer
    dependencies are unavailable.
    """

    def __init__(
        self,
        *,
        max_context_tokens: int = 0,
        reserved_output_tokens: int = 0,
        max_memory_tokens: int = 0,
        max_tool_summary_tokens: int = 0,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.max_memory_tokens = max_memory_tokens
        self.max_tool_summary_tokens = max_tool_summary_tokens

    def check(self, state: Dict[str, Any]) -> BudgetDecision:
        used = rough_token_count(state)
        if self.max_context_tokens <= 0:
            return BudgetDecision(allowed=True, used_context_tokens=used)
        effective_limit = max(0, self.max_context_tokens - self.reserved_output_tokens)
        if used <= effective_limit:
            return BudgetDecision(allowed=True, used_context_tokens=used)
        return BudgetDecision(
            allowed=False,
            used_context_tokens=used,
            pause_reason=(
                f"context budget exceeded: used {used}, limit {effective_limit}, "
                f"reserved_output_tokens {self.reserved_output_tokens}"
            ),
        )

    def budget_from_decision(self, decision: BudgetDecision) -> ContextBudget:
        return ContextBudget(
            max_context_tokens=self.max_context_tokens,
            reserved_output_tokens=self.reserved_output_tokens,
            max_memory_tokens=self.max_memory_tokens,
            max_tool_summary_tokens=self.max_tool_summary_tokens,
            used_context_tokens=decision.used_context_tokens,
            paused=not decision.allowed,
            pause_reason=decision.pause_reason,
        )


def rough_token_count(value: Any) -> int:
    text = _stringify(value)
    if not text:
        return 0
    # Mixed Chinese/English approximation. It intentionally overestimates a bit
    # for short structured JSON so budget checks fail before the provider does.
    return max(1, len(text) // 4)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
