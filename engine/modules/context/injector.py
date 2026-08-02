"""Build node-scoped context injection blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ._types import ContextLedger


CONTEXT_INJECTION_KEY = "__context_injection__"
CONTEXT_INJECTION_TEXT_KEY = "__context_injection_text__"


@dataclass
class ContextInjection:
    """Context block injected before node execution."""

    original_goal: str = ""
    hard_constraints: List[str] | None = None
    current_node_role: str = ""
    current_step_objective: str = ""
    verified_facts: List[str] | None = None
    things_not_to_assume: List[str] | None = None
    expected_output_contract: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_goal": self.original_goal,
            "hard_constraints": list(self.hard_constraints or []),
            "current_node_role": self.current_node_role,
            "current_step_objective": self.current_step_objective,
            "verified_facts": list(self.verified_facts or []),
            "things_not_to_assume": list(self.things_not_to_assume or []),
            "expected_output_contract": self.expected_output_contract,
        }

    def to_text(self) -> str:
        lines = ["Context Injection:"]
        if self.original_goal:
            lines.append(f"Original Goal: {self.original_goal}")
        if self.hard_constraints:
            lines.append("Non-negotiable Constraints:")
            for item in self.hard_constraints:
                lines.append(f"- {item}")
        if self.current_node_role:
            lines.append(f"Current Node Role: {self.current_node_role}")
        if self.current_step_objective:
            lines.append(f"Current Step Objective: {self.current_step_objective}")
        if self.verified_facts:
            lines.append("Verified Facts:")
            for item in self.verified_facts:
                lines.append(f"- {item}")
        if self.things_not_to_assume:
            lines.append("Things Not To Assume:")
            for item in self.things_not_to_assume:
                lines.append(f"- {item}")
        if self.expected_output_contract:
            lines.append(f"Expected Output Contract: {self.expected_output_contract}")
        return "\n".join(lines)


class ContextInjector:
    """Create scoped context injection from ledger and node metadata."""

    def __init__(self, *, max_verified_facts: int = 8, max_unverified: int = 8) -> None:
        self.max_verified_facts = max_verified_facts
        self.max_unverified = max_unverified

    def build(
        self,
        *,
        ledger: ContextLedger,
        node: str,
        metadata: Dict[str, Any],
    ) -> ContextInjection:
        role = str(
            metadata.get("role")
            or metadata.get("description")
            or metadata.get("sys_prompt")
            or node
        )
        objective = str(
            metadata.get("objective")
            or metadata.get("task")
            or ledger.next_action
            or f"execute node {node}"
        )
        output_contract = str(metadata.get("output_contract") or "")
        verified = [
            fact.text
            for fact in ledger.key_facts
            if fact.verified
        ][-self.max_verified_facts :]
        unverified = [
            fact.text
            for fact in ledger.key_facts
            if not fact.verified
        ][-self.max_unverified :]
        return ContextInjection(
            original_goal=ledger.original_goal,
            hard_constraints=list(ledger.hard_constraints),
            current_node_role=role,
            current_step_objective=objective,
            verified_facts=verified,
            things_not_to_assume=unverified,
            expected_output_contract=output_contract,
        )
