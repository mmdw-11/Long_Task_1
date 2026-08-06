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
    active_todo: str = ""
    adjacent_todos: List[str] | None = None
    completed_todos: List[str] | None = None
    todo_policy: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_goal": self.original_goal,
            "hard_constraints": list(self.hard_constraints or []),
            "current_node_role": self.current_node_role,
            "current_step_objective": self.current_step_objective,
            "verified_facts": list(self.verified_facts or []),
            "things_not_to_assume": list(self.things_not_to_assume or []),
            "expected_output_contract": self.expected_output_contract,
            "active_todo": self.active_todo,
            "adjacent_todos": list(self.adjacent_todos or []),
            "completed_todos": list(self.completed_todos or []),
            "todo_policy": self.todo_policy,
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
        if self.active_todo:
            lines.append(f"Active Todo: {self.active_todo}")
        if self.adjacent_todos:
            lines.append("Nearby Todos:")
            for item in self.adjacent_todos:
                lines.append(f"- {item}")
        if self.completed_todos:
            lines.append("Completed Todos:")
            for item in self.completed_todos:
                lines.append(f"- {item}")
        if self.todo_policy:
            lines.append(f"Todo Policy: {self.todo_policy}")
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
        active_todo = _active_todo_text(ledger)
        adjacent_todos = _adjacent_todo_text(ledger)
        completed_todos = [
            f"{item.id}: {item.content}"
            for item in ledger.todo_items
            if item.status == "completed"
        ][-5:]
        return ContextInjection(
            original_goal=ledger.original_goal,
            hard_constraints=list(ledger.hard_constraints),
            current_node_role=role,
            current_step_objective=objective,
            verified_facts=verified,
            things_not_to_assume=unverified,
            expected_output_contract=output_contract,
            active_todo=active_todo,
            adjacent_todos=adjacent_todos,
            completed_todos=completed_todos,
            todo_policy=(
                "Work on the active todo. If the plan is incomplete or stale, "
                "suggest a todo update instead of silently changing direction."
            ),
        )


def _active_todo_text(ledger: ContextLedger) -> str:
    for item in ledger.todo_items:
        if item.id == ledger.active_todo_id:
            return f"{item.id} [{item.status}] {item.content}"
    return ""


def _adjacent_todo_text(ledger: ContextLedger) -> List[str]:
    if not ledger.todo_items:
        return []
    active_index = -1
    for idx, item in enumerate(ledger.todo_items):
        if item.id == ledger.active_todo_id:
            active_index = idx
            break
    if active_index < 0:
        return [
            f"{item.id} [{item.status}] {item.content}"
            for item in ledger.todo_items[:3]
        ]
    start = max(0, active_index - 1)
    end = min(len(ledger.todo_items), active_index + 2)
    return [
        f"{item.id} [{item.status}] {item.content}"
        for item in ledger.todo_items[start:end]
        if item.id != ledger.active_todo_id
    ]
