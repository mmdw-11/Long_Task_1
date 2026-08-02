"""Render context ledgers into human-readable run memory."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from ._types import ContextFact, ContextLedger, FailureSummary, ToolSummary


class ContextLedgerRenderer:
    """Render ``ContextLedger`` to the first run-scoped ``MEMORY.md`` format."""

    def render_markdown(self, ledger: ContextLedger) -> str:
        lines: List[str] = ["# Run Memory", ""]
        self._section(lines, "Original Goal", [ledger.original_goal] if ledger.original_goal else [])
        self._section(lines, "Hard Constraints", ledger.hard_constraints)
        self._section(lines, "Current Plan", ledger.current_plan)
        self._section(lines, "Completed", ledger.completed_steps)
        self._section(lines, "Pending", ledger.pending_steps)
        self._section(lines, "Verified Facts", self._facts(ledger.key_facts, verified=True))
        self._section(lines, "Unverified Assumptions", self._facts(ledger.key_facts, verified=False))
        self._section(lines, "Key Files And Resources", self._resources(ledger.tool_summaries))
        self._section(lines, "Recent Failures", self._failures(ledger.failure_summaries))
        self._section(lines, "Budget", self._budget(ledger))
        self._section(lines, "Next Action", [ledger.next_action] if ledger.next_action else [])
        return "\n".join(lines).rstrip() + "\n"

    def write_markdown(self, ledger: ContextLedger, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render_markdown(ledger), encoding="utf-8")
        return target

    def _section(self, lines: List[str], title: str, items: Iterable[str]) -> None:
        lines.append(f"## {title}")
        values = [str(item).strip() for item in items if str(item).strip()]
        if not values:
            lines.append("")
            lines.append("- None")
            lines.append("")
            return
        for item in values:
            lines.append(f"- {item}")
        lines.append("")

    def _facts(self, facts: List[ContextFact], *, verified: bool) -> List[str]:
        result: List[str] = []
        for fact in facts:
            if fact.verified != verified:
                continue
            result.append(
                f"{fact.text} (source={fact.source}, node={fact.node}, step={fact.step}, confidence={fact.confidence:.2f})"
            )
        return result

    def _resources(self, summaries: List[ToolSummary]) -> List[str]:
        result: List[str] = []
        for summary in summaries[-20:]:
            if summary.raw_ref:
                result.append(f"{summary.tool_name}: {summary.raw_ref}")
        return result

    def _failures(self, failures: List[FailureSummary]) -> List[str]:
        return [
            f"step {failure.step} {failure.node}: {failure.error_type}: {failure.message}"
            for failure in failures[-20:]
        ]

    def _budget(self, ledger: ContextLedger) -> List[str]:
        budget = ledger.budget
        return [
            f"used_context_tokens={budget.used_context_tokens}",
            f"max_context_tokens={budget.max_context_tokens}",
            f"paused={budget.paused}",
            f"pause_reason={budget.pause_reason or 'None'}",
        ]
