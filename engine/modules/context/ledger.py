"""Context ledger persistence and update logic."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ._types import (
    ContextFact,
    ContextLedger,
    FailureSummary,
    ToolSummary,
)
from .renderer import ContextLedgerRenderer
from .compressor import ContextCompressor


class ContextLedgerStore:
    """File-backed context ledger store.

    Each run is stored as ``<root_dir>/<run_id>/ledger.json``. The class keeps
    update logic intentionally deterministic and lightweight so it can run on
    every node transition without adding model calls.
    """

    def __init__(
        self,
        root_dir: str | Path = "runs/context",
        *,
        max_completed_steps: int = 200,
        max_key_facts: int = 200,
        max_tool_summaries: int = 200,
        max_failures: int = 100,
        long_text_threshold: int = 2000,
        summary_max_chars: int = 600,
        memory_filename: str = "MEMORY.md",
        renderer: Optional[ContextLedgerRenderer] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.max_completed_steps = max_completed_steps
        self.max_key_facts = max_key_facts
        self.max_tool_summaries = max_tool_summaries
        self.max_failures = max_failures
        self.summary_max_chars = summary_max_chars
        self.memory_filename = memory_filename
        self.renderer = renderer or ContextLedgerRenderer()
        self.compressor = ContextCompressor(
            archive_dir=self.root_dir / "raw",
            long_text_threshold=long_text_threshold,
            summary_max_chars=summary_max_chars,
        )

    def load_or_create(self, run_id: str, initial_state: Optional[Dict[str, Any]] = None) -> ContextLedger:
        path = self.path_for(run_id)
        if path.exists():
            return ContextLedger.from_dict(json.loads(path.read_text(encoding="utf-8")))
        ledger = self._new_ledger(run_id, initial_state or {})
        self.save(ledger)
        return ledger

    def save(self, ledger: ContextLedger) -> None:
        ledger.touch()
        path = self.path_for(ledger.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(ledger.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.renderer.write_markdown(ledger, self.memory_path_for(ledger.run_id))

    def path_for(self, run_id: str) -> Path:
        return self.root_dir / _safe_id(run_id) / "ledger.json"

    def memory_path_for(self, run_id: str) -> Path:
        return self.root_dir / _safe_id(run_id) / self.memory_filename

    def on_step_start(
        self,
        *,
        run_id: str,
        step: int,
        frontier: List[str],
        state: Dict[str, Any],
    ) -> ContextLedger:
        ledger = self.load_or_create(run_id, state)
        ledger.next_action = f"step {step}: execute {', '.join(frontier)}"
        existing_pending = set(ledger.pending_steps)
        for node in frontier:
            label = f"{step}:{node}"
            if label not in existing_pending:
                ledger.pending_steps.append(label)
        self.save(ledger)
        return ledger

    def on_node_end(
        self,
        *,
        run_id: str,
        node: str,
        step: int,
        update: Optional[Dict[str, Any]],
        state: Dict[str, Any],
        verified: bool = False,
    ) -> ContextLedger:
        ledger = self.load_or_create(run_id, state)
        label = f"{step}:{node}"
        if label not in ledger.completed_steps:
            ledger.completed_steps.append(label)
        if label in ledger.pending_steps:
            ledger.pending_steps.remove(label)

        if update:
            compressed = self.compressor.compress_update(
                run_id=run_id,
                node=node,
                step=step,
                update=update,
            )
            summary_text = str(compressed["short_summary"])
            if summary_text:
                tool_summary = ToolSummary(
                    tool_name=node,
                    raw_ref=str(compressed["raw_ref"]),
                    short_summary=summary_text,
                    extracted_facts=[
                        ContextFact(
                            text=summary_text,
                            source="node_update",
                            confidence=0.6,
                            verified=verified,
                            node=node,
                            step=step,
                        )
                    ],
                    token_count_raw=int(compressed["token_count_raw"]),
                    token_count_summary=int(compressed["token_count_summary"]),
                )
                ledger.tool_summaries.append(tool_summary)
                self._append_unique_fact(ledger, tool_summary.extracted_facts[0])

        ledger.next_action = ""
        self._trim(ledger)
        self.save(ledger)
        return ledger

    def on_node_error(
        self,
        *,
        run_id: str,
        node: str,
        step: int,
        error: BaseException,
        state: Dict[str, Any],
        recoverable: bool = False,
    ) -> ContextLedger:
        ledger = self.load_or_create(run_id, state)
        ledger.failure_summaries.append(
            FailureSummary(
                node=node,
                step=step,
                error_type=type(error).__name__,
                message=str(error),
                recoverable=recoverable,
            )
        )
        ledger.next_action = f"recover from {node} failure"
        self._trim(ledger)
        self.save(ledger)
        return ledger

    def _new_ledger(self, run_id: str, state: Dict[str, Any]) -> ContextLedger:
        goal = _first_present(state, ("original_goal", "goal", "task", "input", "query"))
        constraints = _coerce_str_list(
            state.get("hard_constraints")
            or state.get("constraints")
            or state.get("requirements")
            or []
        )
        plan = _coerce_str_list(state.get("current_plan") or state.get("plan") or [])
        return ContextLedger(
            run_id=run_id,
            original_goal=self._clip(self._stringify(goal)) if goal is not None else "",
            hard_constraints=constraints,
            current_plan=plan,
        )

    def _append_unique_fact(self, ledger: ContextLedger, fact: ContextFact) -> None:
        seen = {(item.text, item.node, item.step) for item in ledger.key_facts}
        key = (fact.text, fact.node, fact.step)
        if key not in seen:
            ledger.key_facts.append(fact)

    def _trim(self, ledger: ContextLedger) -> None:
        ledger.completed_steps = ledger.completed_steps[-self.max_completed_steps :]
        ledger.pending_steps = ledger.pending_steps[-self.max_completed_steps :]
        ledger.key_facts = ledger.key_facts[-self.max_key_facts :]
        ledger.tool_summaries = ledger.tool_summaries[-self.max_tool_summaries :]
        ledger.failure_summaries = ledger.failure_summaries[-self.max_failures :]

    def _clip(self, text: str) -> str:
        if len(text) <= self.summary_max_chars:
            return text
        return text[: self.summary_max_chars - 3] + "..."

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _first_present(state: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = state.get(key)
        if value not in (None, ""):
            return value
    return None


def _coerce_str_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe or "default-run"


def _rough_token_count(text: str) -> int:
    if not text:
        return 0
    # Conservative mixed-language approximation.
    return max(1, len(text) // 4)
