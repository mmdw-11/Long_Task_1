"""Context policy loading for repeatable harness evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .budget import ContextBudgetController
from .drift import DriftDetector
from .injector import ContextInjector
from .ledger import ContextLedgerStore


@dataclass
class ContextPolicy:
    """Configurable context-management policy."""

    max_context_tokens: int = 0
    reserved_output_tokens: int = 0
    max_memory_tokens: int = 0
    max_tool_summary_tokens: int = 0
    memory_top_k: int = 5
    long_text_threshold: int = 2000
    summary_max_chars: int = 600
    max_verified_facts: int = 8
    max_unverified: int = 8
    repeat_node_limit: int = 3
    pending_step_limit: int = 10
    no_progress_step_limit: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextPolicy":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in allowed})

    @classmethod
    def from_file(cls, path: str | Path) -> "ContextPolicy":
        text = Path(path).read_text(encoding="utf-8")
        if str(path).endswith(".json"):
            return cls.from_dict(json.loads(text))
        return cls.from_dict(_parse_simple_yaml(text))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_context_tokens": self.max_context_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "max_memory_tokens": self.max_memory_tokens,
            "max_tool_summary_tokens": self.max_tool_summary_tokens,
            "memory_top_k": self.memory_top_k,
            "long_text_threshold": self.long_text_threshold,
            "summary_max_chars": self.summary_max_chars,
            "max_verified_facts": self.max_verified_facts,
            "max_unverified": self.max_unverified,
            "repeat_node_limit": self.repeat_node_limit,
            "pending_step_limit": self.pending_step_limit,
            "no_progress_step_limit": self.no_progress_step_limit,
        }

    def build_ledger_store(self, root_dir: str | Path) -> ContextLedgerStore:
        return ContextLedgerStore(
            root_dir,
            long_text_threshold=self.long_text_threshold,
            summary_max_chars=self.summary_max_chars,
        )

    def build_budget_controller(self) -> ContextBudgetController:
        return ContextBudgetController(
            max_context_tokens=self.max_context_tokens,
            reserved_output_tokens=self.reserved_output_tokens,
            max_memory_tokens=self.max_memory_tokens,
            max_tool_summary_tokens=self.max_tool_summary_tokens,
        )

    def build_injector(self) -> ContextInjector:
        return ContextInjector(
            max_verified_facts=self.max_verified_facts,
            max_unverified=self.max_unverified,
        )

    def build_drift_detector(self) -> DriftDetector:
        return DriftDetector(
            repeat_node_limit=self.repeat_node_limit,
            pending_step_limit=self.pending_step_limit,
            no_progress_step_limit=self.no_progress_step_limit,
        )


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Parse the flat key/value YAML subset used for context policy files."""
    data: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _coerce_scalar(value.strip())
    return data


def _coerce_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")
