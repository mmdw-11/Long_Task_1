"""Rule-based context drift detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ._types import ContextLedger


DRIFT_RESULT_KEY = "__context_drift__"


@dataclass
class DriftResult:
    """Detected drift state for a run."""

    drifted: bool
    reasons: List[str] = field(default_factory=list)
    severity: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drifted": self.drifted,
            "reasons": list(self.reasons),
            "severity": self.severity,
        }


class DriftDetector:
    """Detect simple long-run drift patterns from ledger state."""

    def __init__(
        self,
        *,
        repeat_node_limit: int = 3,
        pending_step_limit: int = 10,
        no_progress_step_limit: int = 0,
    ) -> None:
        self.repeat_node_limit = repeat_node_limit
        self.pending_step_limit = pending_step_limit
        self.no_progress_step_limit = no_progress_step_limit

    def detect(self, ledger: ContextLedger, *, current_node: str = "") -> DriftResult:
        reasons: List[str] = []
        if current_node and self.repeat_node_limit > 0:
            recent = ledger.completed_steps[-self.repeat_node_limit :]
            recent_nodes = [item.split(":", 1)[1] for item in recent if ":" in item]
            if len(recent_nodes) >= self.repeat_node_limit and all(
                node == current_node for node in recent_nodes
            ):
                reasons.append(
                    f"node {current_node} repeated {self.repeat_node_limit} times"
                )
        if self.pending_step_limit and len(ledger.pending_steps) > self.pending_step_limit:
            reasons.append(f"pending steps exceeded {self.pending_step_limit}")
        if (
            self.no_progress_step_limit
            and len(ledger.pending_steps) >= self.no_progress_step_limit
            and not ledger.completed_steps
        ):
            reasons.append(f"no completed progress for {self.no_progress_step_limit} steps")
        severity = "none"
        if reasons:
            severity = "medium" if len(reasons) == 1 else "high"
        return DriftResult(drifted=bool(reasons), reasons=reasons, severity=severity)
