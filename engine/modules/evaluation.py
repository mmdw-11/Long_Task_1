"""Lightweight post-execution validation for node outputs."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class ValidationFailureAction(str, enum.Enum):
    """What the graph should do after post-execution validation fails."""

    RECORD = "record"
    PAUSE = "pause"
    REROUTE = "reroute"


@dataclass
class EvaluationResult:
    """Result of validating a node update."""

    passed: bool
    score: float = 1.0
    findings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    retryable: bool = False
    action: ValidationFailureAction = ValidationFailureAction.RECORD
    targets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "findings": list(self.findings),
            "metrics": dict(self.metrics),
            "retryable": self.retryable,
            "action": self.action.value,
            "targets": list(self.targets),
        }


class Evaluator:
    """Base evaluator interface."""

    def evaluate_node(
        self,
        *,
        node: str,
        update: Optional[Dict[str, Any]],
        state: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> EvaluationResult:
        return EvaluationResult(passed=True)


class RuleEvaluator(Evaluator):
    """Deterministic node-output evaluator.

    Supported metadata keys:
    - ``required_update_keys``: list of keys that must exist in node update.
    - ``required_files``: list of filesystem paths that must exist after node execution.
    - ``output_contract``: currently recorded as metadata; concrete contract checks can
      be added per node by using required keys/files.
    """

    def evaluate_node(
        self,
        *,
        node: str,
        update: Optional[Dict[str, Any]],
        state: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> EvaluationResult:
        findings: List[str] = []
        if update is not None and not isinstance(update, dict):
            findings.append(f"node {node} returned non-dict update")
        update_dict = update if isinstance(update, dict) else {}
        if metadata.get("requires_parent_validation") and not update_dict.get(
            "__parent_validated__"
        ):
            findings.append("sub-agent output requires parent validation")
        for key in _list_value(metadata.get("required_update_keys")):
            if key not in update_dict:
                findings.append(f"missing required update key: {key}")
        for raw_path in _list_value(metadata.get("required_files")):
            path = Path(raw_path)
            if not path.exists():
                findings.append(f"required file does not exist: {raw_path}")
        action = ValidationFailureAction(str(metadata.get("validation_failure_action") or "pause"))
        targets = _list_value(metadata.get("validation_failure_targets"))
        return EvaluationResult(
            passed=not findings,
            score=0.0 if findings else 1.0,
            findings=findings,
            retryable=bool(findings),
            action=action if findings else ValidationFailureAction.RECORD,
            targets=targets,
        )


def _list_value(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]
