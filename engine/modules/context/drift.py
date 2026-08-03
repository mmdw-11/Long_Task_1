"""Rule-based context drift detection."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ._types import ContextLedger, ToolSummary


DRIFT_RESULT_KEY = "__context_drift__"


class EmbeddingModel(Protocol):
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError


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
    """Detect long-run drift patterns from ledger state.

    Drift is broader than a literal graph loop. It also includes repeatedly
    producing semantically similar summaries, repeatedly touching the same
    resource, or repeatedly calling the same tool without meaningful progress.
    """

    def __init__(
        self,
        *,
        repeat_node_limit: int = 3,
        pending_step_limit: int = 10,
        no_progress_step_limit: int = 0,
        repeated_summary_limit: int = 3,
        repeated_resource_limit: int = 3,
        repeated_tool_limit: int = 0,
        repeated_file_operation_limit: int = 3,
        goal_similarity_threshold: float = 0.0,
        goal_drift_window: int = 2,
        embedding_model: Optional[EmbeddingModel] = None,
    ) -> None:
        self.repeat_node_limit = repeat_node_limit
        self.pending_step_limit = pending_step_limit
        self.no_progress_step_limit = no_progress_step_limit
        self.repeated_summary_limit = repeated_summary_limit
        self.repeated_resource_limit = repeated_resource_limit
        self.repeated_tool_limit = repeated_tool_limit
        self.repeated_file_operation_limit = repeated_file_operation_limit
        self.goal_similarity_threshold = goal_similarity_threshold
        self.goal_drift_window = goal_drift_window
        self.embedding_model = embedding_model

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
        if self.repeated_summary_limit > 1:
            repeated = _recent_repeated_summary(
                ledger.tool_summaries,
                limit=self.repeated_summary_limit,
            )
            if repeated:
                reasons.append(
                    f"similar output summary repeated {self.repeated_summary_limit} times: {repeated}"
                )
        if self.repeated_resource_limit > 1:
            repeated_resource = _recent_repeated_resource(
                ledger.tool_summaries,
                limit=self.repeated_resource_limit,
            )
            if repeated_resource:
                reasons.append(
                    f"same resource referenced {self.repeated_resource_limit} times: {repeated_resource}"
                )
        if self.repeated_tool_limit > 1:
            repeated_tool = _recent_repeated_tool(
                ledger.tool_summaries,
                limit=self.repeated_tool_limit,
            )
            if repeated_tool:
                reasons.append(
                    f"tool {repeated_tool} repeated {self.repeated_tool_limit} times"
                )
        if self.repeated_file_operation_limit > 1:
            repeated_file = _recent_repeated_file_operation(
                ledger.tool_summaries,
                limit=self.repeated_file_operation_limit,
            )
            if repeated_file:
                reasons.append(
                    f"similar file operation repeated {self.repeated_file_operation_limit} times: {repeated_file}"
                )
        if (
            self.embedding_model is not None
            and self.goal_similarity_threshold > 0
            and ledger.original_goal.strip()
        ):
            goal_drift = _goal_embedding_drift(
                ledger,
                embedding_model=self.embedding_model,
                threshold=self.goal_similarity_threshold,
                window=self.goal_drift_window,
            )
            if goal_drift:
                reasons.append(goal_drift)
        severity = "none"
        if reasons:
            severity = "medium" if len(reasons) == 1 else "high"
        return DriftResult(drifted=bool(reasons), reasons=reasons, severity=severity)


def _recent_repeated_summary(summaries: List[ToolSummary], *, limit: int) -> str:
    recent = [item for item in summaries[-limit:] if item.short_summary.strip()]
    if len(recent) < limit:
        return ""
    fingerprints = [_summary_fingerprint(item.short_summary) for item in recent]
    if all(fingerprint and fingerprint == fingerprints[0] for fingerprint in fingerprints):
        return _clip(recent[-1].short_summary)
    return ""


def _recent_repeated_resource(summaries: List[ToolSummary], *, limit: int) -> str:
    recent_refs = [item.raw_ref for item in summaries[-limit:] if item.raw_ref]
    if len(recent_refs) < limit:
        return ""
    if all(ref == recent_refs[0] for ref in recent_refs):
        return recent_refs[0]
    return ""


def _recent_repeated_tool(summaries: List[ToolSummary], *, limit: int) -> str:
    recent_tools = [item.tool_name for item in summaries[-limit:] if item.tool_name]
    if len(recent_tools) < limit:
        return ""
    if all(tool == recent_tools[0] for tool in recent_tools):
        return recent_tools[0]
    return ""


def _recent_repeated_file_operation(summaries: List[ToolSummary], *, limit: int) -> str:
    recent = [
        _file_operation_fingerprint(item.short_summary)
        for item in summaries[-limit:]
        if item.short_summary.strip()
    ]
    recent = [item for item in recent if item]
    if len(recent) < limit:
        return ""
    if all(item == recent[0] for item in recent):
        return recent[0]
    return ""


def _goal_embedding_drift(
    ledger: ContextLedger,
    *,
    embedding_model: EmbeddingModel,
    threshold: float,
    window: int,
) -> str:
    candidates = _recent_goal_candidates(ledger, window=max(1, window))
    if len(candidates) < max(1, window):
        return ""
    goal_vector = embedding_model.embed(ledger.original_goal)
    similarities = [
        _cosine_similarity(goal_vector, embedding_model.embed(candidate))
        for candidate in candidates[-window:]
        if candidate.strip()
    ]
    if len(similarities) < window:
        return ""
    if all(score < threshold for score in similarities):
        score_text = ", ".join(f"{score:.3f}" for score in similarities)
        return (
            "semantic drift from original goal: "
            f"recent similarity scores [{score_text}] below threshold {threshold:.3f}"
        )
    return ""


def _recent_goal_candidates(ledger: ContextLedger, *, window: int) -> List[str]:
    candidates: List[str] = []
    for summary in ledger.tool_summaries[-window:]:
        if summary.short_summary.strip():
            candidates.append(summary.short_summary)
    if len(candidates) < window and ledger.next_action.strip():
        candidates.append(ledger.next_action)
    if len(candidates) < window:
        candidates.extend(ledger.pending_steps[-(window - len(candidates)) :])
    return candidates


def _summary_fingerprint(text: str) -> str:
    normalized = re.sub(r"\d+", "<num>", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized[:240]


def _file_operation_fingerprint(text: str) -> str:
    paths = _extract_paths(text)
    if not paths:
        return ""
    action = "write"
    lowered = text.lower()
    if any(word in lowered for word in ("delete", "remove", "删除")):
        action = "delete"
    elif any(word in lowered for word in ("read", "open", "读取")):
        action = "read"
    elif any(word in lowered for word in ("modify", "update", "patch", "edit", "修改", "更新")):
        action = "modify"
    return f"{action}:{paths[0].lower()}"


def _extract_paths(text: str) -> List[str]:
    pattern = r"(?:[A-Za-z]:\\)?[\w.-]+(?:[\\/][\w.@()-]+)+(?:\.[A-Za-z0-9_]+)?"
    return re.findall(pattern, text)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(float(left[idx]) * float(right[idx]) for idx in range(size))
    left_norm = sum(float(left[idx]) ** 2 for idx in range(size)) ** 0.5
    right_norm = sum(float(right[idx]) ** 2 for idx in range(size)) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _clip(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
