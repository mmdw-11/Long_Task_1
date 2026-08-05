"""Rule and LLM-assisted context drift detection."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ._types import ContextLedger, ToolSummary


DRIFT_RESULT_KEY = "__context_drift__"


class EmbeddingModel(Protocol):
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError


@dataclass
class TaskDriftJudgeResult:
    """LLM decision for whether the current action has left the task goal."""

    is_task_drift: bool
    reason: str
    raw: str = ""
    decision: str = ""
    todo_updates: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.decision:
            self.decision = "drifted" if self.is_task_drift else "on_track"
        if self.decision not in {"on_track", "todo_update_needed", "drifted"}:
            self.decision = "drifted" if self.is_task_drift else "on_track"
        self.is_task_drift = self.decision == "drifted"


class TaskDriftJudge(Protocol):
    """Protocol for model-based task-goal drift judges."""

    def judge(
        self,
        original_global_goal: str,
        task_plan_list: List[str],
        current_step_content: str,
    ) -> TaskDriftJudgeResult:
        raise NotImplementedError


class OpenAITaskDriftJudge:
    """OpenAI-compatible LLM judge for task-goal semantic drift."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Using OpenAITaskDriftJudge requires openai. Install with: pip install openai"
            ) from exc

        resolved_api_key = (
            api_key
            or os.environ.get("TASK_DRIFT_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        resolved_base_url = (
            base_url
            or os.environ.get("TASK_DRIFT_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        )
        if not resolved_api_key and resolved_base_url:
            resolved_api_key = "not-needed"
        self._client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
        )
        self._model = (
            model
            or os.environ.get("TASK_DRIFT_MODEL")
            or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        )
        self._temperature = temperature

    def judge(
        self,
        original_global_goal: str,
        task_plan_list: List[str],
        current_step_content: str,
    ) -> TaskDriftJudgeResult:
        prompt = _build_task_drift_prompt(
            original_global_goal=original_global_goal,
            task_plan_list=task_plan_list,
            current_step_content=current_step_content,
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You judge whether an agent action is off-task. "
                        "Return only a JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        return _parse_task_drift_judge_result(content)


@dataclass
class DriftResult:
    """Detected drift state for a run."""

    drifted: bool
    reasons: List[str] = field(default_factory=list)
    severity: str = "none"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drifted": self.drifted,
            "reasons": list(self.reasons),
            "severity": self.severity,
            "metadata": dict(self.metadata),
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
        goal_similarity_high_threshold: float = 0.85,
        goal_similarity_low_threshold: float = 0.35,
        goal_drift_window: int = 2,
        semantic_drift_mode: str = "off",
        semantic_drift_cache_enabled: bool = True,
        todo_update_mode: str = "suggest",
        embedding_model: Optional[EmbeddingModel] = None,
        task_drift_judge: Optional[TaskDriftJudge] = None,
    ) -> None:
        self.repeat_node_limit = repeat_node_limit
        self.pending_step_limit = pending_step_limit
        self.no_progress_step_limit = no_progress_step_limit
        self.repeated_summary_limit = repeated_summary_limit
        self.repeated_resource_limit = repeated_resource_limit
        self.repeated_tool_limit = repeated_tool_limit
        self.repeated_file_operation_limit = repeated_file_operation_limit
        self.goal_similarity_threshold = goal_similarity_threshold
        self.goal_similarity_high_threshold = goal_similarity_high_threshold
        self.goal_similarity_low_threshold = goal_similarity_low_threshold
        self.goal_drift_window = goal_drift_window
        self.semantic_drift_mode = semantic_drift_mode
        self.semantic_drift_cache_enabled = semantic_drift_cache_enabled
        self.todo_update_mode = todo_update_mode
        self.embedding_model = embedding_model
        self.task_drift_judge = task_drift_judge
        self._semantic_cache: Dict[str, TaskDriftJudgeResult] = {}

    def detect(self, ledger: ContextLedger, *, current_node: str = "") -> DriftResult:
        reasons: List[str] = []
        metadata: Dict[str, Any] = {}
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
        goal_drift = self._detect_task_goal_drift(ledger, metadata=metadata)
        if goal_drift:
            reasons.append(goal_drift)
        severity = "none"
        if reasons:
            severity = "medium" if len(reasons) == 1 else "high"
        return DriftResult(
            drifted=bool(reasons),
            reasons=reasons,
            severity=severity,
            metadata=metadata,
        )

    def _detect_task_goal_drift(
        self,
        ledger: ContextLedger,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        metadata = metadata if metadata is not None else {}
        mode = (self.semantic_drift_mode or "off").lower()
        metadata["semantic_drift_mode"] = mode
        metadata["task_drift_judge"] = (
            type(self.task_drift_judge).__name__ if self.task_drift_judge is not None else None
        )
        if mode == "off" or not ledger.original_goal.strip():
            return ""

        if mode == "embedding":
            if self.embedding_model is None or self.goal_similarity_threshold <= 0:
                return ""
            return _goal_embedding_drift(
                ledger,
                embedding_model=self.embedding_model,
                threshold=self.goal_similarity_threshold,
                window=self.goal_drift_window,
            )

        current_step_content = _current_step_content(ledger, window=self.goal_drift_window)
        metadata["current_step_content"] = _clip(current_step_content, limit=240)
        if not current_step_content.strip():
            return ""

        if mode == "hybrid" and self.embedding_model is not None:
            score = _best_goal_similarity(
                ledger.original_goal,
                current_step_content,
                embedding_model=self.embedding_model,
            )
            metadata["goal_similarity_score"] = score
            if score >= self.goal_similarity_high_threshold:
                return ""

        if mode not in {"llm", "hybrid"} or self.task_drift_judge is None:
            return ""

        result = self._judge_task_drift(
            original_global_goal=ledger.original_goal,
            task_plan_list=_todo_plan_context(ledger),
            current_step_content=current_step_content,
        )
        metadata["task_drift_judge_result"] = {
            "is_task_drift": result.is_task_drift,
            "decision": result.decision,
            "reason": result.reason,
            "todo_updates": list(result.todo_updates),
        }
        metadata["todo_update_mode"] = self.todo_update_mode
        if result.decision == "todo_update_needed":
            return ""
        if result.is_task_drift:
            reason = _clip(result.reason or "LLM judged current action off task")
            return f"semantic drift from task goal: {reason}"
        return ""

    def _judge_task_drift(
        self,
        *,
        original_global_goal: str,
        task_plan_list: List[str],
        current_step_content: str,
    ) -> TaskDriftJudgeResult:
        cache_key = _semantic_cache_key(
            original_global_goal,
            task_plan_list,
            current_step_content,
        )
        if self.semantic_drift_cache_enabled and cache_key in self._semantic_cache:
            return self._semantic_cache[cache_key]
        try:
            result = self.task_drift_judge.judge(
                original_global_goal,
                task_plan_list,
                current_step_content,
            )
        except Exception as exc:  # noqa: BLE001 - drift detection must not kill the run
            result = TaskDriftJudgeResult(
                is_task_drift=False,
                reason=f"task drift judge failed: {exc}",
            )
        if self.semantic_drift_cache_enabled:
            self._semantic_cache[cache_key] = result
        return result


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


def _best_goal_similarity(
    goal: str,
    current_step_content: str,
    *,
    embedding_model: EmbeddingModel,
) -> float:
    goal_vector = embedding_model.embed(goal)
    return _cosine_similarity(goal_vector, embedding_model.embed(current_step_content))


def _current_step_content(ledger: ContextLedger, *, window: int) -> str:
    candidates = _recent_goal_candidates(ledger, window=max(1, window))
    if not candidates:
        return ""
    return "\n".join(item for item in candidates if item.strip())


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


def _build_task_drift_prompt(
    *,
    original_global_goal: str,
    task_plan_list: List[str],
    current_step_content: str,
) -> str:
    plan_text = "\n".join(f"- {item}" for item in task_plan_list) or "- None"
    return (
        "【整体顶层任务目标】\n"
        f"{original_global_goal}\n"
        "【前期整体任务拆解规划步骤】\n"
        f"{plan_text}\n"
        "【当前Agent正在执行的动作、输出内容】\n"
        f"{current_step_content}\n\n"
        "请你只输出JSON格式：\n"
        '{"is_task_drift": bool, "reason": "简短原因说明"}\n'
        "规则：\n"
        "1. 如果当前动作属于整体规划内的子步骤、正常任务拆解工作，is_task_drift=false\n"
        "2. 如果当前动作完全脱离主线目标、开始做无关工作、发散闲聊、偏离业务主线，is_task_drift=true\n"
        "3. 只判断是否脱离任务主线，不判断细节实现是否正确\n"
    )


def _parse_task_drift_judge_result(text: str) -> TaskDriftJudgeResult:
    try:
        data = _parse_json_object(text)
    except json.JSONDecodeError:
        return TaskDriftJudgeResult(
            is_task_drift=False,
            reason="task drift judge returned invalid JSON",
            raw=text,
        )
    decision = str(data.get("decision", "")).strip()
    is_task_drift = bool(data.get("is_task_drift", False))
    if not decision:
        decision = "drifted" if is_task_drift else "on_track"
    updates = data.get("todo_updates") or []
    return TaskDriftJudgeResult(
        is_task_drift=is_task_drift,
        reason=str(data.get("reason", "")),
        raw=text,
        decision=decision,
        todo_updates=[item for item in updates if isinstance(item, dict)],
    )


def _parse_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    return json.loads(stripped or "{}")


def _semantic_cache_key(
    original_global_goal: str,
    task_plan_list: List[str],
    current_step_content: str,
) -> str:
    payload = json.dumps(
        {
            "goal": original_global_goal,
            "plan": task_plan_list,
            "current": current_step_content,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _todo_plan_context(ledger: ContextLedger) -> List[str]:
    if not ledger.todo_items:
        return list(ledger.current_plan)
    result: List[str] = []
    for item in ledger.todo_items:
        marker = "ACTIVE" if item.id == ledger.active_todo_id else item.status
        result.append(f"{item.id} [{marker}] {item.content}")
    return result


def _build_task_drift_prompt(
    *,
    original_global_goal: str,
    task_plan_list: List[str],
    current_step_content: str,
) -> str:
    plan_text = "\n".join(f"- {item}" for item in task_plan_list) or "- None"
    return (
        "【整体顶层任务目标】\n"
        f"{original_global_goal}\n"
        "【结构化待办 / 顶层任务拆解】\n"
        f"{plan_text}\n"
        "【当前 Agent 正在执行的动作、输出内容】\n"
        f"{current_step_content}\n\n"
        "请你只输出 JSON 对象：\n"
        '{"decision": "on_track|todo_update_needed|drifted", '
        '"is_task_drift": bool, "reason": "简短原因", '
        '"todo_updates": [{"action": "insert|replace|cancel|complete|select", '
        '"todo_id": "可选", "content": "可选", "reason": "可选"}]}\n'
        "判断规则：\n"
        "1. 当前动作服务于 active todo、相邻 todo 或整体待办主线，decision=on_track。\n"
        "2. 当前动作是在合理维护计划，例如补充缺失步骤、修正过期步骤、切换到更合适的下一步，decision=todo_update_needed。\n"
        "3. 当前动作既不服务于当前/相邻待办，也不是合理修改待办，而是在做无关工作，decision=drifted。\n"
        "4. 只判断是否脱离任务主线，不判断细节实现是否正确。\n"
    )


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
