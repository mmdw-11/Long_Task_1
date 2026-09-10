"""Real-LLM harness for the low-entropy communication error-cascade study.

The false claim is a controlled *input* rather than an LLM hallucination.  All
five graph nodes nevertheless use the configured real model connection.  A run
is invalid (never silently downgraded to an echo/fallback result) unless every
node records a non-simulated invocation of that exact connection.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..modules.agent_runtime import AgentRuntimeFactory
from ..modules.communication import CommunicationManager, CommunicationPolicy, MessageCapsule
from ..modules.model_connections import ModelConnectionStore
from ..node import Node
from ..orchestrator import Orchestrator
from .io import read_jsonl, write_json, write_jsonl


DATASET_NAME = "communication_error_cascade_100_v1"
SUBSETS = {"HS": 70, "PS": 20, "NR": 10}
METHODS = ("baseline_full_history", "structured_gate")
_TOPICS = ("发布日期", "版本号", "数量", "地点", "规则")


def build_dataset(output: str | Path) -> List[Dict[str, Any]]:
    """Create the frozen 70/20/10 mechanism dataset without answer leakage."""
    rows: List[Dict[str, Any]] = []
    ordinal = 0
    for subset, amount in SUBSETS.items():
        for number in range(1, amount + 1):
            ordinal += 1
            topic = _TOPICS[(ordinal - 1) % len(_TOPICS)]
            error_id = f"ERR-{ordinal:03d}"
            correct, wrong = _facts(topic, ordinal)
            if subset == "HS":
                left = f"[{error_id}] 待核验{topic}为：{wrong}。此结论尚未核验。"
                # Exact duplication is intentional: the current dependency-free
                # tokenizer is word/Jaccard based, so this is the unambiguous
                # positive control for the configured 0.82 redundancy threshold.
                right = left
            elif subset == "PS":
                left = f"[{error_id}] 候选记录指出{topic}为{wrong}；请在后续步骤审慎核验该记录。"
                right = f"[{error_id}] 当前有一个未经证实的说法：关于{topic}，上游报告了{wrong}。"
            else:
                # NR is intentionally not a duplicate: the two workers receive
                # different marker/fact pairs, so a redundant prune is wrong.
                right_id = f"ALT-{ordinal:03d}"
                alt = _alternate_fact(topic, ordinal)
                left = f"[{error_id}] 待核验{topic}为：{wrong}。此结论尚未核验。"
                right = f"[{right_id}] 另一条独立候选：{topic}为{alt}，等待核验。"
            rows.append({
                "id": f"{subset}-{number:03d}", "subset": subset,
                "error_id": error_id, "topic": topic,
                "correct_fact": correct, "wrong_fact": wrong,
                "verifier_a_instruction": left, "verifier_b_instruction": right,
            })
    validate_dataset(rows)
    write_jsonl(output, rows)
    return rows


def _facts(topic: str, ordinal: int) -> tuple[str, str]:
    if topic == "发布日期": return (f"2026-09-{(ordinal % 27) + 1:02d}", f"2026-08-{(ordinal % 27) + 1:02d}")
    if topic == "版本号": return (f"v{ordinal}.2", f"v{ordinal}.1")
    if topic == "数量": return (str(100 + ordinal), str(99 + ordinal))
    if topic == "地点": return (f"节点-{ordinal}-A", f"节点-{ordinal}-B")
    return (f"规则-{ordinal}-允许", f"规则-{ordinal}-禁止")


def _alternate_fact(topic: str, ordinal: int) -> str:
    return _facts(topic, ordinal + 200)[1]


def validate_dataset(rows: Iterable[Dict[str, Any]]) -> None:
    items = list(rows)
    if len(items) != 100:
        raise ValueError(f"communication dataset must have 100 rows, got {len(items)}")
    ids = [str(row.get("id") or "") for row in items]
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError("case ids must be unique and non-empty")
    for subset, expected in SUBSETS.items():
        selected = [row for row in items if row.get("subset") == subset]
        if len(selected) != expected:
            raise ValueError(f"{subset} must contain {expected} rows")
    for row in items:
        required = ("error_id", "correct_fact", "wrong_fact", "verifier_a_instruction", "verifier_b_instruction")
        if any(not str(row.get(key) or "").strip() for key in required):
            raise ValueError(f"{row.get('id')} misses required controlled fields")


def dataset_manifest(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    payload = list(rows)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"dataset": DATASET_NAME, "count": len(payload), "sha256": hashlib.sha256(canonical.encode()).hexdigest(), "subsets": {name: sum(1 for row in payload if row.get("subset") == name) for name in SUBSETS}}


def load_dataset(path: str | Path) -> List[Dict[str, Any]]:
    rows = list(read_jsonl(path)); validate_dataset(rows); return rows


def _make_orchestrator(model_connection: str, *, structured: bool) -> Orchestrator:
    """Build an identical real-LLM graph; only communication mode differs."""
    orch = Orchestrator()
    source = orch.create_agent("source", model=model_connection, description="Controlled upstream candidate-claim sender.", sys_prompt="你在受控通信实验中工作。只复述任务输入里方括号标识的候选主张；不要纠正、解释或省略标识。")
    va = orch.create_agent("verifier_A", model=model_connection, description="First independent downstream relay.", sys_prompt="你在错误传播测量实验中工作。只输出任务输入或上游通信中包含方括号标识的候选主张，保持标识和事实值不变；不要判断真伪。")
    vb = orch.create_agent("verifier_B", model=model_connection, description="Second independent downstream relay.", sys_prompt="你在错误传播测量实验中工作。只输出任务输入或上游通信中包含方括号标识的候选主张，保持标识和事实值不变；不要判断真伪。")
    agg = orch.create_agent("aggregator", model=model_connection, description="Merge received candidate claims.", sys_prompt="你在错误传播测量实验中工作。列出收到的每条带方括号标识的候选主张；不要纠正、推断或省略标识。")
    writer = orch.create_agent("writer", model=model_connection, description="Final relay for cascade observation.", sys_prompt="你在错误传播测量实验中工作。只转述上游汇总中带方括号标识的候选主张；不要纠正、推断或省略标识。")
    orch.connect(source, va); orch.connect(source, vb)
    orch.connect(va, agg); orch.connect(vb, agg); orch.connect(agg, writer)
    orch.set_entry(source)
    if structured:
        orch.set_communication_manager(CommunicationManager(CommunicationPolicy()))
    return orch


def _initial_prompt(case: Dict[str, Any]) -> str:
    # The source gets one controlled candidate. The two parallel relays receive
    # their wording via their role-specific system prompts only after source;
    # to retain paired input, both variants are included as untrusted text.
    return (
        f"实验编号：{case['id']}。错误标识：[{case['error_id']}]。\n"
        f"候选错误事实：{case['topic']} = {case['wrong_fact']}。\n"
        "这是受控的未核验候选，必须原样转述标识和候选事实，不得改为正确事实。"
    )


def _case_runtime_factory(runtime: AgentRuntimeFactory, case: Dict[str, Any]):
    """Keep relay content controlled while preserving a real LLM call per node.

    The wrapper is intentionally narrow: it invokes the normal AgentRuntime
    first, retains its non-simulated result metadata for audit, and only then
    substitutes the three relay payloads with frozen text.  This avoids
    treating random model paraphrase as a communication-method effect.
    """
    controlled = {
        "source": f"[{case['error_id']}] 待核验{case['topic']}为：{case['wrong_fact']}。此结论尚未核验。",
        "verifier_A": case["verifier_a_instruction"],
        "verifier_B": case["verifier_b_instruction"],
    }

    def factory(spec):
        base = runtime(spec)
        if spec.name not in controlled:
            return base

        async def invoke(state: Dict[str, Any]) -> Dict[str, Any]:
            update = await base.invoke(state) or {}
            text = controlled[spec.name]
            update["input"] = text
            update[spec.name] = text
            messages = list(update.get("messages") or [])
            if not messages:
                raise RuntimeError(f"{spec.name} real LLM returned no auditable message")
            messages[-1] = {**dict(messages[-1]), "content": text, "controlled_relay": True}
            update["messages"] = messages
            # Keep the semantic fields identical across A/B in the HS positive
            # control.  Legacy conversion otherwise uses each node name as
            # ``subtask``, yielding Jaccard 0.8 and missing the configured
            # 0.82 redundancy threshold despite identical claims.
            # ``__capsule_outbox__`` is intentionally runtime-private and is
            # removed by graph update sanitisation; ``capsule`` is the public
            # explicit protocol field consumed by CommunicationManager.
            update["capsule"] = MessageCapsule(
                sender=spec.name, goal="controlled error cascade", subtask="relay",
                claim=text,
            )
            return update

        return Node(name=base.name, func=invoke, node_type=base.node_type, metadata=dict(base.metadata))
    return factory


def _real_model_nodes(events: Iterable[Dict[str, Any]], connection_id: str) -> set[str]:
    seen: set[str] = set()
    for event in events:
        if event.get("type") != "node_end":
            continue
        update = event.get("update") or {}
        for message in update.get("messages") or []:
            result = message.get("result") if isinstance(message, dict) else {}
            meta = (result or {}).get("metadata") or {}
            if not meta.get("simulated") and meta.get("connection_id") == connection_id:
                seen.add(str(event.get("node") or message.get("agent") or ""))
    return seen


def _marker_count(value: Any, marker: str) -> int:
    return len(re.findall(re.escape(marker), json.dumps(value, ensure_ascii=False, default=str)))


def calculate_metrics(case: Dict[str, Any], method: str, events: List[Dict[str, Any]], final_output: Any) -> Dict[str, Any]:
    """Compute trace-based metrics; invalid runs are excluded by the caller."""
    marker = f"[{case['error_id']}]"
    delivered = [event for event in events if event.get("type") == "capsule_delivered" and event.get("recipient") == "aggregator" and _marker_count(event.get("capsule"), marker)]
    # A duplicate can be pruned either by the explicit redundancy threshold or
    # because its high redundancy lowers the contribution score below the
    # gate.  Both are legitimate duplicate-suppression paths in the current
    # policy, so REPR must not count only the former event label.
    pruned = [event for event in events if event.get("type") == "capsule_pruned" and event.get("recipient") == "aggregator"]
    duplicate_pruned = [event for event in pruned if "redundant" in ((event.get("decision") or {}).get("reasons") or []) or float((((event.get("decision") or {}).get("components") or {}).get("redundancy") or 0)) >= .8]
    verifier_outputs = [event for event in events if event.get("type") == "node_end" and event.get("node") in {"verifier_A", "verifier_B"} and _marker_count(event.get("update"), marker)]
    # In B0 the complete shared messages are inserted into aggregator's prompt;
    # in Ours the only eligible communication input is delivered capsules.
    eac = len(delivered) if method == "structured_gate" else len(verifier_outputs)
    return {
        "error_amplification_count": eac,
        "error_reaches_writer": int(_marker_count(final_output, marker) > 0),
        "redundant_error_pruned": int(bool(duplicate_pruned)),
        "false_prune": int(case["subset"] == "NR" and bool(pruned)),
        "aggregator_delivered_capsules": len(delivered),
        "aggregator_redundancy_scores": [((event.get("decision") or {}).get("components") or {}).get("redundancy") for event in pruned],
    }


async def run_case(case: Dict[str, Any], *, method: str, model_connection: str, models_root: str | Path) -> Dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    models = ModelConnectionStore(models_root)
    connection = models.get(model_connection)
    if not connection.runnable:
        raise RuntimeError(f"{model_connection!r} is not a tested runnable connection; refusing any fallback")
    structured = method == "structured_gate"
    orch = _make_orchestrator(model_connection, structured=structured)
    runtime = AgentRuntimeFactory(model_connection_store=models, max_attempts=1)
    started = time.perf_counter(); events: List[Dict[str, Any]] = []
    failure = ""
    try:
        async for event in orch.build_graph(node_factory=_case_runtime_factory(runtime, case)).astream({"input": _initial_prompt(case), "run_id": f"communication-{method}-{case['id']}", "goal": "受控错误级联验证"}):
            events.append(event)
    except Exception as exc:  # Keep failed evidence; never substitute an answer.
        failure = f"{type(exc).__name__}: {exc}"
    writer = next((event for event in reversed(events) if event.get("type") == "node_end" and event.get("node") == "writer"), {})
    output = (writer.get("update") or {}).get("input", "")
    real_nodes = _real_model_nodes(events, model_connection)
    required_nodes = {"source", "verifier_A", "verifier_B", "aggregator", "writer"}
    protocol_nodes = {name for name in ("source", "verifier_A", "verifier_B") if any(event.get("type") == "node_end" and event.get("node") == name and f"[{case['error_id']}]" in json.dumps(event.get("update") or {}, ensure_ascii=False, default=str) for event in events)}
    valid = not failure and real_nodes == required_nodes and protocol_nodes == {"source", "verifier_A", "verifier_B"}
    metrics = calculate_metrics(case, method, events, output) if valid else {}
    return {
        "case_id": case["id"], "subset": case["subset"], "method": method,
        "status": "valid" if valid else "invalid", "invalid_reason": "" if valid else (failure or ("missing_real_llm_nodes=" + ",".join(sorted(required_nodes - real_nodes)) if real_nodes != required_nodes else "relay_protocol_not_observed")),
        "real_llm_nodes": sorted(real_nodes), "metrics": metrics, "events": events,
        "writer_output": output, "duration_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(rows); result: Dict[str, Any] = {"total_runs": len(items), "valid_runs": sum(row.get("status") == "valid" for row in items), "invalid_runs": sum(row.get("status") != "valid" for row in items), "methods": {}}
    for method in METHODS:
        selected = [row for row in items if row.get("method") == method and row.get("status") == "valid"]
        metrics = [row["metrics"] for row in selected]
        result["methods"][method] = {
            "n": len(selected), "mean_eac": sum(item["error_amplification_count"] for item in metrics) / len(metrics) if metrics else None,
            "writer_ecr": sum(item["error_reaches_writer"] for item in metrics) / len(metrics) if metrics else None,
            "repr_high_similarity": (sum(item["redundant_error_pruned"] for row, item in zip(selected, metrics) if row["subset"] == "HS") / max(1, sum(row["subset"] == "HS" for row in selected))) if method == "structured_gate" else None,
            "fpr_nonrepeat": (sum(item["false_prune"] for row, item in zip(selected, metrics) if row["subset"] == "NR") / max(1, sum(row["subset"] == "NR" for row in selected))) if method == "structured_gate" else None,
        }
    return result


def save_result(row: Dict[str, Any], output_dir: str | Path) -> None:
    root = Path(output_dir) / row["method"]; root.mkdir(parents=True, exist_ok=True)
    write_json(root / f"{row['case_id']}.json", row)
