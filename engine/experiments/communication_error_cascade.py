"""Two-group, real-LLM evaluation of structured low-entropy communication."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..modules.agent_runtime import AgentRuntimeFactory
from ..modules.communication import CommunicationManager, CommunicationPolicy, EvidenceRef, MessageCapsule
from ..modules.communication.compressor import rough_tokens
from ..modules.model_connections import ModelConnectionStore
from ..node import Node
from ..orchestrator import Orchestrator
from .io import read_jsonl, write_json, write_jsonl

DATASET_NAME = "communication_low_entropy_100_v3"
SUBSETS = {"fanin": 40, "multihop": 20, "independent": 20, "conflict_long": 20}
METHODS = ("baseline_full_history", "structured_gate")
TOPICS = ("发布日期", "版本号", "数量", "地点", "规则")


def _fact(topic: str, ordinal: int) -> tuple[str, str]:
    if topic == "发布日期": return f"2026-09-{ordinal % 27 + 1:02d}", f"2026-08-{ordinal % 27 + 1:02d}"
    if topic == "版本号": return f"v{ordinal}.2", f"v{ordinal}.1"
    if topic == "数量": return str(500 + ordinal), str(499 + ordinal)
    if topic == "地点": return f"节点-{ordinal}-A", f"节点-{ordinal}-B"
    return f"规则-{ordinal}-允许", f"规则-{ordinal}-禁止"


def _payload(claim_id: str, fact_key: str, claim: str, *, kind: str = "claim", confidence: float = .2, detail: str = "") -> Dict[str, Any]:
    return {"claim_id": claim_id, "fact_key": fact_key, "claim": claim, "kind": kind, "confidence": confidence, "detail": detail}


def build_dataset(output: str | Path) -> List[Dict[str, Any]]:
    """Create 100 frozen cases with variable fan-in and multi-hop topology."""
    rows: List[Dict[str, Any]] = []; ordinal = 0
    for subset, amount in SUBSETS.items():
        for index in range(1, amount + 1):
            ordinal += 1; topic = TOPICS[(ordinal - 1) % len(TOPICS)]
            correct, wrong = _fact(topic, ordinal); fact_key = f"{topic}_{ordinal}"; wrong_id = f"wrong-{ordinal:03d}"
            goal = f"核验并汇聚 {topic} 的候选事实"; subtask = f"交接 {topic} 的证据与候选结论"
            wrong_payload = _payload(wrong_id, fact_key, wrong, confidence=.10)
            topology = "fanin"; relays: List[Dict[str, Any]] = []; required = [fact_key]; gold = ""
            if subset == "fanin":
                relays = [wrong_payload.copy() for _ in range(2 + (index % 4))]
            elif subset == "multihop":
                topology = "multihop"; relays = [wrong_payload.copy(), wrong_payload.copy(), wrong_payload.copy()]
            elif subset == "independent":
                required = []
                for part in range(2 + (index % 3)):
                    key = f"{fact_key}_part_{part + 1}"
                    relays.append(_payload(f"independent-{ordinal:03d}-{part + 1}", key, f"{topic} 的独立子事实 {part + 1}：值-{ordinal}-{part + 1}", confidence=.55))
                    required.append(key)
            else:
                detail = "补充上下文：" + (f"该条证据来自阶段{ordinal}的长程协作记录；应压缩过程描述，但不得删除结论、来源与置信度。" * 26)
                relays = [wrong_payload, _payload(f"evidence-{ordinal:03d}", fact_key, correct, kind="evidence", confidence=.95, detail=detail)]
                gold = correct
            rows.append({"id": f"{subset}-{index:03d}", "subset": subset, "topology": topology, "goal": goal, "subtask": subtask, "topic": topic, "wrong_claim_id": wrong_id, "wrong_value": wrong, "relay_payloads": relays, "required_fact_keys": required, "gold_final_value": gold})
    validate_dataset(rows); write_jsonl(output, rows); return rows


def validate_dataset(rows: Iterable[Dict[str, Any]]) -> None:
    items = list(rows)
    if len(items) != 100: raise ValueError(f"dataset must contain 100 rows, got {len(items)}")
    if len({row.get("id") for row in items}) != len(items): raise ValueError("case ids must be unique")
    for subset, expected in SUBSETS.items():
        if sum(row.get("subset") == subset for row in items) != expected: raise ValueError(f"{subset} must contain {expected} rows")
    for row in items:
        if not row.get("relay_payloads") or not row.get("required_fact_keys") or not row.get("goal"): raise ValueError(f"invalid case: {row.get('id')}")


def dataset_manifest(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    payload = list(rows); canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"dataset": DATASET_NAME, "count": len(payload), "sha256": hashlib.sha256(canonical.encode()).hexdigest(), "subsets": {key: sum(row.get("subset") == key for row in payload) for key in SUBSETS}}


def load_dataset(path: str | Path) -> List[Dict[str, Any]]:
    rows = list(read_jsonl(path)); validate_dataset(rows); return rows


def _policy_for(method: str) -> CommunicationPolicy:
    if method != "structured_gate": raise ValueError(f"unsupported method: {method}")
    return CommunicationPolicy(max_capsule_tokens=256, min_contribution=.28, redundancy_threshold=.82, preserve_novel_claims=True, novelty_threshold=.70)


def _make_orchestrator(case: Dict[str, Any], model_connection: str, method: str) -> tuple[Orchestrator, List[str]]:
    orch = Orchestrator()
    source = orch.create_agent("source", model=model_connection, config={"role": "source"}, sys_prompt="你是任务信息源。严格遵守用户任务，不要编造事实。")
    relay_ids: List[str] = []; relay_names: List[str] = []
    for index, _ in enumerate(case["relay_payloads"]):
        name = f"relay_{index + 1}"; relay_names.append(name)
        relay_ids.append(orch.create_agent(name, model=model_connection, config={"role": "relay"}, sys_prompt="你是受控 relay。分析上游任务后简洁交接你的候选事实。"))
    aggregator = orch.create_agent("aggregator", model=model_connection, config={"role": "aggregator"}, sys_prompt="你是证据汇聚者。比较同一 fact_key 的不同 claim；若冲突，优先最高置信 evidence。只输出 JSON：{\"facts\":[{\"fact_key\":\"...\",\"value\":\"...\",\"claim_id\":\"...\",\"confidence\":0.0}],\"conflicts\":[\"fact_key\"]}。")
    writer = orch.create_agent("writer", model=model_connection, config={"role": "writer"}, sys_prompt="你是最终裁决者。根据汇聚结果选择最可信值；不得把低置信 claim 当作已证实事实。只输出 JSON：{\"final_value\":\"...\",\"used_claim_ids\":[\"...\"],\"rejected_claim_ids\":[\"...\"],\"conflicts\":[\"...\"]}。")
    if case["topology"] == "multihop":
        orch.connect(source, relay_ids[0]); orch.connect(source, relay_ids[1]); orch.connect(relay_ids[0], relay_ids[2]); orch.connect(relay_ids[1], relay_ids[2]); orch.connect(relay_ids[0], aggregator); orch.connect(relay_ids[1], aggregator); orch.connect(relay_ids[2], aggregator)
    else:
        for relay in relay_ids: orch.connect(source, relay); orch.connect(relay, aggregator)
    orch.connect(aggregator, writer); orch.set_entry(source)
    if method == "structured_gate": orch.set_communication_manager(CommunicationManager(_policy_for(method)))
    return orch, relay_names


def _initial_prompt(case: Dict[str, Any]) -> str:
    return f"任务：{case['goal']}。当前子任务：{case['subtask']}。请等待 relay 交接候选事实与证据。"


def _payload_text(payload: Dict[str, Any]) -> str: return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _case_runtime_factory(runtime: AgentRuntimeFactory, case: Dict[str, Any], relay_ids: List[str]):
    controlled = dict(zip(relay_ids, case["relay_payloads"]))
    def factory(spec):
        base = runtime(spec)
        if spec.name not in controlled: return base
        async def invoke(state: Dict[str, Any]) -> Dict[str, Any]:
            update = await base.invoke(state) or {}; payload = controlled[spec.name]; text = _payload_text(payload)
            update["input"] = text; update[spec.name] = text
            messages = list(update.get("messages") or [])
            if not messages: raise RuntimeError(f"{spec.name} produced no auditable model message")
            messages[-1] = {**dict(messages[-1]), "content": text, "controlled_relay": True}; update["messages"] = messages
            evidence = [EvidenceRef(content=text, source="controlled-gold" if payload["kind"] == "evidence" else "controlled-claim", confidence=float(payload["confidence"]))]
            update["capsule"] = MessageCapsule(sender=spec.name, claim_id=str(payload["claim_id"]), fact_key=str(payload["fact_key"]), kind=str(payload["kind"]), goal=str(case["goal"]), subtask=str(case["subtask"]), claim=str(payload["claim"]), evidence=evidence, next_action=str(payload.get("detail") or ""), metadata={"controlled_relay": True, "case_id": case["id"]})
            return update
        return Node(name=base.name, func=invoke, node_type=base.node_type, metadata=dict(base.metadata))
    return factory


def _node_meta(events: List[Dict[str, Any]], node: str) -> Dict[str, Any]:
    for event in reversed(events):
        if event.get("type") != "node_end" or event.get("node") != node: continue
        for message in (event.get("update") or {}).get("messages") or []:
            if isinstance(message, dict): return dict(((message.get("result") or {}).get("metadata") or {}))
    return {}


def _extract_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict): return value
    for text in reversed(re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", str(value or ""), flags=re.S)):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict): return parsed
        except json.JSONDecodeError: pass
    return {}


def _aggregator_items(method: str, events: List[Dict[str, Any]], relay_names: set[str]) -> List[Dict[str, Any]]:
    if method == "structured_gate": return [dict(event.get("capsule") or {}) for event in events if event.get("type") == "capsule_delivered" and event.get("recipient") == "aggregator"]
    items: List[Dict[str, Any]] = []
    for event in events:
        if event.get("type") != "node_end" or event.get("node") not in relay_names: continue
        for message in (event.get("update") or {}).get("messages") or []:
            try: items.append(json.loads(str(message.get("content") or "")))
            except (json.JSONDecodeError, AttributeError): pass
    return items


def _transport_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    """Measure the semantic relay payload, excluding local trace/checkpoint metadata."""
    evidence = [{"source": str(value.get("source") or value.get("uri") or ""), "confidence": value.get("confidence")} for value in item.get("evidence") or [] if isinstance(value, dict)][:3]
    payload = {key: item.get(key) for key in ("claim_id", "fact_key", "kind", "claim", "next_action") if item.get(key) not in (None, "", [], {})}
    if evidence: payload["evidence"] = evidence
    return payload


def calculate_metrics(case: Dict[str, Any], method: str, events: List[Dict[str, Any]], writer_output: Any, relay_names: set[str]) -> Dict[str, Any]:
    items = _aggregator_items(method, events, relay_names); fact_keys = {str(item.get("fact_key") or "") for item in items}; required = set(case["required_fact_keys"])
    pruned = [event for event in events if event.get("type") == "capsule_pruned" and event.get("recipient") == "aggregator"]
    compressed = [event for event in events if event.get("type") == "capsule_compressed" and event.get("recipient") == "aggregator"]
    final = _extract_json(writer_output); final_value = str(final.get("final_value") or ""); gold = str(case.get("gold_final_value") or "")
    return {"aggregator_input_items": len(items), "error_amplification_factor": sum(str(item.get("claim_id") or "") == str(case["wrong_claim_id"]) for item in items), "critical_fact_recall": len(fact_keys & required) / max(1, len(required)), "false_prune": int(case["subset"] == "independent" and not required.issubset(fact_keys)), "relay_token_proxy": sum(rough_tokens(_transport_payload(item)) for item in items), "compression_events": len(compressed), "redundant_pruned": int(any("redundant" in ((event.get("decision") or {}).get("reasons") or []) for event in pruned)), "writer_accuracy": None if not gold else int(final_value == gold), "false_claim_adopted": None if not gold else int(final_value == str(case["wrong_value"])), "writer_json_valid": int(bool(final))}


async def run_case(case: Dict[str, Any], *, method: str, model_connection: str, models_root: str | Path) -> Dict[str, Any]:
    if method not in METHODS: raise ValueError(f"unsupported method: {method}")
    models = ModelConnectionStore(models_root); connection = models.get(model_connection)
    if not connection.runnable: raise RuntimeError(f"{model_connection!r} is not a tested runnable connection; refusing fallback")
    orch, relay_ids = _make_orchestrator(case, model_connection, method); runtime = AgentRuntimeFactory(model_connection_store=models, max_attempts=1); events: List[Dict[str, Any]] = []; failure = ""; started = time.perf_counter()
    try:
        async for event in orch.build_graph(node_factory=_case_runtime_factory(runtime, case, relay_ids)).astream({"input": _initial_prompt(case), "run_id": f"low-entropy-{method}-{case['id']}", "goal": case["goal"]}): events.append(event)
    except Exception as exc: failure = f"{type(exc).__name__}: {exc}"
    required_nodes = {"source", *relay_ids, "aggregator", "writer"}; observed = {str(event.get("node")) for event in events if event.get("type") == "node_end" and _node_meta(events, str(event.get("node"))).get("connection_id") == model_connection}
    writer_event = next((event for event in reversed(events) if event.get("type") == "node_end" and event.get("node") == "writer"), {}); writer_output = (writer_event.get("update") or {}).get("input", "")
    valid = not failure and observed == required_nodes; metrics = calculate_metrics(case, method, events, writer_output, set(relay_ids)) if valid else {}
    if metrics: metrics["aggregator_prompt_tokens"] = _node_meta(events, "aggregator").get("prompt_tokens")
    return {"case_id": case["id"], "subset": case["subset"], "method": method, "status": "valid" if valid else "invalid", "invalid_reason": "" if valid else (failure or "missing_real_llm_node"), "real_llm_nodes": sorted(observed), "metrics": metrics, "events": events, "writer_output": writer_output, "duration_ms": round((time.perf_counter() - started) * 1000, 2)}


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(rows); report: Dict[str, Any] = {"total_runs": len(items), "valid_runs": sum(row.get("status") == "valid" for row in items), "invalid_runs": sum(row.get("status") != "valid" for row in items), "methods": {}}
    for method in METHODS:
        selected = [row for row in items if row.get("method") == method and row.get("status") == "valid"]
        def mean(key: str, subset: str | None = None):
            values = [row["metrics"].get(key) for row in selected if (subset is None or row["subset"] == subset) and row["metrics"].get(key) is not None]
            return sum(values) / len(values) if values else None
        report["methods"][method] = {"n": len(selected), "mean_aggregator_input_items": mean("aggregator_input_items"), "mean_error_amplification_factor": mean("error_amplification_factor"), "mean_relay_token_proxy": mean("relay_token_proxy"), "mean_aggregator_prompt_tokens": mean("aggregator_prompt_tokens"), "duplicate_prune_rate": mean("redundant_pruned", "fanin"), "critical_fact_recall_independent": mean("critical_fact_recall", "independent"), "false_prune_rate_independent": mean("false_prune", "independent"), "compression_events_conflict_long": mean("compression_events", "conflict_long"), "writer_accuracy_conflict_long": mean("writer_accuracy", "conflict_long"), "false_claim_adoption_conflict_long": mean("false_claim_adopted", "conflict_long")}
    return report


def save_result(row: Dict[str, Any], output_dir: str | Path) -> None:
    root = Path(output_dir) / row["method"]; root.mkdir(parents=True, exist_ok=True); write_json(root / f"{row['case_id']}.json", row)
