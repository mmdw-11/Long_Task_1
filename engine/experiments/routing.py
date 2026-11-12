"""Paired AUTO versus all-cloud routing experiment."""

from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from engine.modules.agent_runtime import AgentRuntimeFactory
from engine.modules.model_connections import ModelConnection, ModelConnectionStore
from engine.modules.scheduling import ResourceRequest

from .io import write_json, write_jsonl


@dataclass(frozen=True)
class RoutingCase:
    case_id: str
    input: str
    evaluator_type: str
    reference: Any
    stratum: str = "unknown"
    dataset: str = "custom"
    system_prompt: str = ""
    metadata: Dict[str, Any] | None = None


@dataclass(frozen=True)
class Pricing:
    input_per_million_cny: float
    output_per_million_cny: float


class EnvironmentRoutingStore:
    """Ephemeral tier mapping from the project's .env for routing experiments.

    Credentials remain in ``DEEPSEEK_API_KEY`` / ``OPENAI_API_KEY`` and are never
    persisted.  It deliberately maps the same remote model to each tier, so it
    must not be used to claim an edge/device cost advantage.
    """

    def __init__(self, *, device_model: str, device_url: str, edge_model: str,
                 edge_url: str, cloud_model: str, cloud_url: str) -> None:
        values = {
            "device": (device_model, device_url, "ollama"),
            "edge": (edge_model, edge_url, "openai-compatible" if edge_url.rstrip("/").endswith("/v1") else "edge-http"),
            "cloud": (cloud_model, cloud_url, "deepseek"),
        }
        self.items = {
            tier: ModelConnection(
                id=f"env-routing-{tier}", name=f".env {tier} model",
                provider=provider, model_id=model_id,
                base_url=base_url.rstrip("/"), auto_tiers=[tier],
                test_status="succeeded",
            )
            for tier, (model_id, base_url, provider) in values.items()
        }

    def auto_status(self) -> Dict[str, Any]:
        return {"ready": True, "tiers": {
            tier: {"ready": True, "connection": item.to_dict(), "reason": ""}
            for tier, item in self.items.items()
        }}

    def default_for_tier(self, tier: str, *, runnable: bool = True) -> ModelConnection | None:
        return self.items.get(tier)

    def get(self, item_id: str) -> ModelConnection:
        for item in self.items.values():
            if item.id == item_id:
                return item
        raise KeyError(item_id)

    @classmethod
    def from_dotenv(cls) -> "EnvironmentRoutingStore":
        import os
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        device_url = os.environ.get("DEVICE_BASE_URL") or os.environ.get("DEVICE_ENDPOINT") or "http://127.0.0.1:11434/v1"
        edge_ollama_url = os.environ.get("EDGE_OLLAMA_BASE_URL")
        edge_url = edge_ollama_url or os.environ.get("EDGE_ENDPOINT") or os.environ.get("EDGE_BASE_URL") or "http://127.0.0.1:8001/infer"
        cloud_url = os.environ.get("OPENAI_BASE_URL") or "https://api.deepseek.com/v1"
        cloud_url = cloud_url.rstrip("/")
        if not cloud_url.endswith("/v1"):
            cloud_url += "/v1"
        return cls(
            device_model=os.environ.get("DEVICE_MODEL") or "",
            device_url=device_url,
            edge_model=os.environ.get("EDGE_MODEL") or "",
            edge_url=edge_url,
            cloud_model=os.environ.get("OPENAI_MODEL") or "deepseek-v4-flash",
            cloud_url=cloud_url,
        )


EnvironmentDeepSeekStore = EnvironmentRoutingStore


class RoutingExperimentRunner:
    """Compare routing with a fixed cloud baseline and record every attempt."""

    def __init__(
        self, connections: ModelConnectionStore, pricing: Dict[str, Pricing], *,
        auto_arm: str = "strict_auto", validated_local_tiers: tuple[str, ...] = (),
        accuracy_margin_pp: float = 5.0,
    ) -> None:
        if auto_arm not in {"strict_auto", "quality_auto"}:
            raise ValueError("unknown AUTO experiment arm")
        if not 0 <= accuracy_margin_pp <= 100:
            raise ValueError("accuracy margin must be between 0 and 100 percentage points")
        if set(validated_local_tiers) - {"device", "edge"}:
            raise ValueError("validated local tiers must be device or edge")
        self.connections = connections
        self.pricing = pricing
        self.auto_arm = auto_arm
        self.quality_tiers = ("cloud", *validated_local_tiers)
        self.accuracy_margin_pp = accuracy_margin_pp
        self.runtime = AgentRuntimeFactory(model_connection_store=connections)

    def run_case(self, case: RoutingCase, arm: str) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if arm not in {"all_cloud", self.auto_arm}:
            raise ValueError(f"arm must be all_cloud or {self.auto_arm}")
        started = time.perf_counter()
        if arm == "all_cloud":
            connection = self.connections.default_for_tier("cloud")
            if connection is None:
                raise RuntimeError("cloud AUTO default connection is not runnable")
            result = self.runtime._run_pinned_model(connection.id, case.input, case.system_prompt)
            attempts = [self._attempt_from_result(result, "cloud", connection.id, connection.model_id)]
            selected_tier = actual_tier = "cloud"
            gate = "fixed_cloud"
            router_backend = ""
            router_score = None
        else:
            metadata = {**(case.metadata or {}), "strict_route": arm == "strict_auto"}
            if arm == "quality_auto":
                metadata.update(placement_mode="constrained", placement_quality_tiers=self.quality_tiers)
            request = ResourceRequest(
                node="routing_experiment",
                metadata=metadata,
                state={"input": case.input},
            )
            result = self.runtime._run_auto_model(request, case.input, case.system_prompt)
            attempts = [dict(item) for item in result.metadata.get("attempts") or []]
            selected_tier = str(result.metadata.get("selected_tier") or "")
            actual_tier = str(result.metadata.get("actual_tier") or "")
            allocation = result.metadata.get("allocation") or {}
            profile = ((allocation.get("metadata") or {}).get("decision") or {}).get("profile") or {}
            meta = profile.get("metadata") or {}
            gate = str(meta.get("router") or meta.get("gate") or "")
            router_backend = str(meta.get("router_backend") or "")
            router_score = meta.get("score")
        attempt_rows = [
            {"case_id": case.case_id, "arm": arm, "attempt": index + 1, **attempt,
             "cloud_cost_cny": self._attempt_cloud_cost(attempt)}
            for index, attempt in enumerate(attempts)
        ]
        costs = [item["cloud_cost_cny"] for item in attempt_rows]
        usage = {}
        for field in ("prompt_tokens", "completion_tokens"):
            counts = [item.get(field) for item in attempt_rows]
            usage[field] = None if None in counts else sum(counts)
        success = bool(result.success) and _evaluate(case, result.text)
        row = {
            "case_id": case.case_id,
            "arm": arm,
            "dataset": case.dataset,
            "stratum": case.stratum,
            "selected_tier": selected_tier,
            "actual_tier": actual_tier,
            "gate": gate,
            "router_backend": router_backend,
            "router_score": router_score,
            "fallback": bool(result.metadata.get("fallback")),
            "strict_route": arm == "strict_auto",
            **usage,
            "cloud_cost_cny": None if None in costs else sum(costs),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "success": success,
            "raw_success": bool(result.success),
            "failure_type": "" if result.success else str(result.error or "inference_failed"),
            "evaluator": case.evaluator_type,
        }
        return row, attempt_rows

    def run(self, cases: Iterable[RoutingCase], output_dir: str | Path, *, manifest: Dict[str, Any]) -> Dict[str, Any]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "manifest.json", {
            **manifest,
            "auto_arm": self.auto_arm,
            "quality_tiers": self.quality_tiers if self.auto_arm == "quality_auto" else None,
            "accuracy_margin_pp": self.accuracy_margin_pp,
            "auto_status": self.connections.auto_status(),
            "pricing_cny_per_million_tokens": {
                model: {"input": price.input_per_million_cny, "output": price.output_per_million_cny}
                for model, price in self.pricing.items()
            },
        })
        rows: List[Dict[str, Any]] = []
        attempts: List[Dict[str, Any]] = []
        completed = 0
        for case in cases:
            for arm in ("all_cloud", self.auto_arm):
                row, call_rows = self.run_case(case, arm)
                rows.append(row)
                attempts.extend(call_rows)
                completed += 1
                # Persist after every arm so a long run is inspectable and resumable for diagnosis.
                write_jsonl(root / "rows.jsonl", rows)
                write_jsonl(root / "attempts.jsonl", attempts)
                print(json.dumps({"progress": completed, "total": "unknown", "case_id": case.case_id,
                                  "arm": arm, "success": bool(row.get("success"))}, ensure_ascii=False), flush=True)
        write_jsonl(root / "rows.jsonl", rows)
        write_jsonl(root / "attempts.jsonl", attempts)
        summary = summarize_routing_rows(rows, auto_arm=self.auto_arm, accuracy_margin_pp=self.accuracy_margin_pp)
        write_json(root / "summary.json", summary)
        (root / "report.md").write_text(_render_report(summary), encoding="utf-8")
        return summary

    def _attempt_cloud_cost(self, attempt: Dict[str, Any]) -> float | None:
        tier = attempt.get("tier") or (attempt.get("allocation") or {}).get("tier")
        if tier in {"device", "edge"}:
            return 0.0
        price = self.pricing.get(str(attempt.get("model") or ""))
        prompt = attempt.get("prompt_tokens")
        completion = attempt.get("completion_tokens")
        if tier != "cloud" or price is None or prompt is None or completion is None:
            return None
        return round(
            prompt / 1_000_000 * price.input_per_million_cny
            + completion / 1_000_000 * price.output_per_million_cny,
            12,
        )

    @staticmethod
    def _attempt_from_result(result: Any, tier: str, connection_id: str, model: str) -> Dict[str, Any]:
        return {
            "tier": tier, "connection_id": connection_id, "model": model,
            "success": bool(result.success), "error": result.error,
            "prompt_tokens": result.metadata.get("prompt_tokens"),
            "completion_tokens": result.metadata.get("completion_tokens"),
            "total_tokens": result.metadata.get("total_tokens"),
            "latency_ms": result.metadata.get("latency_ms"),
        }


def summarize_routing_rows(
    rows: Iterable[Dict[str, Any]], *, auto_arm: str = "strict_auto", accuracy_margin_pp: float = 5.0,
) -> Dict[str, Any]:
    selected = list(rows)
    result: Dict[str, Any] = {"total_rows": len(selected), "arms": {}}
    for arm in ("all_cloud", auto_arm):
        arm_rows = [row for row in selected if row.get("arm") == arm]
        successes = sum(bool(row.get("success")) for row in arm_rows)
        costs = [row.get("cloud_cost_cny") for row in arm_rows]
        unknown_cost = costs.count(None)
        total_cost = sum(costs) if arm_rows and not unknown_cost else None
        result["arms"][arm] = {
            "n": len(arm_rows),
            "task_success_rate": successes / len(arm_rows) if arm_rows else 0.0,
            "cloud_cost_cny": total_cost,
            "unknown_cloud_cost_rows": unknown_cost,
            "cloud_cost_per_success_cny": total_cost / successes if successes and total_cost is not None else None,
            "p50_latency_ms": statistics.median([float(row.get("latency_ms") or 0) for row in arm_rows]) if arm_rows else None,
        }
    cloud = result["arms"]["all_cloud"]
    auto = result["arms"][auto_arm]
    result["auto_arm"] = auto_arm
    result["success_rate_delta_pp"] = round((auto["task_success_rate"] - cloud["task_success_rate"]) * 100, 6)
    result["cloud_cost_reduction"] = (
        1 - auto["cloud_cost_cny"] / cloud["cloud_cost_cny"]
        if cloud["cloud_cost_cny"] and auto["cloud_cost_cny"] is not None else None
    )
    strict_rows = [row for row in selected if row.get("arm") == "strict_auto"]
    result["strict_route_match_rate"] = (
        sum(bool(row.get("selected_tier")) and row.get("selected_tier") == row.get("actual_tier") for row in strict_rows)
        / len(strict_rows) if strict_rows else None
    )
    intervals = _paired_bootstrap(selected, auto_arm=auto_arm)
    result["paired_bootstrap_95ci"] = intervals
    delta_ci = intervals["success_rate_delta_pp"]
    cloud_cases = {row["case_id"] for row in selected if row.get("case_id") and row.get("arm") == "all_cloud"}
    auto_cases = {row["case_id"] for row in selected if row.get("case_id") and row.get("arm") == auto_arm}
    complete_pairs = cloud_cases == auto_cases and len(cloud_cases) == cloud["n"] == auto["n"]
    result["accuracy_check"] = {
        "margin_pp": accuracy_margin_pp,
        "paired_cases": len(cloud_cases & auto_cases),
        "passes": delta_ci[0] >= -accuracy_margin_pp if delta_ci and complete_pairs else None,
    }
    return result


def _paired_bootstrap(
    rows: List[Dict[str, Any]], samples: int = 1_000, seed: int = 42, *, auto_arm: str = "strict_auto",
) -> Dict[str, List[float] | None]:
    by_case: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row.get("case_id")), {})[str(row.get("arm"))] = row
    pairs = [item for item in by_case.values() if {"all_cloud", auto_arm} <= set(item)]
    if not pairs:
        return {"success_rate_delta_pp": None, "cloud_cost_delta_cny": None}
    rng = random.Random(seed)
    success_deltas: List[float] = []
    cost_deltas: List[float] = []
    costs_known = all(pair[arm].get("cloud_cost_cny") is not None for pair in pairs for arm in ("all_cloud", auto_arm))
    for _ in range(samples):
        draw = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        success_deltas.append(100 * sum(
            float(pair[auto_arm].get("success", False)) - float(pair["all_cloud"].get("success", False))
            for pair in draw
        ) / len(draw))
        if costs_known:
            cost_deltas.append(sum(
                pair[auto_arm]["cloud_cost_cny"] - pair["all_cloud"]["cloud_cost_cny"] for pair in draw
            ) / len(draw))
    def interval(values: List[float]) -> List[float]:
        values.sort()
        return [round(values[int(.025 * (len(values) - 1))], 8), round(values[int(.975 * (len(values) - 1))], 8)]
    return {"success_rate_delta_pp": interval(success_deltas), "cloud_cost_delta_cny": interval(cost_deltas) if costs_known else None}


def load_routing_cases(path: str | Path) -> List[RoutingCase]:
    cases: List[RoutingCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(RoutingCase(
            case_id=str(row["case_id"]), input=str(row["input"]),
            evaluator_type=str(row.get("evaluator_type") or "exact_match"),
            reference=row.get("reference", ""), stratum=str(row.get("stratum") or "unknown"),
            dataset=str(row.get("dataset") or "custom"), system_prompt=str(row.get("system_prompt") or ""),
            metadata=dict(row.get("metadata") or {}),
        ))
    return cases


def _evaluate(case: RoutingCase, output: str) -> bool:
    if case.evaluator_type == "exact_match":
        return output.strip().upper() == str(case.reference).strip().upper()
    if case.evaluator_type == "contains":
        return str(case.reference).strip().lower() in output.lower()
    if case.evaluator_type == "json_schema":
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return False
        return all(key in data for key in (case.reference or []))
    raise ValueError(f"unsupported evaluator_type: {case.evaluator_type}")


def _render_report(summary: Dict[str, Any]) -> str:
    cloud = summary["arms"]["all_cloud"]
    auto_arm = summary["auto_arm"]
    auto = summary["arms"][auto_arm]
    accuracy = summary["accuracy_check"]
    def format_cost(value: float | None) -> str:
        return f"{value:.8f}" if value is not None else "unknown"

    return "\n".join([
        f"# {auto_arm} vs All-Cloud", "",
        "| Arm | N | Task success rate | Cloud cost (CNY) | Cost per success (CNY) |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| All-Cloud | {cloud['n']} | {cloud['task_success_rate']:.4f} | {format_cost(cloud['cloud_cost_cny'])} | {format_cost(cloud['cloud_cost_per_success_cny'])} |",
        f"| {auto_arm} | {auto['n']} | {auto['task_success_rate']:.4f} | {format_cost(auto['cloud_cost_cny'])} | {format_cost(auto['cloud_cost_per_success_cny'])} |",
        "",
        f"- Success-rate delta: {summary['success_rate_delta_pp']:.4f} pp",
        f"- Cloud-cost reduction: {summary['cloud_cost_reduction']}",
        f"- Accuracy check (lower 95% CI bound ≥ −{accuracy['margin_pp']} pp): {accuracy['passes']}",
        f"- Strict route match rate: {summary['strict_route_match_rate']}",
    ]) + "\n"
