"""Run LMSYS prompts on device/edge/cloud and emit BGE-M3 router labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from engine import (
    EdgeHttpExecutor,
    InferenceRequest,
    LocalModelExecutor,
    OpenAICompatibleCloudExecutor,
)


TIERS = ("device", "edge", "cloud")
TIER_LABELS = {"device": 0, "edge": 1, "cloud": 2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/router_learning/lmsys_prompts.jsonl")
    parser.add_argument("--output", default="runs/router_learning/lmsys_tier_labeled_dataset.jsonl")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--quality-threshold", type=float, default=0.72)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    _load_env_file()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = _existing_ids(output) if args.skip_existing else set()

    device = LocalModelExecutor()
    edge = EdgeHttpExecutor()
    cloud = OpenAICompatibleCloudExecutor()

    written = 0
    with output.open("a", encoding="utf-8") as out_fh:
        for idx, item in enumerate(_read_jsonl(args.input)):
            if idx < args.start:
                continue
            sample_id = str(item.get("id") or idx)
            if sample_id in existing_ids:
                continue
            if args.limit and written >= args.limit:
                break

            text = str(item["text"]).strip()
            if not text:
                continue

            tier_results = {
                "device": _run_executor(device, "device", text),
                "edge": _run_executor(edge, "edge", text),
                "cloud": _run_executor(cloud, "cloud", text),
            }
            judge = _judge_tiers(text, tier_results, args.quality_threshold)
            selected_tier = _select_tier(judge, args.quality_threshold)
            label = TIER_LABELS[selected_tier]
            payload = {
                "text": text,
                "label": label,
                "metadata": {
                    **dict(item.get("metadata") or {}),
                    "id": sample_id,
                    "selected_tier": selected_tier,
                    "route_label_schema": "0=device,1=edge,2=cloud",
                    "tier_results": tier_results,
                    "judge": judge,
                },
            }
            out_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            out_fh.flush()
            written += 1
            print(json.dumps({"id": sample_id, "selected_tier": selected_tier, "label": label}, ensure_ascii=False))
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)

    print(json.dumps({"output": str(output), "rows_written": written}, ensure_ascii=False, indent=2))


def _run_executor(executor: Any, tier: str, prompt: str) -> Dict[str, Any]:
    start = time.time()
    try:
        result = executor.run(
            InferenceRequest(
                prompt=prompt,
                allocation={"tier": tier, "endpoint": f"{tier}://default"},
                metadata={"tier": tier},
            )
        )
        metadata = _compact_metadata(result.metadata)
        simulated = bool(metadata.get("simulated") or metadata.get("edge_fallback"))
        error = result.error
        if simulated and not error:
            error = "backend_fallback_or_simulation"
        return {
            "text": result.text,
            "executor": result.executor,
            "model": result.model,
            "endpoint": result.endpoint,
            # A fallback is useful for interactive demos, but is not a valid
            # measurement when constructing real device/edge/cloud labels.
            "success": bool(result.success) and not simulated,
            "error": error,
            "latency_ms": round((time.time() - start) * 1000, 3),
            "metadata": metadata,
        }
    except Exception as exc:  # noqa: BLE001 - labeling should keep record of failures
        return {
            "text": "",
            "executor": type(executor).__name__,
            "model": "",
            "endpoint": f"{tier}://default",
            "success": False,
            "error": str(exc),
            "latency_ms": round((time.time() - start) * 1000, 3),
            "metadata": {},
        }


def _judge_tiers(prompt: str, tier_results: Dict[str, Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    from openai import OpenAI

    from engine.config import load_settings

    settings = load_settings()
    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        organization=settings.organization,
        timeout=120,
    )
    judge_prompt = {
        "task": prompt,
        "quality_threshold": threshold,
        "contest_requirement": (
            "The task is adaptive resource scheduling in a device-edge-cloud heterogeneous environment: "
            "the system must adapt to heterogeneous compute resources, dynamically select the inference location "
            "and split models according to a task's real-time requirement and data-sensitivity level, while fully "
            "using cloud compute for high-complexity tasks."
        ),
        "routing_requirement": (
            "This is an end-edge-cloud heterogeneous resource scheduling task. "
            "Automatically select the inference location and model tier using: "
            "(1) task real-time requirement: hard/interactive latency favors device, then edge; "
            "(2) data sensitivity: private or confidential data must remain on device/edge unless the task explicitly permits cloud transfer; "
            "(3) task complexity and required reasoning quality: cloud is allowed and preferred only when device/edge cannot meet the quality threshold; "
            "(4) actual backend availability and measured latency; and "
            "(5) resource efficiency: among compliant tiers that meet quality, choose device before edge before cloud. "
            "Do not select an unavailable tier, a failed response, or a fallback/simulated response. "
            "Do not infer sensitivity merely because a task is difficult; only treat explicit personal, secret, financial, medical, credential, or proprietary content as sensitive."
        ),
        "evaluation_instruction": (
            "For every available answer, assign a quality score from 0 to 1 based on correctness, completeness, "
            "faithfulness to the prompt, safety, and instruction following. An answer with an error, empty text, "
            "or failed backend must have score 0. Then choose the cheapest compliant tier that reaches quality_threshold. "
            "Return strict JSON only: {\"scores\": {\"device\": number, \"edge\": number, \"cloud\": number}, "
            "\"selected_tier\": \"device|edge|cloud\", \"routing_assessment\": {\"realtime\": \"hard|interactive|normal|batch\", "
            "\"data_sensitivity\": \"public|internal|sensitive\", \"complexity\": \"simple|moderate|high|long_horizon\", "
            "\"cloud_transfer_allowed\": true}, \"reason\": \"brief evidence-based explanation\"}. "
            "A long-horizon task has at least three dependent sub-tasks, conditional branches, iterative refinement, "
            "or persistent cross-application execution."
        ),
        "candidates": {
            tier: {
                "answer": tier_results[tier]["text"],
                "backend_success": tier_results[tier]["success"],
                "error": tier_results[tier]["error"],
                "latency_ms": tier_results[tier]["latency_ms"],
                "model": tier_results[tier]["model"],
            }
            for tier in TIERS
        },
    }
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_JUDGE_MODEL") or settings.model,
        messages=[
            {"role": "system", "content": "You are a strict model-routing evaluator. Return JSON only."},
            {"role": "user", "content": json.dumps(judge_prompt, ensure_ascii=False)},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content or "{}"
    data = _parse_json_object(raw)
    scores = data.get("scores") if isinstance(data, dict) else {}
    clean_scores = {tier: _coerce_score((scores or {}).get(tier)) for tier in TIERS}
    selected = str((data or {}).get("selected_tier") or "").strip().lower()
    return {
        "scores": clean_scores,
        "selected_tier": selected if selected in TIERS else "",
        "routing_assessment": dict((data or {}).get("routing_assessment") or {}),
        "reason": str((data or {}).get("reason") or ""),
        "raw": raw,
    }


def _select_tier(judge: Dict[str, Any], threshold: float) -> str:
    selected = str(judge.get("selected_tier") or "").lower()
    if selected in TIERS:
        return selected
    scores = dict(judge.get("scores") or {})
    for tier in TIERS:
        if float(scores.get(tier, 0.0)) >= threshold:
            return tier
    return "cloud"


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
    try:
        data = json.loads(stripped or "{}")
    except json.JSONDecodeError:
        return {"raw_parse_error": text}
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for item in _read_jsonl(path):
        metadata = dict(item.get("metadata") or {})
        if metadata.get("id"):
            ids.add(str(metadata["id"]))
    return ids


def _compact_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in metadata.items() if key != "usage"}


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _load_env_file() -> None:
    current = Path.cwd()
    for path in [current, *current.parents]:
        env_path = path / ".env"
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return


if __name__ == "__main__":
    main()
