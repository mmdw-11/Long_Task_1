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
        return {
            "text": result.text,
            "executor": result.executor,
            "model": result.model,
            "endpoint": result.endpoint,
            "success": bool(result.success),
            "error": result.error,
            "latency_ms": round((time.time() - start) * 1000, 3),
            "metadata": _compact_metadata(result.metadata),
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
        "instruction": (
            "Score each answer from 0 to 1 for correctness, completeness, and instruction following. "
            "Choose the cheapest sufficient tier: device first, then edge, then cloud. "
            "Return strict JSON with keys scores, selected_tier, reason. "
            "scores must map device/edge/cloud to numbers."
        ),
        "answers": {tier: tier_results[tier]["text"] for tier in TIERS},
    }
    response = client.chat.completions.create(
        model=settings.model,
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
