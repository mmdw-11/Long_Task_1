"""Create the frozen 100-case RouteCostBench from C-Eval validation CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ceval-val-dir", help="Optional local C-Eval ceval-exam/val directory")
    parser.add_argument("--output", default="data/processed/RouteCostBench-100.jsonl")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    buckets = _load_local_buckets(Path(args.ceval_val_dir), args.seed) if args.ceval_val_dir else _load_api_buckets(args.seed, args.count)
    selected = []
    if args.count <= 0:
        raise ValueError("--count must be positive")
    while len(selected) < args.count and any(rows for _, rows in buckets):
        for subject, rows in buckets:
            if rows and len(selected) < args.count:
                selected.append(_to_case(subject, rows.pop()))
    if len(selected) != args.count:
        raise ValueError(f"C-Eval validation source has only {len(selected)} usable rows")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "count": args.count, "seed": args.seed,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, ensure_ascii=False))


def _to_case(subject: str, row: dict[str, str]) -> dict[str, object]:
    answer = str(row.get("answer") or "").strip().upper()
    question = str(row.get("question") or "").strip()
    if not question or answer not in {"A", "B", "C", "D"}:
        raise ValueError(f"invalid C-Eval row in {subject}: {row.get('id')}")
    options = "\n".join(f"{key}. {str(row.get(key) or '').strip()}" for key in "ABCD")
    return {
        "case_id": f"ceval-{subject}-{row.get('id')}", "dataset": "ceval-validation",
        "stratum": subject,
        "input": f"请回答下列选择题。只能输出 A、B、C 或 D 中的一个字母，不要解释。\n\n{question}\n{options}",
        "system_prompt": "严格遵守输出格式。", "evaluator_type": "exact_match",
        "reference": answer, "metadata": {"ceval_subject": subject, "ceval_id": row.get("id")},
    }


def _load_local_buckets(source: Path, seed: int) -> list[tuple[str, list[dict[str, str]]]]:
    buckets = []
    for path in sorted(source.glob("*_val.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        random.Random(f"{seed}:{path.name}").shuffle(rows)
        buckets.append((path.stem.removesuffix("_val"), rows))
    if not buckets:
        raise FileNotFoundError(f"no C-Eval validation CSV files found in {source}")
    return buckets


def _load_api_buckets(seed: int, count: int) -> list[tuple[str, list[dict[str, str]]]]:
    dataset = "ceval/ceval-exam"
    splits_url = "https://datasets-server.huggingface.co/splits?dataset=" + urllib.parse.quote(dataset, safe="")
    try:
        with urllib.request.urlopen(splits_url, timeout=30) as response:
            splits = json.loads(response.read().decode("utf-8")).get("splits") or []
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return _load_hf_library(seed, count)
    subjects = sorted({str(item["config"]) for item in splits if item.get("split") == "val"})
    random.Random(seed).shuffle(subjects)
    subjects = subjects[:min(len(subjects), count)]
    buckets = []
    for subject in subjects:
        query = urllib.parse.urlencode({"dataset": dataset, "config": subject, "split": "val", "offset": 0, "length": 100})
        try:
            with urllib.request.urlopen("https://datasets-server.huggingface.co/rows?" + query, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            return _load_hf_library(seed, count)
        rows = [dict(item.get("row") or {}) for item in payload.get("rows") or []]
        random.Random(f"{seed}:{subject}").shuffle(rows)
        if rows:
            buckets.append((subject, rows))
    if not buckets:
        raise RuntimeError("C-Eval dataset server returned no validation rows")
    return buckets


def _load_hf_library(seed: int, count: int) -> list[tuple[str, list[dict[str, str]]]]:
    """Fallback to the official `datasets` loader when datasets-server rate-limits."""
    from datasets import get_dataset_config_names, load_dataset
    subjects = get_dataset_config_names("ceval/ceval-exam")
    random.Random(seed).shuffle(subjects)
    buckets = []
    available = 0
    for subject in subjects[:min(len(subjects), count)]:
        rows = [dict(row) for row in load_dataset("ceval/ceval-exam", subject, split="val")]
        random.Random(f"{seed}:{subject}").shuffle(rows)
        if rows:
            buckets.append((subject, rows))
            available += len(rows)
            if available >= count:
                break
    return buckets


if __name__ == "__main__":
    main()
