"""Merge original LMSYS prompts with curated Honor mobile prompts.

The source LMSYS file is never changed.  The merged file is an *unlabelled
three-tier evaluation input*: labels are produced later by
``label_lmsys_tiers.py``.  Honor's manually designed binary label is preserved
as audit metadata rather than mixed into the final three-tier ``label`` field.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmsys-input", default="runs/router_learning/lmsys_prompts.jsonl")
    parser.add_argument("--honor-input", default="runs/router_learning/honor_phone_router_dataset.jsonl")
    parser.add_argument("--extension-input", default="runs/router_learning/router_extension_14k.jsonl")
    parser.add_argument("--output", default="runs/router_learning/lmsys_plus_honor_prompts.jsonl")
    parser.add_argument("--dedupe", action="store_true")
    args = parser.parse_args()

    lmsys_path = Path(args.lmsys_input)
    honor_path = Path(args.honor_input)
    extension_path = Path(args.extension_input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    source_counts: Counter[str] = Counter()
    written = 0
    honor_seed_counts: Counter[int] = Counter()
    with output.open("w", encoding="utf-8") as fh:
        for row in _iter_lmsys(lmsys_path):
            if _write_if_new(fh, row, seen, args.dedupe):
                written += 1
                source_counts["lmsys"] += 1
        for row in _iter_honor(honor_path):
            if _write_if_new(fh, row, seen, args.dedupe):
                written += 1
                source_counts["honor"] += 1
                honor_seed_counts[int(row["metadata"]["curated_binary_label"])] += 1
        for row in _iter_extension(extension_path):
            if _write_if_new(fh, row, seen, args.dedupe):
                written += 1
                source_counts["extension"] += 1
                honor_seed_counts[int(row["metadata"]["curated_binary_label"])] += 1

    print(
        json.dumps(
            {
                "output": str(output),
                "rows_written": written,
                "source_counts": dict(source_counts),
                "honor_curated_binary_label_counts": dict(honor_seed_counts),
                "note": "The original LMSYS input was read-only; run label_lmsys_tiers.py on this merged file for final tier labels.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _iter_lmsys(path: Path) -> Iterator[Dict[str, Any]]:
    for row in _read_jsonl(path):
        metadata = dict(row.get("metadata") or {})
        metadata["merged_source"] = "lmsys_original"
        yield {"id": str(row["id"]), "text": str(row["text"]), "metadata": metadata}


def _iter_honor(path: Path) -> Iterator[Dict[str, Any]]:
    for row in _read_jsonl(path):
        metadata = dict(row.get("metadata") or {})
        seed_label = int(row["label"])
        metadata.update(
            {
                "merged_source": "honor_curated",
                "curated_binary_label": seed_label,
                "curated_binary_label_schema": "0=device_local_qwen,1=cloud_deepseek",
            }
        )
        yield {"id": str(row["id"]), "text": str(row["text"]), "metadata": metadata}


def _iter_extension(path: Path) -> Iterator[Dict[str, Any]]:
    for row in _read_jsonl(path):
        metadata = dict(row.get("metadata") or {})
        metadata["merged_source"] = "router_extension"
        yield {"id": str(row["id"]), "text": str(row["text"]), "metadata": metadata}


def _write_if_new(fh: Any, row: Dict[str, Any], seen: set[str], dedupe: bool) -> bool:
    normalized = row["text"].strip()
    if not normalized or (dedupe and normalized in seen):
        return False
    seen.add(normalized)
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    main()
