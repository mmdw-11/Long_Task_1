"""Reduce the pending router dataset while preserving completed work.

The full merged input is archived before replacement.  The resulting input
contains every already-labelled unique task, plus a deterministic sample of
300 remaining LMSYS prompts and 700 remaining 14k extension prompts.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/router_learning/lmsys_plus_honor_prompts.jsonl")
    parser.add_argument("--completed", default="runs/router_learning/final_tier_labeled_dataset.jsonl")
    parser.add_argument("--archive", default="runs/router_learning/lmsys_plus_honor_prompts.full_40789.jsonl")
    parser.add_argument("--output", default="runs/router_learning/lmsys_plus_honor_prompts.jsonl")
    parser.add_argument("--remaining-lmsys", type=int, default=300)
    parser.add_argument("--remaining-extension", type=int, default=700)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    input_path = Path(args.input)
    completed_path = Path(args.completed)
    archive_path = Path(args.archive)
    output_path = Path(args.output)
    all_rows = list(_read_jsonl(input_path))
    completed_ids = _completed_ids(completed_path)
    rows_by_id = {str(row["id"]): row for row in all_rows}
    completed_rows = [rows_by_id[item_id] for item_id in completed_ids if item_id in rows_by_id]

    remaining = [row for row in all_rows if str(row["id"]) not in completed_ids]
    lmsys = [row for row in remaining if row.get("metadata", {}).get("merged_source") == "lmsys_original"]
    extension = [row for row in remaining if row.get("metadata", {}).get("merged_source") == "router_extension"]
    rng = random.Random(args.seed)
    rng.shuffle(lmsys)
    rng.shuffle(extension)
    selected = completed_rows + lmsys[: args.remaining_lmsys] + extension[: args.remaining_extension]
    rng.shuffle(selected)

    if input_path.resolve() != output_path.resolve():
        shutil.copy2(input_path, archive_path)
    elif not archive_path.exists():
        shutil.copy2(input_path, archive_path)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(output_path),
                "archive": str(archive_path),
                "completed_unique_preserved": len(completed_rows),
                "new_lmsys": min(args.remaining_lmsys, len(lmsys)),
                "new_extension": min(args.remaining_extension, len(extension)),
                "total_unique_tasks": len(selected),
                "sources": dict(Counter(row.get("metadata", {}).get("merged_source") for row in selected)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _completed_ids(path: Path) -> List[str]:
    # Preserve the first completed occurrence while eliminating prior duplicate runs.
    result: List[str] = []
    seen: set[str] = set()
    for row in _read_jsonl(path):
        item_id = str((row.get("metadata") or {}).get("id") or "")
        if item_id and item_id not in seen:
            seen.add(item_id)
            result.append(item_id)
    return result


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    main()
