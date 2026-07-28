"""Create the canonical deduplicated three-tier router training dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/router_learning/final_tier_labeled_dataset.jsonl")
    parser.add_argument("--prompt-input", default="runs/router_learning/lmsys_plus_honor_prompts.jsonl")
    parser.add_argument("--output", default="runs/router_learning/final_tier_labeled_dataset.deduped.jsonl")
    parser.add_argument("--report", default="runs/router_learning/final_tier_labeled_dataset.deduped_report.json")
    args = parser.parse_args()

    allowed_ids = {str(row["id"]) for row in _read_jsonl(Path(args.prompt_input))}
    latest: Dict[str, Dict[str, Any]] = {}
    labels_by_id: dict[str, set[int]] = defaultdict(set)
    total_rows = 0
    excluded = 0
    for row in _read_jsonl(Path(args.input)):
        total_rows += 1
        item_id = str((row.get("metadata") or {}).get("id") or "")
        if not item_id or item_id not in allowed_ids:
            excluded += 1
            continue
        latest[item_id] = row  # JSONL append order: retain the latest complete run.
        labels_by_id[item_id].add(int(row["label"]))

    rows = list(latest.values())
    rows.sort(key=lambda row: str((row.get("metadata") or {}).get("id")))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {
        "input_rows": total_rows,
        "excluded_not_in_reduced_prompt_set": excluded,
        "unique_completed_tasks": len(rows),
        "pending_tasks": len(allowed_ids - set(latest)),
        "duplicate_label_conflicts": sum(len(values) > 1 for values in labels_by_id.values()),
        "latest_label_counts": dict(Counter(int(row["label"]) for row in rows)),
        "route_label_schema": "0=device,1=edge,2=cloud",
        "dedupe_policy": "latest complete JSONL row per task id",
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(args.report), **report}, ensure_ascii=False, indent=2))


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    main()
