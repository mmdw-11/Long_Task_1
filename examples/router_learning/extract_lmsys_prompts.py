"""Extract user prompts from LMSYS Chatbot Arena conversations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def main() -> None:
    _project_root = str(Path(__file__).resolve().parent.parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="lmsys/chatbot_arena_conversations")
    parser.add_argument("--output", default="runs/router_learning/lmsys_prompts.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--language", default="", help="Optional exact language filter, e.g. English")
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset_name)["train"]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out.open("w", encoding="utf-8") as fh:
        for row in ds:
            if args.language and str(row.get("language", "")) != args.language:
                continue
            text = _first_user_text(row.get("conversation_a") or [])
            if not text:
                continue
            payload = {
                "id": str(row.get("question_id") or written),
                "text": text,
                "metadata": {
                    "source": args.dataset_name,
                    "language": row.get("language"),
                    "turn": row.get("turn"),
                    "winner": row.get("winner"),
                    "model_a": row.get("model_a"),
                    "model_b": row.get("model_b"),
                },
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
            if args.limit and written >= args.limit:
                break

    print(json.dumps({"output": str(out), "rows": written}, ensure_ascii=False, indent=2))


def _first_user_text(conversation: Iterable[Dict[str, Any]]) -> str:
    for message in conversation:
        if message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""


if __name__ == "__main__":
    main()
