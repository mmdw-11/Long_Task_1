"""Extract user prompts from local LMSYS chatbot arena JSONL export.

Default input:
    .hf_cache/exports/lmsys_chatbot_arena_conversations.jsonl

Default output:
    runs/router_learning/lmsys_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=".hf_cache/exports/lmsys_chatbot_arena_conversations.jsonl")
    parser.add_argument("--output", default="runs/router_learning/lmsys_prompts.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--conversation", choices=["conversation_a", "conversation_b"], default="conversation_a")
    parser.add_argument("--all-user-turns", action="store_true")
    parser.add_argument("--dedupe", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    written = 0
    with output_path.open("w", encoding="utf-8") as out_fh:
        for row_idx, row in enumerate(_read_jsonl(input_path)):
            for turn_idx, prompt in _iter_user_prompts(row, args.conversation, args.all_user_turns):
                prompt = prompt.strip()
                if not prompt:
                    continue
                if args.dedupe and prompt in seen:
                    continue
                seen.add(prompt)
                payload = {
                    "id": str(row.get("question_id") or f"row-{row_idx}") + f":{args.conversation}:{turn_idx}",
                    "text": prompt,
                    "metadata": {
                        "source": "lmsys/chatbot_arena_conversations",
                        "source_file": str(input_path),
                        "conversation": args.conversation,
                        "turn_index": turn_idx,
                        "language": row.get("language"),
                        "turn": row.get("turn"),
                        "winner": row.get("winner"),
                        "model_a": row.get("model_a"),
                        "model_b": row.get("model_b"),
                    },
                }
                out_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                written += 1
                if args.limit and written >= args.limit:
                    print(json.dumps({"output": str(output_path), "rows_written": written}, ensure_ascii=False, indent=2))
                    return

    print(json.dumps({"output": str(output_path), "rows_written": written}, ensure_ascii=False, indent=2))


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_user_prompts(row: Dict[str, Any], conversation_key: str, all_user_turns: bool) -> Iterable[tuple[int, str]]:
    conversation = row.get(conversation_key) or []
    if not isinstance(conversation, list):
        return
    for idx, message in enumerate(conversation):
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").lower() != "user":
            continue
        content = _message_content(message)
        if content:
            yield idx, content
        if not all_user_turns:
            return


def _message_content(message: Dict[str, Any]) -> str:
    content: Optional[Any] = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return "" if content is None else str(content)


if __name__ == "__main__":
    main()
