"""Sample cascade-routing labels into a JSONL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from engine import (
    FallbackCascadeTeacher,
    PseudoCascadeTeacher,
    RealCascadeTeacher,
    build_route_dataset,
    default_training_texts,
)

try:
    from examples.router_learning.prompt_library import PROMPT_LIBRARY, all_prompts
except ModuleNotFoundError:
    from prompt_library import PROMPT_LIBRARY, all_prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="", help="Optional JSONL/text file with prompts.")
    parser.add_argument("--output", default="runs/router_learning/cascade_dataset.jsonl")
    parser.add_argument("--teacher", choices=["real", "fallback", "pseudo"], default="fallback")
    parser.add_argument("--source", choices=["default", "library"], default="library")
    parser.add_argument("--category", default="", help="Optional prompt-library category.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--no-large", action="store_true", help="Do not call the large model after promotion.")
    args = parser.parse_args()

    if args.input:
        texts = _load_texts(args.input)
    elif args.source == "library":
        texts = PROMPT_LIBRARY.get(args.category, []) if args.category else all_prompts()
    else:
        texts = default_training_texts()
    if args.limit > 0:
        texts = texts[: args.limit]
    teacher = _make_teacher(
        args.teacher,
        run_large_on_promote=not args.no_large,
        timeout_seconds=args.timeout,
    )
    dataset = build_route_dataset(texts, teacher=teacher)
    out = dataset.save_jsonl(args.output)

    summary = {
        "output": str(out),
        "examples": len(dataset.examples),
        "large_labels": sum(1 for item in dataset.examples if item.label == 1),
        "small_labels": sum(1 for item in dataset.examples if item.label == 0),
        "teacher": args.teacher,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _make_teacher(mode: str, *, run_large_on_promote: bool, timeout_seconds: float):
    if mode == "real":
        return RealCascadeTeacher(
            run_large_on_promote=run_large_on_promote,
            timeout_seconds=timeout_seconds,
        )
    if mode == "pseudo":
        return PseudoCascadeTeacher()
    return FallbackCascadeTeacher(
        primary=RealCascadeTeacher(
            run_large_on_promote=run_large_on_promote,
            timeout_seconds=timeout_seconds,
        )
    )


def _load_texts(path: str) -> List[str]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        return list(_load_jsonl_texts(source))
    return [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_jsonl_texts(path: Path) -> Iterable[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        text = data.get("text") or data.get("input") or data.get("prompt")
        if text:
            yield str(text)


if __name__ == "__main__":
    main()
