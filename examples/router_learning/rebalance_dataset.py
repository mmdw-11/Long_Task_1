"""Rebalance the routing dataset for a second-round experiment."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from engine import RouteDataset, RouteExample
try:
    from examples.router_learning.prompt_library import PROMPT_LIBRARY
except ModuleNotFoundError:
    from prompt_library import PROMPT_LIBRARY


HARD_CATEGORIES = {"code", "long_doc", "privacy", "multimodal"}
EASY_CATEGORIES = {"calendar"}
HARD_BROWSER_KEYWORDS = {"导出", "对比", "抓取", "总结", "文档", "GitHub", "高铁"}
EASY_BROWSER_KEYWORDS = {"天气", "地图", "路线", "搜索"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/router_learning/expanded_dataset.jsonl")
    parser.add_argument("--output", default="runs/router_learning/balanced_dataset.jsonl")
    parser.add_argument("--target-per-class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base = RouteDataset.load_jsonl(args.input)
    rebalanced = build_balanced_dataset(base, target_per_class=args.target_per_class, rng=rng)
    out = rebalanced.save_jsonl(args.output)

    counts = Counter(int(item.label) for item in rebalanced.examples)
    print(
        json.dumps(
            {
                "output": str(out),
                "examples": len(rebalanced.examples),
                "counts": dict(counts),
                "target_per_class": args.target_per_class,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_balanced_dataset(
    base: RouteDataset,
    *,
    target_per_class: int,
    rng: random.Random,
) -> RouteDataset:
    positives: List[RouteExample] = []
    negatives: List[RouteExample] = []

    for item in base.examples:
        label, reason = relabel_example(item.text, item.metadata)
        example = RouteExample(
            text=item.text,
            label=label,
            metadata={**item.metadata, "rebalance_reason": reason, "rebalance_source": "base"},
        )
        if label == 1:
            positives.append(example)
        else:
            negatives.append(example)

    synthetic = synthesize_from_library()
    for example in synthetic:
        if example.label == 1:
            positives.append(example)
        else:
            negatives.append(example)

    positives = _dedupe_by_text(positives)
    negatives = _dedupe_by_text(negatives)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    selected_pos = positives[:target_per_class]
    selected_neg = negatives[:target_per_class]

    if len(selected_pos) < target_per_class:
        selected_pos.extend(_repeat_fill(selected_pos, target_per_class, rng))
    if len(selected_neg) < target_per_class:
        selected_neg.extend(_repeat_fill(selected_neg, target_per_class, rng))

    final_items = selected_pos[:target_per_class] + selected_neg[:target_per_class]
    rng.shuffle(final_items)
    return RouteDataset(final_items)


def relabel_example(text: str, metadata: Dict[str, Any]) -> Tuple[int, str]:
    normalized = text.lower()
    if any(keyword in normalized for keyword in ["架构", "重构", "复杂", "合同", "长文", "隐私", "图片", "截图", "流程图"]):
        return 1, "keyword_hard_override"
    if any(keyword in normalized for keyword in ["日程", "提醒", "会议", "总结邮件", "待办"]):
        return 0, "keyword_easy_override"
    original = int(metadata.get("original_label", metadata.get("label", 0)) or 0)
    return original, "keep_original"


def synthesize_from_library() -> List[RouteExample]:
    items: List[RouteExample] = []
    for category, prompts in PROMPT_LIBRARY.items():
        for prompt in prompts:
            label, reason = infer_library_label(category, prompt)
            items.append(
                RouteExample(
                    text=prompt,
                    label=label,
                    metadata={
                        "rebalance_source": "prompt_library",
                        "category": category,
                        "rebalance_reason": reason,
                    },
                )
            )
    return items


def infer_library_label(category: str, prompt: str) -> Tuple[int, str]:
    if category in HARD_CATEGORIES:
        return 1, f"hard_category:{category}"
    if category in EASY_CATEGORIES:
        return 0, f"easy_category:{category}"
    if category == "browser":
        if any(word in prompt for word in HARD_BROWSER_KEYWORDS):
            return 1, "browser_hard_keyword"
        if any(word in prompt for word in EASY_BROWSER_KEYWORDS):
            return 0, "browser_easy_keyword"
    return 0, "default_easy"


def _dedupe_by_text(items: Iterable[RouteExample]) -> List[RouteExample]:
    seen: Dict[str, RouteExample] = {}
    for item in items:
        seen[item.text] = item
    return list(seen.values())


def _repeat_fill(items: List[RouteExample], target_size: int, rng: random.Random) -> List[RouteExample]:
    if not items:
        return []
    result: List[RouteExample] = []
    while len(items) + len(result) < target_size:
        source = rng.choice(items)
        result.append(
            RouteExample(
                text=source.text,
                label=source.label,
                metadata={**source.metadata, "rebalance_source": f"{source.metadata.get('rebalance_source', 'unknown')}_repeat"},
            )
        )
    return result


if __name__ == "__main__":
    main()
