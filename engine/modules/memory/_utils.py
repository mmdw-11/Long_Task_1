"""记忆模块内部工具函数。"""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any, List

from ._types import MemoryItem


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def _normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _cosine(left: List[float], right: List[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    return sum(left[i] * right[i] for i in range(size))


def _sparse_score(query_tokens: List[str], item_tokens: List[str]) -> float:
    if not query_tokens or not item_tokens:
        return 0.0
    item_set = set(item_tokens)
    hits = sum(1 for token in query_tokens if token in item_set)
    return hits / max(1, len(set(query_tokens)))


def _recency_score(ts: float) -> float:
    age_seconds = max(0.0, time.time() - ts)
    return 1.0 / (1.0 + age_seconds / 86400.0)


def _memory_index_text(item: MemoryItem) -> str:
    return " ".join(
        [
            item.summary,
            _stringify(item.content),
            " ".join(item.tags),
            _stringify(item.metadata),
        ]
    )


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _summarize_text(text: str, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _rough_token_count(text: str) -> int:
    return max(1, len(_tokenize(text)))


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "default"


def _clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _indent_block(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _format_markdown_memory(item: MemoryItem, raw_text: str) -> str:
    metadata = {
        "id": item.id,
        "scope": item.scope.value,
        "scope_id": item.scope_id,
        "tags": item.tags,
        "ts": item.ts,
        **item.metadata,
    }
    return (
        "---\n"
        f"{json.dumps(metadata, ensure_ascii=False, indent=2)}\n"
        "---\n\n"
        f"# Memory {item.id}\n\n"
        "## Summary\n\n"
        f"{item.summary}\n\n"
        "## Raw\n\n"
        f"{raw_text}\n"
    )
