"""Deterministic compression for node/tool outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


class ContextCompressor:
    """Compress node updates and archive long raw payloads."""

    def __init__(
        self,
        *,
        archive_dir: str | Path,
        long_text_threshold: int = 2000,
        summary_max_chars: int = 600,
    ) -> None:
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.long_text_threshold = long_text_threshold
        self.summary_max_chars = summary_max_chars

    def compress_update(
        self,
        *,
        run_id: str,
        node: str,
        step: int,
        update: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw = _stringify(update)
        raw_ref = ""
        if len(raw) >= self.long_text_threshold:
            raw_ref = self._archive(run_id=run_id, node=node, step=step, raw=raw)
        return {
            "raw_ref": raw_ref,
            "short_summary": self._summarize(update),
            "token_count_raw": _rough_token_count(raw),
            "token_count_summary": _rough_token_count(self._summarize(update)),
        }

    def _archive(self, *, run_id: str, node: str, step: int, raw: str) -> str:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        filename = f"step_{step}_{_safe_name(node)}_{digest}.json"
        path = self.archive_dir / _safe_name(run_id) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw, encoding="utf-8")
        return str(path.relative_to(self.archive_dir.parent))

    def _summarize(self, update: Dict[str, Any]) -> str:
        pieces = []
        for key in ("output", "result", "answer", "input"):
            if key in update:
                pieces.append(f"{key}: {self._clip(_stringify(update[key]))}")
        for key, value in update.items():
            if key in {"messages", "output", "result", "answer", "input"}:
                continue
            if key.startswith("__"):
                continue
            pieces.append(f"{key}: {self._clip(_stringify(value))}")
            if len(pieces) >= 6:
                break
        if not pieces and "messages" in update:
            pieces.append(f"messages: {self._clip(_stringify(update['messages']))}")
        return "\n".join(pieces)

    def _clip(self, text: str) -> str:
        if len(text) <= self.summary_max_chars:
            return text
        return text[: self.summary_max_chars - 3] + "..."


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _rough_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value) or "item"
