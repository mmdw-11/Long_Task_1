from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from typing import Any, Dict

from .types import MessageCapsule


def rough_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return 0 if not text else max(1, len(text) // 4)


class CapsuleCompressor(ABC):
    @abstractmethod
    def compress(self, capsule: MessageCapsule, *, max_tokens: int) -> MessageCapsule: ...


class DeterministicCapsuleCompressor(CapsuleCompressor):
    """Field-aware compression that never removes claims or constraint changes."""

    def compress(self, capsule: MessageCapsule, *, max_tokens: int) -> MessageCapsule:
        result = copy.deepcopy(capsule)
        original = rough_tokens(result.to_dict())
        result.parent_digest = capsule.digest
        seen = set()
        unique = []
        for evidence in result.evidence:
            key = (evidence.uri, evidence.content.strip().lower(), evidence.source)
            if key not in seen:
                seen.add(key); unique.append(evidence)
        result.evidence = unique
        optional = [("goal", 180), ("subtask", 220), ("next_action", 180)]
        for field_name, limit in optional:
            value = getattr(result, field_name)
            if len(value) > limit: setattr(result, field_name, value[: limit - 1] + "…")
        for evidence in result.evidence:
            if len(evidence.content) > 240: evidence.content = evidence.content[:239] + "…"
        while rough_tokens(result.to_dict()) > max_tokens and len(result.evidence) > 1:
            result.evidence.pop()
        result.budget.original_tokens = original
        result.budget.compressed_tokens = rough_tokens(result.to_dict())
        result.budget.used_tokens = result.budget.compressed_tokens
        result.metadata = {**result.metadata, "compressed": result.budget.compressed_tokens < original, "faithful": bool(result.claim or result.constraint_delta)}
        result.refresh_digest()
        return result


class LLMLinguaCapsuleCompressor(CapsuleCompressor):
    """Optional adapter; imports LLMLingua only when instantiated."""

    def __init__(self, model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank") -> None:
        try:
            from llmlingua import PromptCompressor  # type: ignore
        except ImportError as exc:
            raise RuntimeError("install the 'communication' extra to use LLMLingua") from exc
        self._backend = PromptCompressor(model_name=model_name, use_llmlingua2=True)
        self._fallback = DeterministicCapsuleCompressor()

    def compress(self, capsule: MessageCapsule, *, max_tokens: int) -> MessageCapsule:
        result = self._fallback.compress(capsule, max_tokens=max_tokens)
        # Claims and constraints are invariants; only descriptive fields are model-compressed.
        for field_name in ("goal", "subtask", "next_action"):
            text = getattr(result, field_name)
            if text:
                payload = self._backend.compress_prompt(text, rate=.5, force_tokens=["\n", ".", "。"])
                setattr(result, field_name, str(payload.get("compressed_prompt") or text))
        result.refresh_digest()
        return result
