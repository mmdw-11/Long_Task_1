"""Application-facing primary/reference memory-bank adapter."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable, List, Optional

from ._types import MemoryContext, MemoryItem, MemoryScope, WakeupProfile, WakeupResult, wakeup_profile
from .base import MemoryStore
from .store import HybridTieredMemoryStore


class MemoryBankRuntime(MemoryStore):
    """Read from every bound bank while writing only to the primary bank."""

    def __init__(self, root_dir: str | Path, primary_id: str, reference_ids: Iterable[str] = ()) -> None:
        root = Path(root_dir)
        ids = list(dict.fromkeys([primary_id, *reference_ids]))
        self.primary_id = primary_id
        self._stores = {bank_id: HybridTieredMemoryStore(root / bank_id) for bank_id in ids}

    @property
    def primary(self) -> HybridTieredMemoryStore:
        return self._stores[self.primary_id]

    def write(self, item: MemoryItem, *, context: Optional[MemoryContext] = None) -> None:
        item.metadata = {**item.metadata, "memory_bank_id": self.primary_id}
        self.primary.write(item, context=context)

    def read(self, query: str, *, scope: Optional[MemoryScope] = None, context: Optional[MemoryContext] = None, top_k: int = 5) -> List[MemoryItem]:
        candidates: List[MemoryItem] = []
        for bank_id, store in self._stores.items():
            for item in store.read(query, scope=scope, context=context, top_k=top_k):
                item.metadata = {**item.metadata, "memory_bank_id": bank_id}
                candidates.append(item)
        return _dedupe(candidates)[:top_k]

    def cascade_read(self, query: str, *, narrowest: MemoryScope = MemoryScope.WORKING, context: Optional[MemoryContext] = None, top_k: int = 5) -> List[MemoryItem]:
        candidates: List[MemoryItem] = []
        for bank_id, store in self._stores.items():
            for item in store.cascade_read(query, narrowest=narrowest, context=context, top_k=top_k):
                item.metadata = {**item.metadata, "memory_bank_id": bank_id}
                candidates.append(item)
        return _dedupe(candidates)[:top_k]

    def wake(self, query: str, *, profile: WakeupProfile | int | str = 1, context: Optional[MemoryContext] = None) -> WakeupResult:
        effective = profile if isinstance(profile, WakeupProfile) else wakeup_profile(profile)
        items: List[MemoryItem] = []
        for bank_id, store in self._stores.items():
            result = store.wake(query, profile=effective, context=context)
            for item in result.items:
                item.metadata = {**item.metadata, "memory_bank_id": bank_id}
                items.append(item)
        selected = _dedupe(items)[:effective.top_k]
        return WakeupResult(profile=effective, items=selected, context_text=self.primary.format_wakeup_context(selected, effective))

    def clear(self, scope: Optional[MemoryScope] = None, *, context: Optional[MemoryContext] = None) -> None:
        self.primary.clear(scope, context=context)

    def delete(self, memory_id: str) -> None:
        self.primary.delete(memory_id)

    def close(self) -> None:
        for store in self._stores.values():
            store.close()


def list_bank_memories(root_dir: str | Path, bank_id: str, *, query: str = "", scope: Optional[MemoryScope] = None, limit: int = 100) -> List[MemoryItem]:
    store = HybridTieredMemoryStore(Path(root_dir) / bank_id)
    try:
        scopes = [scope] if scope else MemoryScope.hierarchy()
        items: List[MemoryItem] = []
        for item_scope in scopes:
            items.extend(item for item in store.read(query, scope=item_scope, top_k=limit) if not _expired(item))
        return sorted(_dedupe(items), key=lambda item: item.ts, reverse=True)[:limit]
    finally:
        store.close()


def _dedupe(items: Iterable[MemoryItem]) -> List[MemoryItem]:
    seen = set()
    result = []
    for item in sorted(items, key=lambda value: (value.importance, value.ts), reverse=True):
        key = str(item.content).strip().casefold()
        if item.id in seen or key in seen:
            continue
        seen.update({item.id, key})
        result.append(item)
    return result


def _expired(item: MemoryItem) -> bool:
    value=item.metadata.get("expires_at")
    if not value:return False
    try:return datetime.fromisoformat(str(value).replace("Z","+00:00"))<=datetime.now(timezone.utc)
    except ValueError:return False
