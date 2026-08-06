"""Structured todo-list management for run-scoped context ledgers."""

from __future__ import annotations

import time
from typing import List

from ._types import TODO_STATUSES, ContextLedger, TodoEvent, TodoItem


class TodoManager:
    """Mutate ``ContextLedger.todo_items`` with audit records."""

    def __init__(self, ledger: ContextLedger) -> None:
        self.ledger = ledger
        self.ensure_initialized()

    def ensure_initialized(self) -> ContextLedger:
        if not self.ledger.todo_items and self.ledger.current_plan:
            now = time.time()
            self.ledger.todo_items = [
                TodoItem(
                    id=f"todo-{idx}",
                    content=item,
                    status="pending",
                    source="current_plan",
                    created_at=now,
                    updated_at=now,
                    revision=self.ledger.todo_revision,
                )
                for idx, item in enumerate(self.ledger.current_plan, 1)
                if str(item).strip()
            ]
            if self.ledger.todo_items and not self.ledger.active_todo_id:
                self.ledger.active_todo_id = self.ledger.todo_items[0].id
                self.ledger.todo_items[0].status = "in_progress"
        self._sync_current_plan()
        self._normalize_active()
        return self.ledger

    def set_todos(
        self,
        contents: List[str],
        *,
        source: str = "model",
        reason: str = "",
    ) -> ContextLedger:
        before = [item.to_dict() for item in self.ledger.todo_items]
        now = time.time()
        self.ledger.todo_revision += 1
        self.ledger.todo_items = [
            TodoItem(
                id=f"todo-{idx}",
                content=str(content),
                status="pending",
                source=source,
                created_at=now,
                updated_at=now,
                revision=self.ledger.todo_revision,
            )
            for idx, content in enumerate(contents, 1)
            if str(content).strip()
        ]
        self.ledger.active_todo_id = self.ledger.todo_items[0].id if self.ledger.todo_items else ""
        if self.ledger.todo_items:
            self.ledger.todo_items[0].status = "in_progress"
        self._record("set_todos", before={"items": before}, after={"items": [item.to_dict() for item in self.ledger.todo_items]}, reason=reason)
        self._sync_current_plan()
        return self.ledger

    def update_todo_status(
        self,
        todo_id: str,
        status: str,
        *,
        evidence: str = "",
        reason: str = "",
    ) -> ContextLedger:
        if status not in TODO_STATUSES:
            raise ValueError(f"invalid todo status: {status}")
        item = self._get(todo_id)
        before = item.to_dict()
        self.ledger.todo_revision += 1
        item.status = status
        item.evidence = evidence or item.evidence
        item.updated_at = time.time()
        item.revision = self.ledger.todo_revision
        if status == "in_progress":
            self.ledger.active_todo_id = item.id
            for other in self.ledger.todo_items:
                if other.id != item.id and other.status == "in_progress":
                    other.status = "pending"
                    other.updated_at = item.updated_at
                    other.revision = self.ledger.todo_revision
        elif self.ledger.active_todo_id == item.id and status in {"completed", "blocked", "cancelled"}:
            self._select_next_active()
        self._record("update_status", todo_id=item.id, before=before, after=item.to_dict(), reason=reason)
        self._sync_current_plan()
        return self.ledger

    def insert_todo(
        self,
        content: str,
        *,
        after_id: str = "",
        source: str = "model",
        reason: str = "",
    ) -> ContextLedger:
        if not str(content).strip():
            raise ValueError("todo content is empty")
        self.ledger.todo_revision += 1
        item = TodoItem(
            id=self._next_id(),
            content=str(content),
            source=source,
            revision=self.ledger.todo_revision,
        )
        index = len(self.ledger.todo_items)
        if after_id:
            index = self._index(after_id) + 1
        self.ledger.todo_items.insert(index, item)
        if not self.ledger.active_todo_id:
            self.ledger.active_todo_id = item.id
            item.status = "in_progress"
        self._record("insert", todo_id=item.id, after=item.to_dict(), reason=reason)
        self._sync_current_plan()
        return self.ledger

    def replace_todo(
        self,
        todo_id: str,
        content: str,
        *,
        reason: str = "",
    ) -> ContextLedger:
        if not str(content).strip():
            raise ValueError("todo content is empty")
        item = self._get(todo_id)
        before = item.to_dict()
        self.ledger.todo_revision += 1
        item.content = str(content)
        item.updated_at = time.time()
        item.revision = self.ledger.todo_revision
        self._record("replace", todo_id=item.id, before=before, after=item.to_dict(), reason=reason)
        self._sync_current_plan()
        return self.ledger

    def cancel_todo(
        self,
        todo_id: str,
        *,
        reason: str = "",
    ) -> ContextLedger:
        return self.update_todo_status(todo_id, "cancelled", reason=reason)

    def select_active_todo(
        self,
        todo_id: str,
        *,
        reason: str = "",
    ) -> ContextLedger:
        return self.update_todo_status(todo_id, "in_progress", reason=reason)

    def _get(self, todo_id: str) -> TodoItem:
        for item in self.ledger.todo_items:
            if item.id == todo_id:
                return item
        raise KeyError(f"unknown todo id: {todo_id}")

    def _index(self, todo_id: str) -> int:
        for idx, item in enumerate(self.ledger.todo_items):
            if item.id == todo_id:
                return idx
        raise KeyError(f"unknown todo id: {todo_id}")

    def _next_id(self) -> str:
        existing = {item.id for item in self.ledger.todo_items}
        idx = len(existing) + 1
        while f"todo-{idx}" in existing:
            idx += 1
        return f"todo-{idx}"

    def _select_next_active(self) -> None:
        self.ledger.active_todo_id = ""
        for item in self.ledger.todo_items:
            if item.status == "pending":
                item.status = "in_progress"
                item.updated_at = time.time()
                item.revision = self.ledger.todo_revision
                self.ledger.active_todo_id = item.id
                return

    def _normalize_active(self) -> None:
        active = None
        for item in self.ledger.todo_items:
            if item.id == self.ledger.active_todo_id:
                active = item
                break
        if active is None or active.status in {"completed", "blocked", "cancelled"}:
            self._select_next_active()
            return
        for item in self.ledger.todo_items:
            if item.id != active.id and item.status == "in_progress":
                item.status = "pending"

    def _sync_current_plan(self) -> None:
        if self.ledger.todo_items:
            self.ledger.current_plan = [
                item.content
                for item in self.ledger.todo_items
                if item.status != "cancelled"
            ]

    def _record(
        self,
        action: str,
        *,
        todo_id: str = "",
        before: dict | None = None,
        after: dict | None = None,
        reason: str = "",
    ) -> None:
        self.ledger.todo_events.append(
            TodoEvent(
                revision=self.ledger.todo_revision,
                action=action,
                todo_id=todo_id,
                before=before or {},
                after=after or {},
                reason=reason,
            )
        )
