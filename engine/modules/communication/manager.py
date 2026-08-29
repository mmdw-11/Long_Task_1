from __future__ import annotations

import copy
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .compressor import CapsuleCompressor, DeterministicCapsuleCompressor, rough_tokens
from .types import CapsuleDecision, CapsuleEnvelope, MessageCapsule, Provenance, StateDelta

CAPSULE_CONTEXT_KEY = "__capsule_context__"
CAPSULE_OUTBOX_KEY = "__capsule_outbox__"
CAPSULE_EVENTS_KEY = "__communication_events__"
COMMUNICATION_STATE_KEY = "__communication_state__"


@dataclass
class CommunicationPolicy:
    min_contribution: float = 0.28
    max_capsule_tokens: int = 512
    redundancy_threshold: float = 0.82
    task_relevance_weight: float = 0.24
    information_gain_weight: float = 0.22
    evidence_quality_weight: float = 0.18
    role_fit_weight: float = 0.12
    freshness_weight: float = 0.10
    redundancy_weight: float = 0.20
    budget_pressure_weight: float = 0.10
    broadcast_when_unresolved: bool = True
    preserve_constraints: bool = True
    preserve_conflicts: bool = True


class CommunicationManager:
    def __init__(self, policy: Optional[CommunicationPolicy] = None, *, compressor: Optional[CapsuleCompressor] = None, clock: Callable[[], float] = time.time) -> None:
        self.policy = policy or CommunicationPolicy()
        self.compressor = compressor or DeterministicCapsuleCompressor()
        self.clock = clock
        self._mailboxes: Dict[str, List[CapsuleEnvelope]] = {}
        self._seen: Dict[str, set[str]] = {}
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self._revisions: Dict[str, int] = {}
        self.metrics = {"created": 0, "delivered": 0, "pruned": 0, "compressed": 0, "original_tokens": 0, "delivered_tokens": 0, "broadcasts": 0}

    def capsule_from_update(self, update: Dict[str, Any], *, sender: str, recipients: Optional[List[str]], state: Dict[str, Any], step: int) -> MessageCapsule:
        explicit = update.get(CAPSULE_OUTBOX_KEY) or update.get("capsule")
        if isinstance(explicit, MessageCapsule): capsule = copy.deepcopy(explicit)
        elif isinstance(explicit, dict): capsule = MessageCapsule.from_dict(explicit)
        else:
            messages = update.get("messages") or []
            candidate = messages[-1] if isinstance(messages, list) and messages else update.get("result") or update.get("output") or update.get("answer") or ""
            capsule = MessageCapsule.from_legacy(candidate, sender=sender, recipients=recipients, provenance=Provenance(task_id=str(state.get("run_id") or state.get("task_id") or ""), agent_id=sender, trace_id=str(state.get("trace_id") or ""), node=sender, step=step))
            capsule.goal = str(state.get("goal") or state.get("original_goal") or "")
            capsule.subtask = str(update.get("subtask") or sender)
            capsule.next_action = str(update.get("next_action") or "")
            capsule.constraint_delta = dict(update.get("constraint_delta") or {})
        capsule.recipients = list(capsule.recipients or recipients or [])
        capsule.state_delta = self.diff_state(sender, state)
        capsule.state_revision = capsule.state_delta.revision
        capsule.refresh_digest(); self.metrics["created"] += 1
        return capsule

    def diff_state(self, sender: str, state: Dict[str, Any]) -> StateDelta:
        prior = self._snapshots.get(sender, {})
        visible = {k: v for k, v in state.items() if not k.startswith("__") and k != "messages"}
        changed = {k: v for k, v in visible.items() if prior.get(k) != v}
        removed = [k for k in prior if k not in visible]
        base = self._revisions.get(sender, 0); revision = base + 1
        self._snapshots[sender] = copy.deepcopy(visible); self._revisions[sender] = revision
        return StateDelta(base_revision=base, revision=revision, changed=changed, removed=removed)

    def route(self, capsule: MessageCapsule, *, candidates: Iterable[str], roles: Optional[Dict[str, str]] = None) -> Tuple[List[CapsuleEnvelope], List[Dict[str, Any]]]:
        resolved = [x for x in capsule.recipients if x]
        if not resolved: resolved = list(dict.fromkeys(candidates))
        if not resolved and self.policy.broadcast_when_unresolved: resolved = list((roles or {}).keys()); self.metrics["broadcasts"] += 1
        events: List[Dict[str, Any]] = [{"type": "capsule_created", "capsule": capsule.to_dict()}]
        envelopes: List[CapsuleEnvelope] = []
        for recipient in resolved:
            decision = self.evaluate(capsule, recipient=recipient, role=(roles or {}).get(recipient, ""))
            if decision.action == "prune":
                self.metrics["pruned"] += 1; events.append({"type": "capsule_pruned", "recipient": recipient, "digest": capsule.digest, "decision": decision.to_dict()}); continue
            delivered = capsule
            self.metrics["original_tokens"] += rough_tokens(capsule.to_dict())
            if rough_tokens(capsule.to_dict()) > self.policy.max_capsule_tokens:
                delivered = self.compressor.compress(capsule, max_tokens=self.policy.max_capsule_tokens)
                if delivered.metadata.get("compressed"):
                    self.metrics["compressed"] += 1; events.append({"type": "capsule_compressed", "recipient": recipient, "digest": delivered.digest, "parent_digest": delivered.parent_digest, "original_tokens": delivered.budget.original_tokens, "compressed_tokens": delivered.budget.compressed_tokens})
            envelope = CapsuleEnvelope(delivered, recipient, decision)
            self._mailboxes.setdefault(recipient, []).append(envelope); self._seen.setdefault(recipient, set()).add(delivered.digest)
            self.metrics["delivered"] += 1; self.metrics["delivered_tokens"] += rough_tokens(delivered.to_dict())
            events.append({"type": "capsule_delivered", "recipient": recipient, "capsule": delivered.to_dict(), "decision": decision.to_dict()})
            if delivered.state_delta and (delivered.state_delta.changed or delivered.state_delta.removed): events.append({"type": "state_delta", "sender": delivered.sender, "recipient": recipient, "delta": delivered.state_delta.to_dict()})
            envelopes.append(envelope)
        return envelopes, events

    def evaluate(self, capsule: MessageCapsule, *, recipient: str, role: str = "") -> CapsuleDecision:
        if capsule.is_expired(self.clock()): return CapsuleDecision("prune", 0.0, ["expired"], {"freshness": 0.0})
        text = " ".join([capsule.goal, capsule.subtask, capsule.claim, capsule.next_action])
        relevance = _overlap(capsule.goal + " " + capsule.subtask, capsule.claim + " " + role)
        duplicate = max((_similarity(text, self._capsule_text(e.capsule)) for e in self._mailboxes.get(recipient, [])), default=0.0)
        info = 1.0 - duplicate
        evidence = sum(max(0.0, min(1.0, e.confidence)) for e in capsule.evidence) / max(1, len(capsule.evidence))
        role_fit = 1.0 if not role or role.lower() in text.lower() or recipient in capsule.recipients else 0.45
        freshness = 1.0 if capsule.expires_at is None else max(0.0, min(1.0, (capsule.expires_at - self.clock()) / 3600.0))
        pressure = min(1.0, rough_tokens(capsule.to_dict()) / max(1, capsule.budget.max_tokens))
        c = {"task_relevance": relevance, "information_gain": info, "evidence_quality": evidence, "role_fit": role_fit, "freshness": freshness, "redundancy": duplicate, "budget_pressure": pressure}
        score = self.policy.task_relevance_weight*relevance + self.policy.information_gain_weight*info + self.policy.evidence_quality_weight*evidence + self.policy.role_fit_weight*role_fit + self.policy.freshness_weight*freshness - self.policy.redundancy_weight*duplicate - self.policy.budget_pressure_weight*pressure
        protected = bool(capsule.constraint_delta and self.policy.preserve_constraints) or bool(capsule.metadata.get("conflict") and self.policy.preserve_conflicts)
        redundant = duplicate >= self.policy.redundancy_threshold
        action = "deliver" if protected or (not redundant and score >= self.policy.min_contribution) else "prune"
        reasons = (["protected_semantics"] if protected else []) + (["redundant"] if redundant and not protected else []) + (["low_contribution"] if action == "prune" and not redundant else []) + (["contribution_threshold_met"] if action == "deliver" and not protected else [])
        return CapsuleDecision(action, round(score, 6), reasons, {k: round(v, 6) for k, v in c.items()})

    def receive(self, recipient: str, *, consume: bool = False) -> List[CapsuleEnvelope]:
        items = list(self._mailboxes.get(recipient, []))
        if consume: self._mailboxes[recipient] = []
        return items

    def inject(self, state: Dict[str, Any], recipient: str) -> None:
        items = self.receive(recipient)
        state[CAPSULE_CONTEXT_KEY] = [x.to_dict() for x in items]

    def snapshot(self) -> Dict[str, Any]:
        return {"schema_version": 1, "revisions": dict(self._revisions), "seen_digests": {k: sorted(v) for k, v in self._seen.items()}, "metrics": dict(self.metrics)}

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore checkpoint-safe watermarks; payload mailboxes remain in graph state."""
        if int(snapshot.get("schema_version", 1)) != 1:
            raise ValueError("unsupported communication snapshot schema")
        self._revisions = {str(k): int(v) for k, v in (snapshot.get("revisions") or {}).items()}
        self._seen = {str(k): set(v) for k, v in (snapshot.get("seen_digests") or {}).items()}
        self.metrics.update({k: int(v) for k, v in (snapshot.get("metrics") or {}).items() if k in self.metrics})

    @staticmethod
    def _capsule_text(c: MessageCapsule) -> str: return " ".join([c.goal, c.subtask, c.claim, c.next_action])


def _tokens(text: str) -> set[str]: return set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))
def _similarity(a: str, b: str) -> float:
    x, y = _tokens(a), _tokens(b)
    return len(x & y) / len(x | y) if x and y else 0.0
def _overlap(a: str, b: str) -> float:
    x, y = _tokens(a), _tokens(b)
    return len(x & y) / len(x) if x else (0.5 if y else 0.0)
