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
    # A non-redundant claim is an information-bearing unit.  Do not discard it
    # merely because its optional evidence/metadata fields are sparse; send it
    # as a clearly labelled low-confidence capsule and let the receiver decide.
    preserve_novel_claims: bool = True
    novelty_threshold: float = 0.70


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
        capsule.state_delta = self.diff_state(sender, state, update=update)
        capsule.state_revision = capsule.state_delta.revision
        capsule.refresh_digest(); self.metrics["created"] += 1
        return capsule

    def diff_state(self, sender: str, state: Dict[str, Any], *, update: Optional[Dict[str, Any]] = None) -> StateDelta:
        prior = self._snapshots.get(sender, {})
        # ``capsule`` is an explicit output protocol field, but is not business
        # state.  Persisting it in StateDelta makes the previous envelope nest
        # inside the next one, inflates token pressure, and defeats the point of
        # a bounded communication channel.
        # State deltas are opt-in-by-output: transporting the entire shared
        # graph state turns a compact Capsule back into a hidden full-history
        # channel.  Fields already represented by Capsule itself (claim,
        # evidence, goal) and runtime bookkeeping never cross this boundary.
        source = update if isinstance(update, dict) else state
        excluded = {"messages", "capsule", "input", "run_id", "task_id", "trace_id", "goal", "original_goal", "citations", "retrieval_metadata", sender}
        visible = {k: v for k, v in source.items() if not k.startswith("__") and k not in excluded}
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
            conflict = self._detect_conflict(capsule, recipient)
            if conflict:
                capsule = copy.deepcopy(capsule)
                capsule.metadata = {**capsule.metadata, "conflict": True, "conflicts_with": conflict}
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
            if conflict:
                events.append({"type": "capsule_conflict_detected", "recipient": recipient, "capsule": delivered.to_dict(), "conflicts_with": conflict})
            if delivered.state_delta and (delivered.state_delta.changed or delivered.state_delta.removed): events.append({"type": "state_delta", "sender": delivered.sender, "recipient": recipient, "delta": delivered.state_delta.to_dict()})
            envelopes.append(envelope)
        return envelopes, events

    def evaluate(self, capsule: MessageCapsule, *, recipient: str, role: str = "") -> CapsuleDecision:
        if capsule.is_expired(self.clock()): return CapsuleDecision("prune", 0.0, ["expired"], {"freshness": 0.0})
        text = self._capsule_text(capsule)
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
        known_fact_keys = {envelope.capsule.fact_key for envelope in self._mailboxes.get(recipient, []) if envelope.capsule.fact_key}
        # A distinct fact slot is semantically novel even if prose shares much
        # of its surrounding wording with an earlier message.
        novel_claim = bool(capsule.claim.strip()) and ((bool(capsule.fact_key) and capsule.fact_key not in known_fact_keys) or duplicate < self.policy.novelty_threshold)
        if protected:
            action, reasons = "deliver", ["protected_semantics"]
        elif redundant:
            action, reasons = "prune", ["redundant"]
        elif self.policy.preserve_novel_claims and novel_claim:
            action, reasons = "deliver", ["novel_claim_protected"]
        elif score >= self.policy.min_contribution:
            action, reasons = "deliver", ["contribution_threshold_met"]
        else:
            action, reasons = "prune", ["low_contribution"]
        return CapsuleDecision(action, round(score, 6), reasons, {k: round(v, 6) for k, v in c.items()})

    def receive(self, recipient: str, *, consume: bool = False) -> List[CapsuleEnvelope]:
        items = list(self._mailboxes.get(recipient, []))
        if consume: self._mailboxes[recipient] = []
        return items

    def inject(self, state: Dict[str, Any], recipient: str) -> None:
        items = self.receive(recipient)
        # The model receives the compact semantic protocol; the full envelope
        # remains available in events/checkpoints for reproducibility.
        state[CAPSULE_CONTEXT_KEY] = [
            {"capsule": item.capsule.to_prompt_dict(), "decision": item.decision.to_dict()}
            for item in items
        ]

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
    def _capsule_text(c: MessageCapsule) -> str:
        # ``subtask`` commonly defaults to the sender node name.  Including
        # it makes two otherwise identical relays look artificially distinct
        # solely because they came from different agents, weakening duplicate
        # suppression at a fan-in recipient.  Goal/claim/next-action are the
        # recipient-relevant semantic payload for redundancy comparison.
        return " ".join([c.goal, c.claim, c.next_action])

    def _detect_conflict(self, capsule: MessageCapsule, recipient: str) -> Dict[str, Any] | None:
        """Mark contradictory fact values; never hide either side of a conflict."""
        if not capsule.fact_key or not capsule.claim.strip():
            return None
        incoming = _normalized_claim(capsule.claim)
        for envelope in self._mailboxes.get(recipient, []):
            existing = envelope.capsule
            if existing.fact_key != capsule.fact_key or not existing.claim.strip():
                continue
            if _normalized_claim(existing.claim) == incoming:
                continue
            return {
                "fact_key": capsule.fact_key,
                "existing_claim_id": existing.claim_id,
                "existing_sender": existing.sender,
                "existing_confidence": _confidence(existing),
                "incoming_confidence": _confidence(capsule),
            }
        return None


def _tokens(text: str) -> set[str]: return set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))
def _similarity(a: str, b: str) -> float:
    x, y = _tokens(a), _tokens(b)
    return len(x & y) / len(x | y) if x and y else 0.0
def _overlap(a: str, b: str) -> float:
    x, y = _tokens(a), _tokens(b)
    return len(x & y) / len(x) if x else (0.5 if y else 0.0)


def _normalized_claim(value: str) -> str:
    return " ".join(sorted(_tokens(value)))


def _confidence(capsule: MessageCapsule) -> float:
    if capsule.evidence:
        return max(max(0.0, min(1.0, float(item.confidence))) for item in capsule.evidence)
    return max(0.0, min(1.0, 1.0 - float(capsule.uncertainty)))
