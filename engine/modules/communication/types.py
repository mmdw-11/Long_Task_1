from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


@dataclass
class EvidenceRef:
    content: str = ""
    uri: str = ""
    source: str = ""
    confidence: float = 0.5
    observed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]: return asdict(self)
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceRef": return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Provenance:
    task_id: str = ""
    agent_id: str = ""
    trace_id: str = ""
    node: str = ""
    step: int = 0
    source_type: str = "agent"

    def to_dict(self) -> Dict[str, Any]: return asdict(self)
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Provenance": return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CommunicationBudget:
    max_tokens: int = 512
    used_tokens: int = 0
    original_tokens: int = 0
    compressed_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]: return asdict(self)
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommunicationBudget": return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StateDelta:
    base_revision: int = 0
    revision: int = 0
    changed: Dict[str, Any] = field(default_factory=dict)
    removed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]: return asdict(self)
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateDelta": return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MessageCapsule:
    sender: str
    recipients: List[str] = field(default_factory=list)
    goal: str = ""
    subtask: str = ""
    claim: str = ""
    evidence: List[EvidenceRef] = field(default_factory=list)
    constraint_delta: Dict[str, Any] = field(default_factory=dict)
    next_action: str = ""
    uncertainty: float = 0.0
    ttl: int = 3
    expires_at: Optional[float] = None
    budget: CommunicationBudget = field(default_factory=CommunicationBudget)
    provenance: Provenance = field(default_factory=Provenance)
    state_delta: Optional[StateDelta] = None
    id: str = field(default_factory=lambda: f"cap-{uuid.uuid4().hex}")
    schema_version: str = "1.0"
    digest: str = ""
    created_at: float = field(default_factory=time.time)
    parent_digest: str = ""
    state_revision: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.uncertainty = min(1.0, max(0.0, float(self.uncertainty)))
        self.ttl = max(0, int(self.ttl))
        if not self.digest:
            self.digest = self.compute_digest()

    def digest_payload(self) -> Dict[str, Any]:
        data = self.to_dict()
        for key in ("id", "digest", "created_at"):
            data.pop(key, None)
        return data

    def compute_digest(self) -> str:
        return hashlib.sha256(_canonical(self.digest_payload()).encode("utf-8")).hexdigest()

    def refresh_digest(self) -> None: self.digest = self.compute_digest()

    def is_expired(self, now: Optional[float] = None) -> bool:
        return self.ttl <= 0 or (self.expires_at is not None and (now or time.time()) >= self.expires_at)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessageCapsule":
        payload = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        payload["evidence"] = [x if isinstance(x, EvidenceRef) else EvidenceRef.from_dict(x) for x in payload.get("evidence", []) if isinstance(x, (dict, EvidenceRef))]
        if isinstance(payload.get("budget"), dict): payload["budget"] = CommunicationBudget.from_dict(payload["budget"])
        if isinstance(payload.get("provenance"), dict): payload["provenance"] = Provenance.from_dict(payload["provenance"])
        if isinstance(payload.get("state_delta"), dict): payload["state_delta"] = StateDelta.from_dict(payload["state_delta"])
        return cls(**payload)

    @classmethod
    def from_legacy(cls, message: Any, *, sender: str, recipients: Optional[List[str]] = None, provenance: Optional[Provenance] = None) -> "MessageCapsule":
        if isinstance(message, dict):
            text = str(message.get("content") or message.get("text") or message.get("output") or message.get("result") or message)
            sender = str(message.get("agent") or message.get("sender") or sender)
        else:
            text = str(message)
        return cls(sender=sender, recipients=list(recipients or []), claim=text, provenance=provenance or Provenance(agent_id=sender), metadata={"legacy": True})


@dataclass
class CapsuleDecision:
    action: str
    score: float
    reasons: List[str] = field(default_factory=list)
    components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]: return asdict(self)


@dataclass
class CapsuleEnvelope:
    capsule: MessageCapsule
    recipient: str
    decision: CapsuleDecision
    delivered_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"capsule": self.capsule.to_dict(), "recipient": self.recipient, "decision": self.decision.to_dict(), "delivered_at": self.delivered_at}
