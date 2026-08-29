"""Structured, low-entropy communication for graph agents."""

from .types import (
    CapsuleDecision,
    CapsuleEnvelope,
    CommunicationBudget,
    EvidenceRef,
    MessageCapsule,
    Provenance,
    StateDelta,
)
from .compressor import CapsuleCompressor, DeterministicCapsuleCompressor, LLMLinguaCapsuleCompressor
from .manager import (
    CAPSULE_CONTEXT_KEY,
    CAPSULE_EVENTS_KEY,
    CAPSULE_OUTBOX_KEY,
    COMMUNICATION_STATE_KEY,
    CommunicationManager,
    CommunicationPolicy,
)

__all__ = [
    "CAPSULE_CONTEXT_KEY", "CAPSULE_EVENTS_KEY", "CAPSULE_OUTBOX_KEY",
    "COMMUNICATION_STATE_KEY", "CapsuleDecision", "CapsuleEnvelope",
    "CapsuleCompressor", "CommunicationBudget", "CommunicationManager",
    "CommunicationPolicy", "DeterministicCapsuleCompressor", "EvidenceRef",
    "MessageCapsule", "Provenance", "StateDelta",
    "LLMLinguaCapsuleCompressor",
]
