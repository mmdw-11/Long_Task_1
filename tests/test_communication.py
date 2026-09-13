import asyncio

from engine import (
    CommunicationManager,
    CommunicationPolicy,
    EvidenceRef,
    MessageCapsule,
    Orchestrator,
)


def test_capsule_digest_is_canonical_and_roundtrips():
    first = MessageCapsule(sender="researcher", claim="x", constraint_delta={"b": 2, "a": 1})
    second = MessageCapsule.from_dict(first.to_dict())
    assert first.digest == second.digest
    second.claim = "changed"
    second.refresh_digest()
    assert second.digest != first.digest


def test_gate_prunes_duplicate_but_preserves_constraint_delta():
    manager = CommunicationManager(CommunicationPolicy(min_contribution=0.2))
    capsule = MessageCapsule(sender="a", recipients=["b"], goal="fix tests", claim="fix tests", evidence=[EvidenceRef(content="failure", confidence=.9)])
    first, _ = manager.route(capsule, candidates=["b"])
    duplicate = MessageCapsule(sender="a", recipients=["b"], goal="fix tests", claim="fix tests")
    second, events = manager.route(duplicate, candidates=["b"])
    protected = MessageCapsule(sender="a", recipients=["b"], claim="", constraint_delta={"deadline": "today"})
    third, _ = manager.route(protected, candidates=["b"])
    assert first and not second and third
    assert any(event["type"] == "capsule_pruned" for event in events)


def test_orchestrator_emits_capsule_events_and_targets_successor():
    manager = CommunicationManager(CommunicationPolicy(min_contribution=-1))
    orch = Orchestrator()
    a = orch.create_agent("planner", config={"role": "planner"})
    b = orch.create_agent("worker", config={"role": "worker"})
    orch.connect(a, b); orch.set_entry(a); orch.set_communication_manager(manager)
    events = []

    async def run():
        async for event in orch.build_graph().astream({"run_id": "comm-run", "goal": "ship"}):
            events.append(event)
    asyncio.run(run())
    assert any(event["type"] == "capsule_created" for event in events)
    delivered = [event for event in events if event["type"] == "capsule_delivered"]
    assert delivered and delivered[0]["recipient"] == "worker"


def test_expired_capsule_is_never_delivered():
    manager = CommunicationManager(clock=lambda: 100.0)
    capsule = MessageCapsule(sender="a", ttl=0, claim="old")
    envelopes, events = manager.route(capsule, candidates=["b"])
    assert envelopes == []
    assert events[-1]["decision"]["reasons"] == ["expired"]


def test_novel_claim_is_preserved_despite_sparse_optional_fields():
    manager = CommunicationManager(CommunicationPolicy(min_contribution=.99, novelty_threshold=.70))
    first, _ = manager.route(MessageCapsule(sender="a", recipients=["b"], goal="release", claim="[ERR-1] date is wrong"), candidates=["b"])
    second, events = manager.route(MessageCapsule(sender="c", recipients=["b"], goal="release", claim="[ALT-1] version is independent"), candidates=["b"])
    assert first and second
    assert events[-1]["decision"]["reasons"] == ["novel_claim_protected"]


def test_state_delta_excludes_previous_capsule_envelope():
    manager = CommunicationManager()
    state = {"goal": "ship", "capsule": MessageCapsule(sender="a", claim="nested"), "messages": [{"content": "old"}]}
    delta = manager.diff_state("a", state)
    assert "capsule" not in delta.changed
    assert "messages" not in delta.changed
    assert "goal" not in delta.changed  # goal is already a first-class Capsule field


def test_conflicting_fact_values_are_delivered_and_audited():
    manager = CommunicationManager()
    wrong = MessageCapsule(sender="a", recipients=["b"], claim_id="wrong", fact_key="release_date", claim="2026-08-01", evidence=[EvidenceRef(content="unverified", confidence=.1)])
    correct = MessageCapsule(sender="c", recipients=["b"], claim_id="evidence", fact_key="release_date", kind="evidence", claim="2026-09-01", evidence=[EvidenceRef(content="official", confidence=.95)])
    first, _ = manager.route(wrong, candidates=["b"])
    second, events = manager.route(correct, candidates=["b"])
    assert first and second
    assert any(event["type"] == "capsule_conflict_detected" for event in events)
    assert second[0].capsule.metadata["conflict"] is True


def test_prompt_view_excludes_audit_and_state_delta_fields():
    capsule = MessageCapsule(sender="a", claim_id="c1", fact_key="release_date", claim="2026-09-01", state_delta=None)
    prompt = capsule.to_prompt_dict()
    assert prompt["claim"] == "2026-09-01"
    assert "state_delta" not in prompt
    assert "provenance" not in prompt
    assert "digest" not in prompt
