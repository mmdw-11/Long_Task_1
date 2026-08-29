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
