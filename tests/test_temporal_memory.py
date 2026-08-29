from engine import (
    Fact,
    FactStatus,
    HybridTieredMemoryStore,
    MessageCapsule,
    Provenance,
    TemporalEvidence,
    TemporalEvidenceMemoryStore,
)


def make_store(tmp_path, now=100.0):
    base = HybridTieredMemoryStore(tmp_path, enable_memory_update=False)
    return base, TemporalEvidenceMemoryStore(base, clock=lambda: now)


def test_temporal_schema_is_additive_and_fact_can_be_retrieved(tmp_path):
    base, store = make_store(tmp_path)
    fact = Fact(subject="task-1", predicate="status", object="running", valid_from=10, task_id="task-1")
    stored = store.upsert_fact(fact, [TemporalEvidence(content="tool says running", source="api", source_type="tool", observed_at=10)])
    results = store.retrieve_facts("task status running", as_of=20, task_id="task-1")
    assert results[0].fact.id == stored.id
    assert results[0].score_breakdown["temporal"] == 1.0
    assert base._get_conn().execute("SELECT version FROM temporal_schema").fetchone()[0] == 1
    base.close()


def test_conflict_is_explicit_and_hidden_by_default(tmp_path):
    base, store = make_store(tmp_path)
    a = store.upsert_fact(Fact(subject="task", predicate="owner", object="Alice", valid_from=1))
    b = store.upsert_fact(Fact(subject="task", predicate="owner", object="Bob", valid_from=1))
    assert a.id != b.id
    conflicts = store.detect_conflicts(subject="task", predicate="owner")
    assert len(conflicts) == 1 and {x.status for x in conflicts[0]} == {FactStatus.DISPUTED}
    assert store.retrieve_facts("task owner", as_of=2) == []
    assert len(store.retrieve_facts("task owner", as_of=2, include_conflicts=True)) == 2
    base.close()


def test_replace_closes_old_half_open_interval_and_builds_version_chain(tmp_path):
    base, store = make_store(tmp_path)
    old = store.upsert_fact(Fact(subject="task", predicate="status", object="planned", valid_from=1))
    new = store.upsert_fact(Fact(subject="task", predicate="status", object="running", valid_from=10), replace=True)
    history = store.get_fact_history("task", "status")
    assert history[0].status == FactStatus.SUPERSEDED and history[0].valid_to == 10
    assert new.supersedes_id == old.id and new.version == 2
    assert store.retrieve_facts("status planned", as_of=10)[0].fact.object == "running"
    assert store.retrieve_facts("status planned", as_of=9)[0].fact.object == "planned"
    base.close()


def test_capsule_extracts_claim_and_constraint_facts(tmp_path):
    base, store = make_store(tmp_path)
    capsule = MessageCapsule(sender="planner", goal="release", subtask="plan", claim="tests pass", constraint_delta={"deadline": "Friday"}, provenance=Provenance(task_id="t1", agent_id="planner"))
    facts = store.extract_from_capsule(capsule)
    assert {fact.predicate for fact in facts} == {"claims", "constraint:deadline"}
    assert all(fact.task_id == "t1" for fact in facts)
    base.close()
