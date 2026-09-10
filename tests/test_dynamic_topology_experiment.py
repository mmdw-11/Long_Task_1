from __future__ import annotations

from engine.experiments.dynamic_topology import (
    FAMILIES,
    build_dynamic_topology_dataset,
    dataset_manifest,
    load_dynamic_topology_dataset,
)


def test_dynamic_topology_dataset_is_balanced_and_frozen(tmp_path):
    path = tmp_path / "dynamic.jsonl"
    rows = build_dynamic_topology_dataset(path)
    assert len(rows) == 40
    assert [sum(item["family"] == family for item in rows) for family in FAMILIES] == [10, 10, 10, 10]
    assert load_dynamic_topology_dataset(path) == rows
    assert len(dataset_manifest(rows)["sha256"]) == 64


def test_dynamic_topology_dataset_does_not_leak_answers_or_agent_ids(tmp_path):
    rows = build_dynamic_topology_dataset(tmp_path / "dynamic.jsonl")
    for row in rows:
        assert "gold_answer" not in row
        assert "expected_answer" not in row
        assert "recommended_agent" not in row
        assert "researcher" not in row["task"]
        assert "analyst" not in row["task"]
