from engine.experiments.communication_error_cascade import (
    METHODS, build_dataset, calculate_metrics, load_dataset, summarize,
)


def test_error_cascade_dataset_has_frozen_100_case_strata(tmp_path):
    path = tmp_path / "communication.jsonl"
    rows = build_dataset(path)
    assert len(rows) == 100
    assert {name: sum(row["subset"] == name for row in rows) for name in ("HS", "PS", "NR")} == {"HS": 70, "PS": 20, "NR": 10}
    assert load_dataset(path) == rows
    assert METHODS == ("baseline_full_history", "structured_gate")


def test_metrics_count_only_delivered_or_legacy_relay_errors(tmp_path):
    case = build_dataset(tmp_path / "communication.jsonl")[0]
    marker = f"[{case['error_id']}]"
    events = [
        {"type": "node_end", "node": "verifier_A", "update": {"messages": [{"content": marker}]}},
        {"type": "node_end", "node": "verifier_B", "update": {"messages": [{"content": marker}]}},
        {"type": "capsule_delivered", "recipient": "aggregator", "capsule": {"claim": marker}},
        {"type": "capsule_pruned", "recipient": "aggregator", "decision": {"reasons": ["redundant"], "components": {"redundancy": .99}}},
    ]
    baseline = calculate_metrics(case, "baseline_full_history", events, marker)
    structured = calculate_metrics(case, "structured_gate", events, marker)
    assert baseline["error_amplification_count"] == 2
    assert structured["error_amplification_count"] == 1
    assert structured["redundant_error_pruned"] == 1


def test_summary_excludes_invalid_rows_from_denominators():
    report = summarize([
        {"method": "baseline_full_history", "status": "valid", "subset": "HS", "metrics": {"error_amplification_count": 2, "error_reaches_writer": 1, "redundant_error_pruned": 0, "false_prune": 0}},
        {"method": "structured_gate", "status": "valid", "subset": "HS", "metrics": {"error_amplification_count": 1, "error_reaches_writer": 1, "redundant_error_pruned": 1, "false_prune": 0}},
        {"method": "structured_gate", "status": "invalid", "subset": "HS", "metrics": {}},
    ])
    assert report["valid_runs"] == 2
    assert report["methods"]["structured_gate"]["n"] == 1
    assert report["methods"]["structured_gate"]["repr_high_similarity"] == 1
