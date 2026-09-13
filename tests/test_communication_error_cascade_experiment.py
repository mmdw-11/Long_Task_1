from engine.experiments.communication_error_cascade import METHODS, SUBSETS, build_dataset, calculate_metrics, load_dataset, summarize


def test_dataset_has_variable_topologies_and_frozen_strata(tmp_path):
    path = tmp_path / "communication.jsonl"; rows = build_dataset(path)
    assert len(rows) == 100
    assert {name: sum(row["subset"] == name for row in rows) for name in SUBSETS} == SUBSETS
    assert any(row["topology"] == "multihop" for row in rows)
    assert any(row["subset"] == "independent" and len(row["required_fact_keys"]) > 2 for row in rows)
    assert load_dataset(path) == rows
    assert METHODS == ("baseline_full_history", "structured_gate")


def test_metrics_distinguish_duplicate_suppression_from_independent_fact_loss(tmp_path):
    duplicate = next(row for row in build_dataset(tmp_path / "communication.jsonl") if row["subset"] == "fanin")
    items = duplicate["relay_payloads"][:2]
    events = [
        {"type": "capsule_delivered", "recipient": "aggregator", "capsule": item}
        for item in items[:1]
    ] + [{"type": "capsule_pruned", "recipient": "aggregator", "decision": {"reasons": ["redundant"]}}]
    metrics = calculate_metrics(duplicate, "structured_gate", events, "{}", {"relay_1", "relay_2"})
    assert metrics["aggregator_input_items"] == 1
    assert metrics["redundant_pruned"] == 1

    independent = next(row for row in build_dataset(tmp_path / "communication2.jsonl") if row["subset"] == "independent")
    events = [{"type": "capsule_delivered", "recipient": "aggregator", "capsule": independent["relay_payloads"][0]}]
    metrics = calculate_metrics(independent, "structured_gate", events, "{}", {"relay_1", "relay_2"})
    assert metrics["critical_fact_recall"] < 1
    assert metrics["false_prune"] == 1


def test_summary_excludes_invalid_rows_and_uses_new_metrics():
    metrics = {"aggregator_input_items": 1, "error_amplification_factor": 1, "relay_token_proxy": 2, "aggregator_prompt_tokens": 10, "redundant_pruned": 1, "critical_fact_recall": 1, "false_prune": 0, "compression_events": 1, "writer_accuracy": 1, "false_claim_adopted": 0, "writer_json_valid": 1}
    report = summarize([
        {"method": "baseline_full_history", "status": "valid", "subset": "fanin", "metrics": metrics},
        {"method": "structured_gate", "status": "valid", "subset": "fanin", "metrics": metrics},
        {"method": "structured_gate", "status": "invalid", "subset": "fanin", "metrics": {}},
    ])
    assert report["valid_runs"] == 2
    assert report["methods"]["structured_gate"]["n"] == 1
    assert report["methods"]["structured_gate"]["duplicate_prune_rate"] == 1
