from engine.experiments.routing import RoutingCase, summarize_routing_rows
from engine.modules.agent_runtime import AgentRuntimeFactory
from engine.modules.execution import InferenceResult
from engine.modules.model_connections import ModelConnectionStore
from engine.modules.scheduling import ResourceAllocation, ResourceRequest, ResourceTier
from engine.modules.scheduling.production import AdvancedLearnedTaskGate


def _connections(tmp_path):
    store = ModelConnectionStore(tmp_path / "models")
    for tier in ("device", "edge", "cloud"):
        item = store.create({
            "name": tier, "provider": "compatible", "model_id": f"{tier}-model",
            "base_url": "http://models.example/v1", "tier": "general", "test_status": "succeeded",
        })
        store.assign_auto_tier(tier, item.id)
    return store


def test_strict_auto_does_not_fallback_to_another_tier(tmp_path, monkeypatch):
    store = _connections(tmp_path)
    runtime = AgentRuntimeFactory(model_connection_store=store)
    cloud = store.default_for_tier("cloud")
    calls = []

    class Scheduler:
        def acquire(self, request):
            return ResourceAllocation(ResourceTier.CLOUD, endpoint="cloud://model")

    monkeypatch.setattr(runtime, "_connection_scheduler", lambda *args, **kwargs: Scheduler())
    monkeypatch.setattr(runtime, "_run_pinned_model", lambda connection_id, prompt, system: calls.append(connection_id) or InferenceResult(text="", executor="test", endpoint="", success=False, error="failed"))

    result = runtime._run_auto_model(ResourceRequest(node="test", metadata={"strict_route": True}, state={"input": "x"}), "x", "")

    assert calls == [cloud.id]
    assert result.metadata["strict_route"] is True
    assert len(result.metadata["attempts"]) == 1


def test_bge_route_preserves_sensitive_profile():
    class Model:
        def predict_proba(self, text):
            return {0: 0.0, 1: 0.1, 2: 0.9}

    gate = AdvancedLearnedTaskGate(Model())
    profile = gate.evaluate(ResourceRequest(node="x", state={"input": "客户 api_key=secret"}))

    assert profile.metadata["route_tier"] == "cloud"
    assert profile.sensitivity.value == "secret"
    assert profile.requires_trusted_workspace is True


def test_routing_summary_reports_cost_and_strict_match_rate():
    summary = summarize_routing_rows([
        {"arm": "all_cloud", "success": True, "cloud_cost_cny": 2, "latency_ms": 10},
        {"arm": "strict_auto", "success": True, "cloud_cost_cny": 1, "latency_ms": 5, "selected_tier": "edge", "actual_tier": "edge"},
    ])

    assert summary["success_rate_delta_pp"] == 0.0
    assert summary["cloud_cost_reduction"] == 0.5
    assert summary["strict_route_match_rate"] == 1.0
    assert summary["paired_bootstrap_95ci"]["success_rate_delta_pp"] == [0.0, 0.0]
