from engine import (
    AdaptiveResourceScheduler,
    EdgeHttpExecutor,
    ExecutorRegistry,
    InferenceRequest,
    InferenceResult,
    LocalEchoExecutor,
    LocalModelExecutor,
    MutableResourceMonitor,
    ResilientInferenceRunner,
    ResourceRequest,
    ResourceTier,
)


class FakeMulticlassClassifier:
    classes_ = [0, 1, 2]

    def predict_proba(self, vector):
        return [[0.1, 0.7, 0.2]]


def test_local_echo_executor_returns_observable_result():
    executor = LocalEchoExecutor()
    result = executor.run(
        InferenceRequest(
            prompt="hello local",
            allocation={"tier": "device", "endpoint": "local"},
        )
    )

    assert result.executor == "LocalEchoExecutor"
    assert result.model == "local-echo"
    assert "hello local" in result.text


def test_local_model_executor_falls_back_without_device_config(monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    monkeypatch.delenv("DEVICE_BASE_URL", raising=False)
    monkeypatch.delenv("DEVICE_MODEL", raising=False)
    executor = LocalModelExecutor()

    result = executor.run(
        InferenceRequest(
            prompt="hello device",
            allocation={"tier": "device", "endpoint": "local"},
        )
    )

    assert result.executor == "LocalEchoExecutor"
    assert result.metadata["fallback_reason"] == "missing DEVICE_BASE_URL or DEVICE_MODEL"
    assert "hello device" in result.text


def test_registry_routes_by_resource_tier():
    class FakeExecutor:
        def run(self, request):
            from engine import InferenceResult

            return InferenceResult(
                text="cloud ok",
                executor="fake",
                endpoint=request.allocation["endpoint"],
                model="fake-model",
            )

    registry = ExecutorRegistry({ResourceTier.CLOUD: FakeExecutor(), ResourceTier.DEVICE: LocalEchoExecutor()})
    result = registry.run(
        InferenceRequest(
            prompt="public deep task",
            allocation={"tier": "cloud", "endpoint": "cloud://default"},
        )
    )

    assert result.text == "cloud ok"
    assert result.endpoint == "cloud://default"


def test_edge_executor_falls_back_without_http_endpoint(monkeypatch):
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    executor = EdgeHttpExecutor()
    result = executor.run(
        InferenceRequest(
            prompt="edge task",
            allocation={"tier": "edge", "endpoint": "edge://default"},
        )
    )

    assert result.metadata["edge_fallback"] is True
    assert result.endpoint == "edge://default"
    assert "edge task" in result.text


def test_resilient_runner_reschedules_after_retryable_cloud_failure():
    class CloudFailsOnce:
        def __init__(self):
            self.calls = 0

        def run(self, request):
            self.calls += 1
            if self.calls == 1:
                return InferenceResult(
                    text="",
                    executor="fake-cloud",
                    endpoint=request.allocation["endpoint"],
                    success=False,
                    error="429 rate limit",
                    retryable=True,
                )
            return InferenceResult(
                text="cloud recovered",
                executor="fake-cloud",
                endpoint=request.allocation["endpoint"],
            )

    class EdgeOk:
        def run(self, request):
            return InferenceResult(
                text="edge ok",
                executor="fake-edge",
                endpoint=request.allocation["endpoint"],
            )

    monitor = MutableResourceMonitor()
    scheduler = AdaptiveResourceScheduler(monitor=monitor)
    runner = ResilientInferenceRunner(
        scheduler=scheduler,
        registry=ExecutorRegistry(
            {
                ResourceTier.CLOUD: CloudFailsOnce(),
                ResourceTier.EDGE: EdgeOk(),
                ResourceTier.DEVICE: LocalEchoExecutor(),
            }
        ),
        max_attempts=2,
    )

    result = runner.run(
        resource_request=ResourceRequest(
            node="deep_public",
            state={"input": "对公开发布的 3 万字行业报告做深度总结和多维对比"},
        ),
        prompt="对公开发布的 3 万字行业报告做深度总结和多维对比",
    )

    assert result.success is True
    assert result.text == "edge ok"
    attempts = result.metadata["attempts"]
    assert attempts[0]["allocation"]["tier"] == "cloud"
    assert attempts[1]["allocation"]["tier"] == "edge"
    assert monitor.snapshot(scheduler.resources)[ResourceTier.CLOUD].rate_limited is True


def test_embedding_router_maps_multiclass_classifier_probabilities(tmp_path, monkeypatch):
    import json
    import pickle

    from engine.modules.scheduling.advanced_training import EmbeddingClassifierRouter

    class FakeEncoder:
        def encode(self, texts, normalize_embeddings=True):
            return [[0.0, 1.0] for _ in texts]

    class FakeSentenceTransformer:
        def __init__(self, model_name, local_files_only=True):
            self.model_name = model_name

        def encode(self, texts, normalize_embeddings=True):
            return FakeEncoder().encode(texts, normalize_embeddings=normalize_embeddings)

    artifact_dir = tmp_path / "bge"
    artifact_dir.mkdir()
    (artifact_dir / "summary.json").write_text(json.dumps({"model_name": "fake"}), encoding="utf-8")
    with (artifact_dir / "classifier.pkl").open("wb") as fh:
        pickle.dump(FakeMulticlassClassifier(), fh)

    import types
    import sys

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    router = EmbeddingClassifierRouter(artifact_dir)

    assert router.predict_proba("hello") == {0: 0.1, 1: 0.7, 2: 0.2}
    assert router.predict("hello") == 1
