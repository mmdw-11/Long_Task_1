from engine import (
    AdaptiveResourceScheduler,
    EdgeHttpExecutor,
    ExecutorRegistry,
    InferenceRequest,
    InferenceResult,
    LocalEchoExecutor,
    MutableResourceMonitor,
    ResilientInferenceRunner,
    ResourceRequest,
    ResourceTier,
)


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


def test_edge_executor_falls_back_without_http_endpoint():
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
