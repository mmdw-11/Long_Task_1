import pytest

from engine import (
    AdaptiveResourceScheduler,
    BinaryTextRouterModel,
    END,
    LearnedTaskGate,
    RealtimeRequirement,
    ResourceProfile,
    ResourceRequest,
    ResourceStatus,
    ResourceTier,
    SensitivityLevel,
    StateGraph,
    StaticResourceMonitor,
    TaskComplexity,
)
from engine.hooks import HookManager, RESOURCE_ALLOCATION_KEY
from engine.modules.scheduling import OpenAITaskGate
from engine.modules.scheduling.production import (
    AdvancedLearnedTaskGate,
    DEFAULT_ROUTER_BACKEND,
    load_production_gate,
    resolve_default_router_path,
    resolve_router_backend,
)


def test_high_complexity_public_task_prefers_cloud():
    scheduler = AdaptiveResourceScheduler()

    allocation = scheduler.acquire(
        _request(
            "deep_analysis",
            {
                "complexity": "high",
                "sensitivity": "public",
                "input": "大规模长文全量分析",
            },
        )
    )

    decision = allocation.metadata["decision"]
    assert allocation.tier == ResourceTier.CLOUD
    assert decision["reason"] == "cloud_selected_for_high_complexity"
    assert decision["model_split"][-1]["tier"] == "cloud"
    assert allocation.metadata["trace"]["gate_profile"]["complexity"] == "high"
    assert allocation.metadata["trace"]["heuristic_profile"]["complexity"] == "high"
    assert "cloud_for_high_complexity" in allocation.metadata["trace"]["policy_hits"]


def test_default_resources_read_device_edge_cloud_env(monkeypatch):
    monkeypatch.setenv("DEVICE_ENDPOINT", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("DEVICE_MODEL", "qwen2.5-0.5b-instruct")
    monkeypatch.setenv("EDGE_ENDPOINT", "http://127.0.0.1:8001/infer")
    monkeypatch.setenv("EDGE_MODEL", "qwen2.5-7b-instruct")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    monkeypatch.setenv("OPENAI_MODEL", "glm-5.2")

    resources = {profile.tier: profile for profile in AdaptiveResourceScheduler.default_resources()}

    assert resources[ResourceTier.DEVICE].endpoint == "http://127.0.0.1:11434/v1"
    assert resources[ResourceTier.DEVICE].models["small"] == "qwen2.5-0.5b-instruct"
    assert resources[ResourceTier.EDGE].endpoint == "http://127.0.0.1:8001/infer"
    assert resources[ResourceTier.EDGE].models["medium"] == "qwen2.5-7b-instruct"
    assert resources[ResourceTier.CLOUD].endpoint == "https://open.bigmodel.cn/api/paas/v4"
    assert resources[ResourceTier.CLOUD].models["large"] == "glm-5.2"


def test_openai_gate_maps_json_response_to_profile():
    gate = OpenAITaskGate()
    data = {
        "realtime": "hard",
        "sensitivity": "secret",
        "complexity": "high",
        "task_type": "recovery",
        "requires_trusted_workspace": True,
        "reason": "包含密钥且需要紧急处理",
    }

    profile = gate._profile_from_data(data, ResourceRequest(node="triage"))

    assert profile.realtime.value == "hard"
    assert profile.sensitivity.value == "secret"
    assert profile.complexity.value == "high"
    assert profile.task_type == "recovery"
    assert profile.requires_trusted_workspace is True
    assert profile.metadata["gate"] == "openai"


def test_trace_records_gate_differences():
    class FixedGate:
        def evaluate(self, request):
            from engine import TaskProfile

            return TaskProfile(
                realtime=RealtimeRequirement.HARD,
                sensitivity=SensitivityLevel.SECRET,
                complexity=TaskComplexity.HIGH,
                task_type="recovery",
                requires_trusted_workspace=True,
            )

    scheduler = AdaptiveResourceScheduler(gate=FixedGate())
    allocation = scheduler.acquire(
        _request(
            "triage",
            {
                "input": "普通任务",
            },
        )
    )

    trace = allocation.metadata["trace"]
    assert trace["gate_profile"]["sensitivity"] == "secret"
    assert trace["heuristic_profile"]["sensitivity"] == "internal"
    assert "gate_sensitivity_diff" in trace["policy_hits"]
    assert "gate_complexity_diff" in trace["policy_hits"]


def test_scheduler_honors_learned_route_tier_metadata():
    class EdgeGate:
        def evaluate(self, request):
            from engine import TaskProfile

            return TaskProfile(
                complexity=TaskComplexity.HIGH,
                metadata={"router": "learned", "route_tier": "edge"},
            )

    scheduler = AdaptiveResourceScheduler(gate=EdgeGate())
    allocation = scheduler.acquire(ResourceRequest(node="route", state={"input": "medium edge task"}))

    assert allocation.tier == ResourceTier.EDGE
    assert allocation.metadata["decision"]["model_split"][-1]["tier"] == "edge"


def test_sensitive_high_complexity_stays_in_trusted_workspace_without_review():
    scheduler = AdaptiveResourceScheduler()

    allocation = scheduler.acquire(
        _request(
            "private_analysis",
            {
                "complexity": "high",
                "sensitivity": "confidential",
                "input": "客户邮件与内部数据的复杂分析",
            },
        )
    )

    decision = allocation.metadata["decision"]
    assert allocation.tier == ResourceTier.EDGE
    assert decision["reason"] == "sensitive_task_kept_in_trusted_workspace"
    assert decision["requires_human_review"] is False


def test_sensitive_cloud_transfer_requires_review_when_no_trusted_tier_available():
    scheduler = AdaptiveResourceScheduler(
        resources=[
            ResourceProfile(
                ResourceTier.CLOUD,
                endpoint="cloud://only",
                trusted=False,
                max_complexity=TaskComplexity.EXTREME,
            )
        ]
    )

    allocation = scheduler.acquire(
        _request(
            "cloud_only",
            {
                "complexity": "high",
                "sensitivity": "secret",
                "input": "api_key 和私钥分析",
            },
        )
    )

    decision = allocation.metadata["decision"]
    assert allocation.tier == ResourceTier.CLOUD
    assert decision["requires_human_review"] is True
    assert decision["reason"] == "external_sensitive_transfer_requires_human_review"


def test_cloud_rate_limited_high_complexity_falls_back_to_edge():
    scheduler = AdaptiveResourceScheduler(
        monitor=StaticResourceMonitor([
            ResourceStatus(ResourceTier.CLOUD, available=True, rate_limited=True),
        ])
    )

    allocation = scheduler.acquire(
        _request(
            "deep_public",
            {
                "complexity": "high",
                "sensitivity": "public",
                "input": "公开长文复杂分析",
            },
        )
    )

    assert allocation.tier == ResourceTier.EDGE
    status = allocation.metadata["resource_status"]["cloud"]
    assert status["rate_limited"] is True
    assert allocation.metadata["trace"]["metadata"]["resource_status"]["cloud"]["rate_limited"] is True


def test_sensitive_high_complexity_uses_device_when_edge_unavailable():
    scheduler = AdaptiveResourceScheduler(
        monitor=StaticResourceMonitor([
            ResourceStatus(ResourceTier.EDGE, available=False),
            ResourceStatus(ResourceTier.CLOUD, available=True),
        ])
    )

    allocation = scheduler.acquire(
        _request(
            "private_work",
            {
                "complexity": "high",
                "sensitivity": "confidential",
                "input": "客户内部邮件复杂分析",
            },
        )
    )

    assert allocation.tier == ResourceTier.DEVICE
    assert allocation.metadata["decision"]["reason"] == "sensitive_task_kept_in_trusted_workspace"


@pytest.mark.asyncio
async def test_hook_injects_resource_allocation_into_node_state():
    graph = StateGraph()

    def worker(state):
        return {"allocation": state[RESOURCE_ALLOCATION_KEY]}

    graph.add_node(
        "worker",
        worker,
        metadata={
            "complexity": "high",
            "sensitivity": "public",
            "realtime": "batch",
        },
    )
    graph.set_entry_point("worker")
    graph.add_edge("worker", END)

    compiled = graph.compile()
    compiled.hooks = HookManager(scheduler=AdaptiveResourceScheduler())
    state = await compiled.ainvoke({"input": "大规模复杂推理"})

    assert state["allocation"]["tier"] == ResourceTier.CLOUD.value
    assert state["allocation"]["metadata"]["decision"]["profile"]["complexity"] == "high"


def _request(node, values):
    from engine import ResourceRequest

    metadata = {
        key: value
        for key, value in values.items()
        if key in {"complexity", "sensitivity", "realtime"}
    }
    state = {
        key: value
        for key, value in values.items()
        if key not in {"complexity", "sensitivity", "realtime"}
    }
    return ResourceRequest(node=node, metadata=metadata, state=state)


def test_scheduler_loads_trained_router_from_router_path(tmp_path):
    from engine import RouteDataset

    dataset = RouteDataset()
    dataset.add("创建明天下午三点的项目会日程", 0)
    dataset.add("分析复杂代码架构风险并给出重构方案", 1)
    dataset.add("总结客户邮件的要点", 0)
    dataset.add("对长文档做全量风险评估", 1)
    model = BinaryTextRouterModel().fit(dataset.examples)
    model_path = model.save(tmp_path / "router.json")

    scheduler = AdaptiveResourceScheduler(router_path=str(model_path))
    assert isinstance(scheduler.gate, LearnedTaskGate)

    allocation = scheduler.acquire(
        ResourceRequest(node="route", state={"input": "分析复杂代码架构风险并给出重构方案"})
    )
    assert allocation.metadata["decision"]["profile"]["metadata"]["router"] == "learned"


def test_production_router_defaults_to_bge_m3_backend():
    assert DEFAULT_ROUTER_BACKEND == "bge_m3"
    assert resolve_router_backend() == "bge_m3"
    path = resolve_default_router_path()
    assert path is not None
    assert path.name == "balanced_bge_m3"


def test_scheduler_can_auto_enable_production_router_from_env(monkeypatch):
    monkeypatch.setenv("USE_PRODUCTION_ROUTER", "1")
    monkeypatch.setenv("ROUTER_MODEL_BACKEND", "bge_m3")
    scheduler = AdaptiveResourceScheduler()
    assert isinstance(scheduler.gate, AdvancedLearnedTaskGate)


def test_production_router_can_switch_to_bert_full():
    gate = load_production_gate(backend="bert_full")
    assert isinstance(gate, AdvancedLearnedTaskGate)
    profile = gate.evaluate(ResourceRequest(node="route", state={"input": "请分析复杂代码架构风险并给出重构方案"}))
    assert profile.metadata["router_backend"] == "bert_full"
