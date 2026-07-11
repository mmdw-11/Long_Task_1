import pytest

from engine import (
    AdaptiveResourceScheduler,
    END,
    RealtimeRequirement,
    ResourceProfile,
    ResourceTier,
    SensitivityLevel,
    StateGraph,
    TaskComplexity,
)
from engine.hooks import HookManager, RESOURCE_ALLOCATION_KEY


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
