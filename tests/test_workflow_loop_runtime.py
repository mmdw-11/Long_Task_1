import pytest

from engine.modules.product_ops import ToolCatalogStore
from engine.modules.workflow_runtime import WorkflowNodeRuntimeFactory
from engine.orchestrator import AgentSpec


@pytest.mark.asyncio
async def test_count_loop_runs_exact_configured_iterations(tmp_path):
    factory = WorkflowNodeRuntimeFactory(ToolCatalogStore(tmp_path / "tools"))
    node = factory(AgentSpec(id="loop-1", name="循环", config={"node_kind": "loop", "max_iterations": 3}))

    first = await node.invoke({})
    second = await node.invoke(first)
    third = await node.invoke(second)
    finished = await node.invoke(third)

    assert [first["route_loop-1"], second["route_loop-1"], third["route_loop-1"]] == ["continue"] * 3
    assert finished["route_loop-1"] == "done"


@pytest.mark.asyncio
async def test_array_loop_exposes_item_and_index_and_honors_limit(tmp_path):
    factory = WorkflowNodeRuntimeFactory(ToolCatalogStore(tmp_path / "tools"))
    node = factory(AgentSpec(id="loop-2", name="数组循环", config={
        "node_kind": "loop", "loop_type": "array", "items_path": "items",
        "item_field": "current", "index_field": "index", "max_iterations": 2,
    }))

    first = await node.invoke({"items": ["a", "b", "c"]})
    second = await node.invoke({"items": ["a", "b", "c"], **first})
    finished = await node.invoke({"items": ["a", "b", "c"], **second})

    assert (first["current"], first["index"]) == ("a", 0)
    assert (second["current"], second["index"]) == ("b", 1)
    assert finished["route_loop-2"] == "done"
