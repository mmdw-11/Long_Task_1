import pytest

from engine.modules.product_ops import ToolCatalogStore
from engine.modules.workflow_runtime import WorkflowNodeRuntimeFactory
from engine.modules.workflow_scripts import run_workflow_script
from engine.orchestrator import AgentSpec


def test_python_script_returns_object():
    assert run_workflow_script("python", "def main(params):\n    return {'total': params['a'] + params['b']}", {"a": 2, "b": 3}) == {"total": 5}


def test_python_script_preserves_cjk_multiline_json_input():
    task = '总目标：调研事实\n验收标准：["至少两个真实来源"]\n完成当前 TODO。'
    assert run_workflow_script("python", "def main(params):\n    return {'echo': params['input']}", {"input": task}) == {"echo": task}


def test_javascript_script_returns_object():
    assert run_workflow_script("javascript", "function main(params) { return {total: params.a + params.b}; }", {"a": 2, "b": 4}) == {"total": 6}


@pytest.mark.asyncio
async def test_condition_groups_are_or_between_groups_and_and_inside(tmp_path):
    factory = WorkflowNodeRuntimeFactory(ToolCatalogStore(tmp_path / "tools"))
    node = factory(AgentSpec(id="condition-1", name="判断", config={
        "node_kind": "condition", "default_route": "other", "branches": [{"route": "matched", "groups": [
            {"conditions": [{"field": "score", "operator": "greater_than", "value": 90}, {"field": "enabled", "operator": "equals", "value": True}]},
            {"conditions": [{"field": "role", "operator": "equals", "value": "admin"}]},
        ]}],
    }))
    assert (await node.invoke({"score": 70, "role": "admin"}))["route_condition-1"] == "matched"
    assert (await node.invoke({"score": 95, "enabled": False}))["route_condition-1"] == "other"


@pytest.mark.asyncio
async def test_assignment_and_multi_array_batch(tmp_path):
    factory = WorkflowNodeRuntimeFactory(ToolCatalogStore(tmp_path / "tools"))
    assign = factory(AgentSpec(id="assign-1", name="赋值", config={"node_kind": "assign", "assignments": [{"target": "items", "operation": "append", "source": "input"}]}))
    assert (await assign.invoke({"input": "b", "items": ["a"]}))["items"] == ["a", "b"]
    batch = factory(AgentSpec(id="batch-1", name="批处理", config={"node_kind": "batch", "input_arrays": [{"path": "a", "item_field": "left"}, {"path": "b", "item_field": "right"}], "max_items": 10}))
    first = await batch.invoke({"a": [1, 2, 3], "b": ["x", "y"]})
    second = await batch.invoke({"a": [1, 2, 3], "b": ["x", "y"], **first})
    assert (first["left"], first["right"], first["batch_index"]) == (1, "x", 0)
    assert (second["left"], second["right"], second["batch_index"]) == (2, "y", 1)
    finished = await batch.invoke({"a": [1, 2, 3], "b": ["x", "y"], **second, "__batch_results_batch-1": ["one", "two"]})
    assert finished["route_batch-1"] == "done"
    assert finished["batch_output"] == ["one", "two"]


@pytest.mark.asyncio
async def test_batch_runs_child_graph_for_each_item_and_keeps_order(tmp_path):
    graph = {"agents": [
        {"id": "bs", "name": "批处理开始", "config": {"node_kind": "batch_start", "parent_id": "batch"}},
        {"id": "script", "name": "转换", "config": {"node_kind": "script", "language": "python", "inputs": [{"name": "value", "path": "item"}], "code": "def main(params):\n    return {'value': params['value'] * 2}", "output_field": "value"}},
        {"id": "be", "name": "批处理结束", "config": {"node_kind": "batch_end", "parent_id": "batch"}},
    ], "connections": [
        {"source": "bs", "target": "script", "conditional": False},
        {"source": "script", "target": "be", "conditional": False},
    ]}
    factory = WorkflowNodeRuntimeFactory(ToolCatalogStore(tmp_path / "tools"), graph=graph)
    node = factory(AgentSpec(id="batch", name="批处理", children=["bs", "script", "be"], config={"node_kind": "batch", "input_arrays": [{"path": "items", "item_field": "item"}], "concurrency": 2}))
    output = await node.invoke({"items": [3, 1, 2]})
    assert [item["index"] for item in output["batch_output"]] == [0, 1, 2]
    assert [item["status"] for item in output["batch_output"]] == ["succeeded"] * 3, output["batch_output"]
    assert [item["output"]["value"] for item in output["batch_output"]] == [6, 2, 4]
    assert all("events" not in item for item in output["batch_output"])
    assert output["route_batch"] == "done"


@pytest.mark.asyncio
async def test_loop_runs_its_child_canvas_without_top_level_boundary_edges(tmp_path):
    graph = {"agents": [
        {"id": "ls", "name": "循环开始", "config": {"node_kind": "loop_start", "parent_id": "loop"}},
        {"id": "script", "name": "转换", "config": {"node_kind": "script", "language": "python", "inputs": [{"name": "loop_index", "path": "loop_index"}], "code": "def main(params):\n    return {'input': params['loop_index'] + 1}", "output_field": "input"}},
        {"id": "le", "name": "迭代结束", "config": {"node_kind": "loop_end", "parent_id": "loop"}},
    ], "connections": [
        {"source": "ls", "target": "script", "conditional": False},
        {"source": "script", "target": "le", "conditional": False},
    ]}
    factory = WorkflowNodeRuntimeFactory(ToolCatalogStore(tmp_path / "tools"), graph=graph)
    node = factory(AgentSpec(id="loop", name="循环", children=["ls", "script", "le"], config={"node_kind": "loop", "max_iterations": 3, "output_field": "loop_output"}))

    output = await node.invoke({"input": "ignored"})

    assert [item["input"] for item in output["loop_output"]] == [1, 2, 3]
    assert output["route_loop"] == "done"
