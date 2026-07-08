"""Orchestrator（前端接口层）的单元测试。运行：pytest tests/"""

import pytest

from engine import Orchestrator


def test_create_and_sub_agents():
    orch = Orchestrator()
    p = orch.create_agent("parent")
    c1 = orch.add_sub_agent(p, name="child1")
    c2 = orch.add_sub_agent(p, name="child2")

    parent = orch.get_agent(p)
    assert parent.children == [c1, c2]
    # add_sub_agent 默认自动连边 parent->child
    data = orch.to_dict()
    edges = {(c["source"], c["target"]) for c in data["connections"]}
    assert (p, c1) in edges and (p, c2) in edges


def test_duplicate_name_rejected():
    orch = Orchestrator()
    orch.create_agent("dup")
    with pytest.raises(ValueError):
        orch.create_agent("dup")


def test_connect_and_export_import_roundtrip():
    orch = Orchestrator()
    a = orch.create_agent("a")
    b = orch.create_agent("b")
    orch.connect(a, b)
    orch.set_entry(a)

    data = orch.to_dict()
    orch2 = Orchestrator.from_dict(data)
    assert orch2.to_dict() == data


@pytest.mark.asyncio
async def test_build_and_run_echo():
    orch = Orchestrator()
    a = orch.create_agent("a")
    b = orch.add_sub_agent(a, name="b")
    orch.set_entry(a)

    compiled = orch.build_graph()
    state = await compiled.ainvoke({"input": "hi"})
    # echo 工厂会累积每个节点的消息
    agents = [m["agent"] for m in state["messages"]]
    assert agents == ["a", "b"]


@pytest.mark.asyncio
async def test_conditional_connection():
    orch = Orchestrator()
    router = orch.create_agent("router")
    left = orch.create_agent("left")
    right = orch.create_agent("right")
    orch.set_entry(router)
    orch.connect_conditional(router, "branch", {"L": left, "R": right})

    # 注入一个把 branch 设为 L 的入口状态
    compiled = orch.build_graph()
    state = await compiled.ainvoke({"input": "x", "branch": "L"})
    names = [m["agent"] for m in state["messages"]]
    assert "left" in names and "right" not in names
