"""核心图引擎的单元测试。运行：pytest tests/"""

import pytest

from engine import (
    END,
    START,
    GraphExecutionError,
    StateGraph,
    add_messages,
)


@pytest.mark.asyncio
async def test_sequential_pipeline():
    graph = StateGraph()
    graph.add_node("a", lambda s: {"value": (s.get("value", 0) + 1)})
    graph.add_node("b", lambda s: {"value": (s.get("value", 0) + 10)})
    graph.set_entry_point("a")
    graph.add_edge("a", "b")
    graph.set_finish_point("b")

    state = await graph.compile().ainvoke({"value": 0})
    assert state["value"] == 11


@pytest.mark.asyncio
async def test_conditional_and_loop():
    graph = StateGraph()
    graph.add_node("inc", lambda s: {"n": s.get("n", 0) + 1})
    graph.set_entry_point("inc")
    graph.add_conditional_edges(
        "inc",
        lambda s: "again" if s["n"] < 3 else "stop",
        path_map={"again": "inc", "stop": END},
    )
    state = await graph.compile().ainvoke({"n": 0})
    assert state["n"] == 3


@pytest.mark.asyncio
async def test_fanout_fanin_dedup():
    # a 扇出到 b、c，二者再汇聚到 d；d 只应执行一次。
    graph = StateGraph(schema={"log": add_messages})
    graph.add_node("a", lambda s: {"log": ["a"]})
    graph.add_node("b", lambda s: {"log": ["b"]})
    graph.add_node("c", lambda s: {"log": ["c"]})
    graph.add_node("d", lambda s: {"log": ["d"]})
    graph.set_entry_point("a")
    graph.add_edge("a", "b")
    graph.add_edge("a", "c")
    graph.add_edge("b", "d")
    graph.add_edge("c", "d")
    graph.set_finish_point("d")

    state = await graph.compile().ainvoke({})
    assert state["log"].count("d") == 1
    assert set(state["log"]) == {"a", "b", "c", "d"}


@pytest.mark.asyncio
async def test_async_node():
    async def anode(s):
        return {"ok": True}

    graph = StateGraph()
    graph.add_node("x", anode)
    graph.set_entry_point("x")
    graph.set_finish_point("x")
    state = await graph.compile().ainvoke({})
    assert state["ok"] is True


def test_missing_entry_raises():
    graph = StateGraph()
    graph.add_node("x", lambda s: None)
    with pytest.raises(GraphExecutionError):
        graph.compile()


@pytest.mark.asyncio
async def test_recursion_limit():
    graph = StateGraph()
    graph.add_node("loop", lambda s: {"n": s.get("n", 0) + 1})
    graph.set_entry_point("loop")
    graph.add_edge("loop", "loop")  # 无限循环
    with pytest.raises(GraphExecutionError):
        await graph.compile().ainvoke({}, recursion_limit=5)
