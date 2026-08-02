import asyncio

from engine import GraphCheckpoint, GraphCheckpointStore, StateGraph


def test_graph_checkpoint_store_roundtrip(tmp_path):
    store = GraphCheckpointStore(tmp_path)
    saved = store.save(
        run_id="run-1",
        step=2,
        frontier=["b"],
        state={"value": 1},
        checkpoint_id="mid",
    )

    loaded = store.load("run-1", "mid")

    assert loaded.run_id == saved.run_id
    assert loaded.step == 2
    assert loaded.frontier == ["b"]
    assert loaded.state == {"value": 1}


def test_compiled_graph_writes_graph_checkpoints(tmp_path):
    graph = StateGraph()
    graph.add_node("a", lambda state: {"a": True})
    graph.add_node("b", lambda state: {"b": True})
    graph.set_entry_point("a")
    graph.add_edge("a", "b")
    compiled = graph.compile()
    store = GraphCheckpointStore(tmp_path)

    state = asyncio.run(
        compiled.ainvoke({"run_id": "run-checkpoint"}, checkpoint_store=store)
    )

    assert state["a"] is True
    assert state["b"] is True
    assert store.load("run-checkpoint", "final").status == "completed"
    assert store.path_for("run-checkpoint", "step_0000_before").exists()


def test_compiled_graph_resumes_from_checkpoint(tmp_path):
    graph = StateGraph()
    graph.add_node("a", lambda state: {"a": True})
    graph.add_node("b", lambda state: {"b": state["a"]})
    graph.set_entry_point("a")
    graph.add_edge("a", "b")
    compiled = graph.compile()
    checkpoint = GraphCheckpoint(
        run_id="resume-run",
        checkpoint_id="after-a",
        step=1,
        frontier=["b"],
        state={"run_id": "resume-run", "a": True},
    )

    state = asyncio.run(
        compiled.ainvoke(
            run_id="resume-run",
            checkpoint_store=GraphCheckpointStore(tmp_path),
            resume_from=checkpoint,
        )
    )

    assert state["a"] is True
    assert state["b"] is True
