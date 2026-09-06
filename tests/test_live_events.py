"""Streaming contract tests, entirely offline and with deterministic gates."""
import asyncio
import json
import threading
from types import SimpleNamespace as NS

import pytest

from engine.modules.live_events import chat_completion, emit, live_events, sink, VisibleText
from engine.modules.product_ops import ToolCatalogStore
from engine.modules.tool_runtime import ToolRuntime


def chunk(text=None, calls=None, reasoning=None, finish=None):
    return NS(choices=[NS(index=0, finish_reason=finish, delta=NS(content=text, tool_calls=calls, reasoning_content=reasoning))])


def capture(create):
    events = []
    token = sink.set(events.append)
    try:
        response = chat_completion(create, model="test", messages=[])
        return response, events
    finally:
        sink.reset(token)


def test_fragments_and_reasoning_are_separated():
    def create(**kwargs):
        assert kwargs["stream"] is True
        return iter([
            chunk(reasoning="PRIVATE THOUGHT"), chunk("<thi"), chunk("nk>hidden</think>你好"),
            chunk(calls=[NS(index=0, id="call-1", function=NS(name="read", arguments='{"path":'))]),
            chunk(calls=[NS(index=0, id=None, function=NS(name=None, arguments='"a.txt"}'))]),
            chunk(finish="tool_calls"),
        ])
    response, events = capture(create)
    message = response.choices[0].message
    assert message.content == "你好"
    assert message.reasoning_content == "PRIVATE THOUGHT"
    assert json.loads(message.tool_calls[0].function.arguments) == {"path": "a.txt"}
    assert message.tool_calls[0].id == "call-1"
    assert "PRIVATE" not in json.dumps(events)
    assert "hidden" not in json.dumps(events)
    assert "".join(e["delta"] for e in events if e["type"] == "answer_delta") == "你好"


def test_partial_failure_is_not_retried_and_keeps_text():
    calls, events = [], []
    def create(**kwargs):
        calls.append(kwargs)
        def stream():
            yield chunk("已收到的正文")
            raise ConnectionError("provider disconnected")
        return stream()
    token = sink.set(events.append)
    try:
        with pytest.raises(ConnectionError):
            chat_completion(create, model="test")
    finally:
        sink.reset(token)
    assert len(calls) == 1
    assert events[-1]["type"] == "model_failed"
    assert any(e.get("delta") == "已收到的正文" for e in events)


def test_eof_without_finish_is_failure():
    with pytest.raises(RuntimeError, match="未完整结束"):
        capture(lambda **kwargs: iter([chunk("partial")]))


def test_explicit_unsupported_stream_can_fall_back_once():
    calls = []
    class Unsupported(Exception):
        status_code = 400
    def create(**kwargs):
        calls.append(kwargs["stream"])
        if kwargs["stream"]:
            raise Unsupported("stream is not supported")
        return NS(choices=[NS(message=NS(content="<think>hidden</think>answer"))])
    response, events = capture(create)
    assert calls == [True, False]
    assert response.choices[0].message.content == "answer"
    assert any(e["type"] == "answer_mode" for e in events)


def test_auth_failure_never_falls_back():
    class Unauthorized(Exception):
        status_code = 401
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        raise Unauthorized("invalid credentials")
    with pytest.raises(Unauthorized):
        capture(create)
    assert len(calls) == 1


def test_no_sink_preserves_legacy_protocol():
    expected = object()
    def create(**kwargs):
        assert "stream" not in kwargs
        return expected
    assert chat_completion(create, model="legacy") is expected


def test_live_events_arrive_before_worker_finishes_and_stay_isolated():
    async def scenario():
        release = threading.Event()
        def worker(label):
            emit("tool_started", message=label)
            assert release.wait(3)
            emit("tool_finished", message=label)
        async def graph(label):
            yield {"type": "node_start"}
            await asyncio.to_thread(worker, label)
            yield {"type": "node_end"}
        streams = [live_events(graph(label)) for label in ("A", "B")]
        try:
            for stream in streams:
                assert (await anext(stream))["type"] == "node_start"
            first = await asyncio.wait_for(asyncio.gather(*(anext(s) for s in streams)), 2)
            assert [e["message"] for e in first] == ["A", "B"]
            assert not release.is_set()
            release.set()
            for stream, label in zip(streams, ("A", "B")):
                remaining = [e async for e in stream]
                assert remaining[0]["message"] == label
                assert remaining[-1]["type"] == "node_end"
        finally:
            release.set()
            for stream in streams:
                await stream.aclose()
    asyncio.run(scenario())


def test_sync_hook_event_not_lost_at_graph_completion():
    async def scenario():
        async def graph():
            emit("skill_applied", skill_id="s1")
            yield {"type": "final"}
        return [e async for e in live_events(graph())]
    assert [e["type"] for e in asyncio.run(scenario())] == ["skill_applied", "final"]


def test_tool_progress_does_not_leak_arguments_or_results(tmp_path):
    catalog = ToolCatalogStore(tmp_path / "tools")
    tool = catalog.create(name="echo", display_name="echo", metadata={"adapter": "echo", "risk": "low"})
    events = []
    token = sink.set(events.append)
    try:
        result = ToolRuntime(catalog).execute(tool, "private payload", arguments={"token": "secret"})
    finally:
        sink.reset(token)
    assert result.status == "succeeded"
    assert [e["type"] for e in events] == ["tool_started", "tool_finished"]
    assert "secret" not in json.dumps(events) and "private payload" not in json.dumps(events)


def test_pending_approval_does_not_report_tool_as_executed(tmp_path):
    catalog = ToolCatalogStore(tmp_path / "tools")
    tool = catalog.create(name="echo", display_name="echo", metadata={"adapter": "echo", "risk": "high"})
    events = []
    token = sink.set(events.append)
    try:
        result = ToolRuntime(catalog).execute(tool, "do not execute")
    finally:
        sink.reset(token)
    assert result.status == "approval_required"
    assert [e["type"] for e in events] == ["tool_finished"]
    assert events[0]["status"] == "approval_required"


@pytest.mark.parametrize("source,expected", [("plain < text", "plain < text"), ("<think>secret</think>visible", "visible"), ("<analysis>secret", ""), ("<think>secret</think><think>again</think>end", "end")])
def test_reasoning_tag_filter_handles_single_character_chunks(source, expected):
    parser = VisibleText()
    assert "".join(parser.feed(c) for c in source) + parser.feed("", final=True) == expected


def test_cancellation_stops_follow_up_work():
    async def scenario():
        release, started = threading.Event(), threading.Event()
        cancel = False
        operations = []
        def worker():
            emit("tool_started")
            started.set()
            release.wait(2)
            emit("tool_finished")
            from engine.modules.live_events import checkpoint
            checkpoint()
            operations.append("must not run")
        async def graph():
            await asyncio.to_thread(worker)
            yield {"type": "final"}
        stream = live_events(graph(), cancel_check=lambda: cancel)
        try:
            assert (await anext(stream))["type"] == "tool_started"
            cancel = True
            await asyncio.sleep(.15)
            release.set()
            assert (await asyncio.wait_for(anext(stream), 1))["type"] == "tool_finished"
            await stream.aclose()
        finally:
            release.set()
        await asyncio.sleep(.05)
        assert not operations
    asyncio.run(scenario())


def test_run_api_persists_deltas_and_sse_resumes_after_sequence(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from engine.server.app import create_app
    from engine.modules.workflows import RunStore, WorkflowStore
    from engine.node import Node
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    monkeypatch.chdir(tmp_path)
    def factory(spec):
        async def execute(state):
            response = await asyncio.to_thread(chat_completion, lambda **kwargs: iter([chunk("hello"), chunk(" world", finish="stop")]), model="test")
            text = response.choices[0].message.content
            return {"input": text, "messages": [{"agent": spec.name, "content": text}]}
        return Node(spec.name, execute)
    client = TestClient(create_app(node_factory=factory, run_store=RunStore(tmp_path / "runs"), workflow_store=WorkflowStore(tmp_path / "workflows")))
    agent = client.post("/api/agents", json={"name": "streaming-node"}).json()["id"]
    client.post("/api/graph/entry", json={"agent_id": agent})
    run = client.post("/api/runs", json={"input": {"input": "hello"}}).json()
    done = client.get(f"/api/runs/{run['id']}").json()
    assert done["status"] == "succeeded", done.get("error")
    events = done["events"]
    delta = next(e for e in events if e["type"] == "answer_delta")
    assert delta["delta"] == "hello world"
    assert delta["node"] == "streaming-node"
    assert delta["sequence"] < next(e["sequence"] for e in events if e["type"] == "node_end")
    assert [e["sequence"] for e in events] == list(range(1, len(events)+1))
    response = client.get(f"/api/runs/{run['id']}/events?after=0", headers={"Last-Event-ID": str(delta["sequence"])})
    assert response.status_code == 200
    remaining = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert all(e.get("sequence", len(events)+1) > delta["sequence"] for e in remaining)
    assert remaining[-1]["type"] == "run_completed"


def test_native_streaming_tool_loop_and_approval_resume_same_run(tmp_path, monkeypatch):
    import openai
    from fastapi.testclient import TestClient
    from engine.server.app import create_app
    from engine.modules.workflows import RunStore, WorkflowStore
    from engine.modules.model_connections import ModelConnectionStore
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    monkeypatch.chdir(tmp_path)
    requests, executed, approval_statuses = [], [], []
    original = ToolRuntime._execute
    def tracked_execute(self, tool, task_text, **kwargs):
        if kwargs.get("bypass_approval"):
            executed.append(tool.id)
            approval_statuses.append(run_store.list()[0].status)
        return original(self, tool, task_text, **kwargs)
    monkeypatch.setattr(ToolRuntime, "_execute", tracked_execute)
    def create(**kwargs):
        assert kwargs["stream"] is True
        requests.append(kwargs)
        if len(requests) == 1:
            name = kwargs["tools"][0]["function"]["name"]
            return iter([
                chunk(reasoning="private provider reasoning"),
                chunk(calls=[NS(index=0, id="call-1", function=NS(name=name, arguments='{"text":'))]),
                chunk(calls=[NS(index=0, id=None, function=NS(name=None, arguments='"hello"}'))]),
                chunk(finish="tool_calls"),
            ])
        return iter([chunk("已经完成"), chunk("获批的操作。", finish="stop")])
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: NS(chat=NS(completions=NS(create=create))))
    models = ModelConnectionStore(tmp_path / "models")
    connection = models.create({"name":"test","provider":"openai-compatible","model_id":"pinned-test","base_url":"https://model.invalid/v1","api_key":"test-only","test_status":"succeeded"})
    tools = ToolCatalogStore(tmp_path / "tools")
    tool = tools.create(name="echo", display_name="测试工具", description="测试工具", metadata={"adapter":"echo","risk":"high"})
    run_store = RunStore(tmp_path / "runs")
    client = TestClient(create_app(model_connection_store=models, tool_catalog_store=tools, run_store=run_store, workflow_store=WorkflowStore(tmp_path / "workflows")))
    agent = client.post("/api/agents", json={"name":"approval-agent","model":connection.id,"config":{"tool_ids":[tool.id]}}).json()["id"]
    client.post("/api/graph/entry", json={"agent_id":agent})
    run = client.post("/api/runs", json={"input":{"input":"使用测试工具"}}).json()
    waiting = client.get(f"/api/runs/{run['id']}").json()
    assert waiting["status"] == "waiting_approval", waiting.get("error")
    assert executed == []
    approval = next(e for e in waiting["events"] if e["type"] == "approval_required")
    approved = client.post(f"/api/runs/{run['id']}/approvals/{approval['sequence']}/approve", json={"reason":"test"})
    assert approved.status_code == 200, approved.text
    done = approved.json()
    assert done["id"] == run["id"] and done["status"] == "succeeded", done.get("error")
    assert executed == [tool.id]
    assert approval_statuses == ["running"]
    assert any(e["type"] == "approval_resumed" for e in done["events"])
    assert all(request["model"] == "pinned-test" for request in requests)
    assert any(e["type"] == "tool_started" for e in done["events"])
    assert "".join(e["delta"] for e in done["events"] if e["type"] == "answer_delta") == "已经完成获批的操作。"
    assert "private provider reasoning" not in json.dumps(done["events"])
    repeated = client.post(f"/api/runs/{run['id']}/approvals/{approval['sequence']}/approve", json={"reason":"test"})
    assert repeated.status_code == 200
    assert executed == [tool.id], "reconnecting/retrying approval must not repeat side effects"


def test_run_store_stream_snapshots_preserve_cancellation(tmp_path):
    from engine.modules.workflows import RunStore
    store = RunStore(tmp_path)
    record = store.create(input={}, recursion_limit=10)
    record.status = "running"
    store.save(record)
    store.mark_cancel_requested(record.id, reason="stop")
    record.events.append({"sequence": 1, "type": "answer_delta", "delta": "partial"})
    store.save(record)
    assert store.get(record.id).status == "cancel_requested"
    record.status = "succeeded"
    store.save(record)
    assert store.get(record.id).status == "canceled"
    assert store.get(record.id).events[0]["delta"] == "partial"


def test_real_http_sse_delivers_text_before_run_finishes(tmp_path, monkeypatch):
    """TestClient buffers streams; use a loopback ASGI server for timing proof."""
    import socket
    import time
    import httpx
    import uvicorn
    from engine.server.app import create_app
    from engine.node import Node
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_GRAPH_LOAD_DOTENV", "0")
    release = threading.Event()
    def create(**kwargs):
        yield chunk("这是先到达的正文。" * 30)  # exceeds the batching threshold
        assert release.wait(5), "client did not receive the first delta in time"
        yield chunk("这是后到达的正文。", finish="stop")
    def factory(spec):
        async def execute(state):
            response = await asyncio.to_thread(chat_completion, create, model="offline-stream-test")
            return {"input":response.choices[0].message.content}
        return Node(spec.name, execute)
    app = create_app(node_factory=factory, auth_required=False)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(target=lambda:server.run(sockets=[listener]), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic()+5
        while not server.started and time.monotonic()<deadline:
            time.sleep(.01)
        assert server.started
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=8, trust_env=False) as client:
            info = client.get('/api/system/runtime').json()
            assert info['run_stream_protocol'] == 2
            assert client.get('/openapi.json').json()['info']['x-run-stream-protocol'] == 2
            agent = client.post('/api/agents', json={'name':'HTTP streaming test'}).json()['id']
            client.post('/api/graph/entry', json={'agent_id':agent})
            run = client.post('/api/runs', json={'input':{'input':'safe local test'}}).json()
            received = []
            with client.stream('GET', f"/api/runs/{run['id']}/events") as stream:
                for line in stream.iter_lines():
                    if not line.startswith('data: '):
                        continue
                    event = json.loads(line[6:])
                    received.append(event)
                    if event.get('type') == 'answer_delta' and not release.is_set():
                        assert client.get(f"/api/runs/{run['id']}").json()['status'] == 'running'
                        release.set()
            assert release.is_set()
            assert received[-1]['type'] == 'run_completed'
            assert received[-1]['status'] == 'succeeded'
            assert len([e for e in received if e['type']=='answer_delta']) >= 2
    finally:
        release.set()
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
