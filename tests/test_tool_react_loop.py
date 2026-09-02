"""Regression coverage for serial model-driven tool calling."""

import json
from types import SimpleNamespace

from engine.modules.agent_runtime import AgentRuntimeFactory
from engine.modules.model_connections import ModelConnectionStore
from engine.modules.product_ops import ToolCatalogStore
from engine.orchestrator import AgentSpec


def _tool_call(call_id: str, name: str, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments, ensure_ascii=False)),
    )


def test_native_tool_loop_serializes_dependent_calls(tmp_path, monkeypatch):
    catalog = ToolCatalogStore(tmp_path / "tools")
    resolve = catalog.create(
        name="resolve_library",
        display_name="解析库 ID",
        description="解析库名",
        metadata={
            "adapter": "echo",
            "risk": "read",
            "input_schema": {
                "type": "object",
                "properties": {"libraryName": {"type": "string"}, "query": {"type": "string"}},
                "required": ["libraryName", "query"],
            },
        },
    )
    query = catalog.create(
        name="query_docs",
        display_name="查询文档",
        description="按库 ID 查询文档",
        metadata={
            "adapter": "echo",
            "risk": "read",
            "input_schema": {
                "type": "object",
                "properties": {"libraryId": {"type": "string"}, "query": {"type": "string"}},
                "required": ["libraryId", "query"],
            },
        },
    )
    models = ModelConnectionStore(tmp_path / "models")
    models.create({
        "id": "tool-model", "name": "tool model", "provider": "test", "model_id": "test-model",
        "base_url": "https://example.test/v1", "api_key": "test", "test_status": "succeeded",
    })

    responses = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[
            _tool_call("call-1", "resolve_library", {"libraryName": "react", "query": "hooks"}),
        ]))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[
            _tool_call("call-2", "query_docs", {"libraryId": "/facebook/react", "query": "useEffect"}),
        ]))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="React 文档查询完成。", tool_calls=[]))]),
    ]
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return responses.pop(0)

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    factory = AgentRuntimeFactory(tool_catalog_store=catalog, model_connection_store=models)
    spec = AgentSpec(
        id="agent-1", name="docs", model="tool-model",
        config={"tool_ids": [resolve.id, query.id]},
    )

    result, audit = factory._run_model_tool_loop(spec, "查询 React hooks", "", factory._available_tools(spec), {"input": "查询 React hooks"})

    assert result.text == "React 文档查询完成。"
    assert [item["name"] for item in audit] == ["resolve_library", "query_docs"]
    assert audit[0]["arguments"] == {"libraryName": "react", "query": "hooks"}
    assert audit[1]["arguments"] == {"libraryId": "/facebook/react", "query": "useEffect"}
    assert len(calls) == 3
    assert calls[1]["messages"][-1]["role"] == "tool"
