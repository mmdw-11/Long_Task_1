"""Regression coverage for serial model-driven tool calling."""

import json
from types import SimpleNamespace

from engine.modules.agent_runtime import AgentRuntimeFactory
from engine.modules.model_connections import ModelConnectionStore
from engine.modules.product_ops import ToolCatalogStore
from engine.modules.tool_runtime import ToolRuntime, ensure_builtin_tools
from engine.modules.tools.contracts import decode_tool_arguments
from engine.modules.workspace_tools import WorkspaceStore
from engine.modules.workflows import RunStore, WorkflowStore
from engine.orchestrator import AgentSpec
from engine.server.app import create_app

from fastapi.testclient import TestClient


def _tool_call(call_id: str, name: str, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments, ensure_ascii=False)),
    )


def test_decode_tool_arguments_unwraps_provider_argument_envelope():
    assert decode_tool_arguments('{"arguments":{"path":"data_organizer.py","create":true}}') == {
        "path": "data_organizer.py",
        "create": True,
    }


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


def test_engineering_scope_approval_continues_local_code_tools(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    workspace_store = WorkspaceStore(tmp_path / "workspaces")
    workspace = workspace_store.create(name="demo", root_path=str(project))
    catalog = ToolCatalogStore(tmp_path / "tools")
    ensure_builtin_tools(catalog)
    write_tool = next(item for item in catalog.list() if item.name == "workspace_write_files")
    patch_tool = next(item for item in catalog.list() if item.name == "workspace_apply_patch")
    models = ModelConnectionStore(tmp_path / "models")
    models.create({
        "id": "tool-model", "name": "tool model", "provider": "test", "model_id": "test-model",
        "base_url": "https://example.test/v1", "api_key": "test", "test_status": "succeeded",
    })

    runtime = ToolRuntime(catalog, workspace_store=workspace_store)
    approval = runtime.execute(
        write_tool,
        "创建一个 Java 项目",
        arguments={
            "workspace_id": workspace.id,
            "files": [{"path": "src/Main.java", "content": "class Main { static String v(){ return \"old\"; } }\n"}],
        },
    ).to_dict()
    assert approval["status"] == "approval_required"
    approved_result = runtime.execute(
        write_tool,
        "创建一个 Java 项目",
        arguments=approval["arguments"],
        bypass_approval=True,
    ).to_dict()
    assert approved_result["status"] == "succeeded"

    responses = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[
            _tool_call("call-2", "workspace_apply_patch", {
                "path": "src/Main.java",
                "old_text": "old",
                "new_text": "new",
            }),
        ]))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="项目已创建并修正完成。", tool_calls=[]))]),
    ]

    class FakeCompletions:
        def create(self, **_kwargs):
            return responses.pop(0)

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    factory = AgentRuntimeFactory(tool_catalog_store=catalog, model_connection_store=models, workspace_store=workspace_store)
    spec = AgentSpec(
        id="agent-1", name="coder", model="tool-model",
        config={"tool_ids": [write_tool.id, patch_tool.id], "workspace_id": workspace.id},
    )
    continuation, audit = factory.continue_tool_loop_after_tool_decisions(
        spec,
        task_text="创建一个 Java 项目",
        outcomes=[{"approved": True, "tool_call": approval, "result": approved_result}],
        state={"input": "创建一个 Java 项目", "workspace_id": workspace.id, "__approved_tool_scopes__": ["workspace_engineering"]},
    )

    assert continuation.text == "项目已创建并修正完成。"
    assert audit[0]["status"] == "succeeded"
    assert audit[0]["approval_required"] is False
    assert "new" in (project / "src" / "Main.java").read_text(encoding="utf-8")


def test_approval_api_records_engineering_scope_after_local_write(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    workspace_store = WorkspaceStore(tmp_path / "workspaces")
    workspace = workspace_store.create(name="demo", root_path=str(project))
    catalog = ToolCatalogStore(tmp_path / "tools")
    ensure_builtin_tools(catalog)
    write_tool = next(item for item in catalog.list() if item.name == "workspace_write_files")
    run_store = RunStore(tmp_path / "runs")
    app = create_app(
        workflow_store=WorkflowStore(tmp_path / "workflows"),
        run_store=run_store,
        tool_catalog_store=catalog,
        workspace_store=workspace_store,
        auth_required=False,
    )
    client = TestClient(app)

    record = run_store.create(
        input={"input": "创建一个 Java 项目", "original_goal": "创建一个 Java 项目"},
        recursion_limit=10,
    )
    record.status = "waiting_approval"
    record.state = {"input": "创建一个 Java 项目", "workspace_id": workspace.id}
    record.events = [
        {
            "sequence": 1,
            "type": "approval_required",
            "node": "coder",
            "tool_call": {
                "id": write_tool.id,
                "name": write_tool.name,
                "display_name": write_tool.display_name,
                "status": "approval_required",
                "arguments": {
                    "workspace_id": workspace.id,
                    "arguments": {
                        "files": [{"path": "src/Main.java", "content": "public class Main {}\n"}],
                    },
                },
            },
        }
    ]
    run_store.save(record)

    response = client.post(f"/api/runs/{record.id}/approvals/1/approve", json={"reason": "用户批准"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "succeeded"
    assert "workspace_engineering" in payload["state"]["__approved_tool_scopes__"]
    assert (project / "src" / "Main.java").read_text(encoding="utf-8") == "public class Main {}\n"
