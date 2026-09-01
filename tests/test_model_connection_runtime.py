"""End-to-end verification for direct model credentials and Agent model binding."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from engine.modules.agent_runtime import AgentRuntimeFactory
from engine.modules.model_connections import ModelConnectionStore
from engine.orchestrator import AgentSpec


def test_agent_uses_directly_configured_model_connection(tmp_path):
    observed: dict[str, str] = {}

    class FakeOpenAIHandler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
            observed["authorization"] = self.headers.get("Authorization", "")
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode())
            observed["model"] = body["model"]
            response = json.dumps({"choices": [{"message": {"content": "这是模拟模型的正常回答。"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        store = ModelConnectionStore(tmp_path / "models")
        connection = store.create({
            "name": "用户直接配置的模型",
            "provider": "openai-compatible",
            "model_id": "demo-model-001",
            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
            "api_key": "user-entered-key",
            "tier": "cloud",
            "test_status": "succeeded",
        })
        node = AgentRuntimeFactory(model_connection_store=store)(AgentSpec(
            id="agent-demo", name="演示 Agent", model=connection.id, sys_prompt="请简洁回答。"
        ))

        result = asyncio.run(node.invoke({"input": "请回答这条问题"}))

        assert result is not None
        assert result["input"] == "这是模拟模型的正常回答。"
        assert observed == {"authorization": "Bearer user-entered-key", "model": "demo-model-001"}
    finally:
        server.shutdown()
        server.server_close()


def test_auto_does_not_fall_back_to_environment_models_when_connections_are_unready(tmp_path):
    """An explicit AUTO choice must use the UI connection catalog exclusively."""
    store = ModelConnectionStore(tmp_path / "models")
    node = AgentRuntimeFactory(model_connection_store=store)(AgentSpec(
        id="agent-auto", name="AUTO agent", model="auto", sys_prompt="请简洁回答。"
    ))

    with pytest.raises(RuntimeError, match="AUTO 尚未就绪：端、边、云模型未配置可用的默认连接"):
        asyncio.run(node.invoke({"input": "请回答这条问题"}))
