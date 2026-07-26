"""Small edge HTTP wrapper backed by Ollama.

Run:
    python examples/edge_cloud_scheduler/edge_ollama_server.py

Endpoint:
    POST http://127.0.0.1:8001/infer

Request:
    {"prompt": "...", "system_prompt": "...", "metadata": {...}}

Response:
    {"text": "...", "model": "...", "metadata": {...}}
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

import requests


def load_env() -> None:
    current = Path.cwd()
    for path in [current, *current.parents]:
        env_path = path / ".env"
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return


class EdgeOllamaHandler(BaseHTTPRequestHandler):
    server_version = "EdgeOllama/0.1"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"ok": True, "model": edge_model()})
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if self.path != "/infer":
            self._send_json({"error": "not found"}, status=404)
            return

        try:
            payload = self._read_json()
            prompt = str(payload.get("prompt") or "")
            system_prompt = str(payload.get("system_prompt") or "Answer concisely and accurately.")
            metadata = dict(payload.get("metadata") or {})
            response = requests.post(
                ollama_base_url().rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + os.environ.get("EDGE_OLLAMA_API_KEY", "ollama")},
                json={
                    "model": edge_model(),
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    # Bound local generation so one pathological prompt cannot
                    # block later requests in the single Ollama queue.
                    "max_tokens": int(os.environ.get("EDGE_MAX_TOKENS", "384")),
                },
                timeout=float(os.environ.get("EDGE_TIMEOUT_SECONDS", "150")),
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"].get("content") or ""
            self._send_json(
                {
                    "text": text,
                    "model": edge_model(),
                    "metadata": {
                        "backend": "edge-ollama",
                        "ollama_base_url": ollama_base_url(),
                        "request_metadata": metadata,
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001 - server returns structured error
            self._send_json({"error": str(exc), "model": edge_model()}, status=500)

    def log_message(self, fmt: str, *args: Any) -> None:
        if os.environ.get("EDGE_LOG_REQUESTS", "0") == "1":
            super().log_message(fmt, *args)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("content-length") or "0")
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def _send_json(self, payload: Dict[str, Any], *, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def ollama_base_url() -> str:
    return os.environ.get("EDGE_OLLAMA_BASE_URL") or os.environ.get("DEVICE_BASE_URL") or "http://127.0.0.1:11434/v1"


def edge_model() -> str:
    return os.environ.get("EDGE_MODEL") or os.environ.get("DEVICE_MODEL") or "qwen2.5:0.5b"


def main() -> None:
    load_env()
    host = os.environ.get("EDGE_HOST", "127.0.0.1")
    port = int(os.environ.get("EDGE_PORT", "8001"))
    server = ThreadingHTTPServer((host, port), EdgeOllamaHandler)
    print(f"edge ollama server listening on http://{host}:{port}/infer model={edge_model()}")
    server.serve_forever()


if __name__ == "__main__":
    main()
