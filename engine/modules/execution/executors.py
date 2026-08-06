"""Device, edge, and cloud inference executor implementations."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from ._types import InferenceRequest, InferenceResult
from .base import InferenceExecutor


class LocalEchoExecutor(InferenceExecutor):
    """Observable local fallback executor.

    This does not call a real local LLM. It returns a truncated prompt so demos
    and tests can run without a configured device model.
    """

    def __init__(self, *, label: str = "device-local") -> None:
        self.label = label

    def run(self, request: InferenceRequest) -> InferenceResult:
        text = request.prompt.strip()
        if len(text) > 240:
            text = text[:240] + "..."
        return InferenceResult(
            text=f"[{self.label}] {text}",
            executor=type(self).__name__,
            endpoint=request.allocation.get("endpoint", "local"),
            model="local-echo",
            metadata={"simulated": True},
        )


class OpenAICompatibleExecutor(InferenceExecutor):
    """Configurable OpenAI-compatible chat executor.

    It is used for real device/edge/cloud backends when they expose a
    ``/chat/completions`` compatible endpoint. Missing configuration falls
    back to the provided executor so demos and tests can still run offline.
    """

    def __init__(
        self,
        *,
        env_prefix: str,
        fallback: Optional[InferenceExecutor] = None,
        default_endpoint: str = "",
        default_model: str = "",
        label: str = "openai-compatible",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.env_prefix = env_prefix
        self.fallback = fallback
        self.default_endpoint = default_endpoint
        self.default_model = default_model
        self.label = label
        self.timeout_seconds = timeout_seconds

    def run(self, request: InferenceRequest) -> InferenceResult:
        _load_dotenv()
        endpoint = _env_or_default(f"{self.env_prefix}_BASE_URL", self.default_endpoint)
        model = _env_or_default(f"{self.env_prefix}_MODEL", self.default_model)
        api_key = os.environ.get(f"{self.env_prefix}_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

        if not endpoint or not model:
            if self.fallback is None:
                return InferenceResult(
                    text="",
                    executor=type(self).__name__,
                    endpoint=endpoint or request.allocation.get("endpoint", ""),
                    model=model,
                    success=False,
                    error=f"missing {self.env_prefix}_BASE_URL or {self.env_prefix}_MODEL",
                    retryable=False,
                )
            result = self.fallback.run(request)
            result.metadata = {
                **result.metadata,
                "fallback_reason": f"missing {self.env_prefix}_BASE_URL or {self.env_prefix}_MODEL",
            }
            return result

        from openai import OpenAI

        client = OpenAI(api_key=api_key or "not-needed", base_url=endpoint, timeout=self.timeout_seconds)
        system = request.system_prompt or "Answer concisely and accurately."
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": request.prompt},
            ],
            temperature=0,
            max_tokens=int(os.environ.get(f"{self.env_prefix}_MAX_TOKENS", "384")),
        )
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        usage_data = json.loads(usage.model_dump_json()) if usage is not None else {}
        return InferenceResult(
            text=text,
            executor=type(self).__name__,
            endpoint=endpoint,
            model=model,
            metadata={"usage": usage_data, "backend": self.label},
        )


class LocalModelExecutor(OpenAICompatibleExecutor):
    """Device-side real model executor with echo fallback."""

    def __init__(self, *, fallback: Optional[InferenceExecutor] = None) -> None:
        super().__init__(
            env_prefix="DEVICE",
            fallback=fallback or LocalEchoExecutor(),
            default_endpoint=os.environ.get("DEVICE_ENDPOINT", ""),
            default_model=os.environ.get("DEVICE_MODEL", ""),
            label="device",
            timeout_seconds=float(os.environ.get("DEVICE_TIMEOUT_SECONDS", "60")),
        )


class EdgeHttpExecutor(InferenceExecutor):
    """Edge HTTP executor with local fallback.

    Expected request body:
    ``{"prompt": "...", "system_prompt": "...", "metadata": {...}}``

    Expected response body:
    ``{"text": "...", "model": "...", "metadata": {...}}``
    """

    def __init__(self, *, fallback: Optional[InferenceExecutor] = None) -> None:
        self.fallback = fallback or LocalEchoExecutor(label="edge-fallback")

    def run(self, request: InferenceRequest) -> InferenceResult:
        _load_dotenv()
        endpoint = request.allocation.get("endpoint", "")
        if os.environ.get("AGENT_GRAPH_LOAD_DOTENV") != "0":
            endpoint = os.environ.get("EDGE_ENDPOINT") or endpoint
        if not endpoint.startswith(("http://", "https://")):
            result = self.fallback.run(request)
            result.endpoint = endpoint or result.endpoint
            result.metadata = {**result.metadata, "edge_fallback": True}
            return result

        try:
            import requests

            response = requests.post(
                endpoint,
                json={
                    "prompt": request.prompt,
                    "system_prompt": request.system_prompt,
                    "metadata": request.metadata,
                },
                timeout=float(os.environ.get("EDGE_REQUEST_TIMEOUT_SECONDS", "150")),
            )
            response.raise_for_status()
            data: Dict[str, Any] = response.json()
            return InferenceResult(
                text=str(data.get("text", "")),
                executor=type(self).__name__,
                endpoint=endpoint,
                model=str(data.get("model", os.environ.get("EDGE_MODEL", ""))),
                metadata=dict(data.get("metadata") or {}),
            )
        except Exception as exc:  # noqa: BLE001 - edge fallback keeps workflow alive
            result = self.fallback.run(request)
            result.endpoint = endpoint
            result.metadata = {
                **result.metadata,
                "edge_fallback": True,
                "edge_error": str(exc),
            }
            return result


class OpenAICompatibleCloudExecutor(InferenceExecutor):
    """Cloud executor using project-level OpenAI-compatible settings."""

    def __init__(self, *, model: Optional[str] = None) -> None:
        self.model = model

    def run(self, request: InferenceRequest) -> InferenceResult:
        from openai import OpenAI

        from engine.config import load_settings

        settings = load_settings()
        client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            organization=settings.organization,
        )
        system = request.system_prompt or (
            "You are the cloud executor for complex tasks. "
            "Answer concisely and structurally."
        )
        response = client.chat.completions.create(
            model=self.model or settings.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": request.prompt},
            ],
            temperature=0,
        )
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        usage_data = json.loads(usage.model_dump_json()) if usage is not None else {}
        return InferenceResult(
            text=text,
            executor=type(self).__name__,
            endpoint=request.allocation.get("endpoint", "cloud"),
            model=self.model or settings.model,
            metadata={"usage": usage_data},
        )


def _env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _load_dotenv() -> None:
    if os.environ.get("AGENT_GRAPH_LOAD_DOTENV") == "0":
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
        return
    except Exception:
        pass
    env_path = _find_dotenv()
    if env_path is None:
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _find_dotenv():
    from pathlib import Path

    current = Path.cwd()
    for path in [current, *current.parents]:
        candidate = path / ".env"
        if candidate.exists():
            return candidate
    return None
