"""端、边、云推理执行器实现。"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ._types import InferenceRequest, InferenceResult
from .base import InferenceExecutor


class LocalEchoExecutor(InferenceExecutor):
    """端侧本地执行器桩。

    当前不依赖本地 LLM，只返回可观察的本地处理结果。后续可替换成 llama.cpp、
    Ollama 或端侧小模型。
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


class EdgeHttpExecutor(InferenceExecutor):
    """边缘 HTTP 执行器。

    期望边缘服务提供兼容的 JSON 接口：
    request:  {"prompt": "...", "system_prompt": "...", "metadata": {...}}
    response: {"text": "...", "model": "...", "metadata": {...}}

    若未配置真实 HTTP endpoint，可选择 fallback 执行器。
    """

    def __init__(self, *, fallback: Optional[InferenceExecutor] = None) -> None:
        self.fallback = fallback or LocalEchoExecutor(label="edge-fallback")

    def run(self, request: InferenceRequest) -> InferenceResult:
        endpoint = request.allocation.get("endpoint", "")
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
                timeout=30,
            )
            response.raise_for_status()
            data: Dict[str, Any] = response.json()
            return InferenceResult(
                text=str(data.get("text", "")),
                executor=type(self).__name__,
                endpoint=endpoint,
                model=str(data.get("model", "")),
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
    """OpenAI SDK 兼容云端执行器。

    支持 OpenAI 官方和 BigModel 等兼容 ``chat.completions`` 的服务。
    """

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
        system = request.system_prompt or "你是云端高复杂度任务执行器，回答要简洁、结构化。"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": request.prompt},
        ]
        response = client.chat.completions.create(
            model=self.model or settings.model,
            messages=messages,
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
