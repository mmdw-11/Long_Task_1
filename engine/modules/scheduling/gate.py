"""任务门控与画像构建。"""

from __future__ import annotations

import re
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ._types import (
    RealtimeRequirement,
    ResourceRequest,
    SensitivityLevel,
    TaskComplexity,
    TaskProfile,
)


class TaskGate(ABC):
    """任务门控单元接口。

    默认实现是本地启发式规则；生产环境可替换成本地 LLM skill 或外部模型路由器。
    """

    @abstractmethod
    def evaluate(self, request: ResourceRequest) -> TaskProfile:
        """评估任务画像。"""
        raise NotImplementedError


class HeuristicTaskGate(TaskGate):
    """无需外部模型的本地门控单元。

    显式 metadata 优先，其次根据文本长度和关键词判断实时性、敏感等级、复杂度与
    任务类型。它是确定性实现，便于单元测试。
    """

    _secret_patterns = [
        r"api[_ -]?key",
        r"password",
        r"token",
        r"secret",
        r"身份证",
        r"银行卡",
        r"密钥",
        r"私钥",
    ]
    _urgent_words = ["urgent", "实时", "立即", "马上", "紧急", "低延迟"]
    _cloud_words = ["大规模", "复杂", "长文", "图像", "视频", "全量分析", "深度推理"]

    def evaluate(self, request: ResourceRequest) -> TaskProfile:
        meta = dict(request.metadata or {})
        text = self._collect_text(request)
        sensitivity = self._enum_from_meta(
            meta, "sensitivity", SensitivityLevel, self._infer_sensitivity(text)
        )
        realtime = self._enum_from_meta(
            meta, "realtime", RealtimeRequirement, self._infer_realtime(text)
        )
        complexity = self._enum_from_meta(
            meta, "complexity", TaskComplexity, self._infer_complexity(text)
        )
        task_type = str(meta.get("task_type") or self._infer_task_type(text))
        return TaskProfile(
            realtime=realtime,
            sensitivity=sensitivity,
            complexity=complexity,
            task_type=task_type,
            requires_trusted_workspace=bool(
                meta.get("requires_trusted_workspace")
                or sensitivity in (SensitivityLevel.CONFIDENTIAL, SensitivityLevel.SECRET)
            ),
            human_approved=bool(
                meta.get("human_approved")
                or request.state.get("human_approved")
                or request.state.get("cloud_audit_approved")
            ),
            metadata={
                "node": request.node,
                "text_length": len(text),
                "explicit": {
                    key: value
                    for key, value in meta.items()
                    if key in {"sensitivity", "realtime", "complexity", "task_type"}
                },
            },
        )

    def _collect_text(self, request: ResourceRequest) -> str:
        parts = [request.node]
        for key in ("input", "task", "goal", "query", "messages"):
            value = request.state.get(key)
            if value:
                parts.append(str(value))
        for key in ("description", "sys_prompt"):
            value = request.metadata.get(key)
            if value:
                parts.append(str(value))
        return "\n".join(parts)

    def _infer_sensitivity(self, text: str) -> SensitivityLevel:
        lowered = text.lower()
        if any(re.search(pattern, lowered) for pattern in self._secret_patterns):
            return SensitivityLevel.SECRET
        if any(word in text for word in ["客户", "隐私", "个人", "内部", "邮件"]):
            return SensitivityLevel.CONFIDENTIAL
        return SensitivityLevel.INTERNAL

    def _infer_realtime(self, text: str) -> RealtimeRequirement:
        if any(word in text for word in self._urgent_words):
            return RealtimeRequirement.HARD
        if len(text) < 600:
            return RealtimeRequirement.INTERACTIVE
        return RealtimeRequirement.NORMAL

    def _infer_complexity(self, text: str) -> TaskComplexity:
        if (
            len(text) > 6000
            or any(word in text for word in self._cloud_words)
            or re.search(r"\d+\s*万\s*字", text)
            or re.search(r"[几数多十百千万]+\s*万\s*字", text)
        ):
            return TaskComplexity.HIGH
        if len(text) > 1800:
            return TaskComplexity.MEDIUM
        return TaskComplexity.LOW

    def _infer_task_type(self, text: str) -> str:
        if any(word in text for word in ["日程", "calendar", "会议"]):
            return "calendar"
        if any(word in text for word in ["邮件", "email", "回复"]):
            return "email"
        if any(word in text for word in ["代码", "测试", "bug"]):
            return "code"
        return "general"

    def _enum_from_meta(self, meta: Dict[str, Any], key: str, enum_cls: Any, default: Any) -> Any:
        value = meta.get(key)
        if value is None:
            return default
        if isinstance(value, enum_cls):
            return value
        return enum_cls(str(value).lower())


class OpenAITaskGate(TaskGate):
    """使用 OpenAI/兼容接口构建任务画像的门控单元。

    该实现是真实大模型路径：它会调用 ``engine.config.load_settings`` 读取 `.env`，
    再请求 Chat Completions 输出 JSON 画像。调度器仍会在画像之后执行可信工作区
    规则，避免把安全边界完全交给模型。
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        fallback: Optional[TaskGate] = None,
        allow_fallback: bool = False,
    ) -> None:
        self.model = model
        self.fallback = fallback or HeuristicTaskGate()
        self.allow_fallback = allow_fallback

    def evaluate(self, request: ResourceRequest) -> TaskProfile:
        prompt = self._build_prompt(request)
        try:
            data = self._call_model(prompt)
            return self._profile_from_data(data, request)
        except Exception as exc:  # noqa: BLE001 - optionally fallback for demos
            if not self.allow_fallback:
                raise
            profile = self.fallback.evaluate(request)
            profile.metadata = {
                **profile.metadata,
                "gate": "openai_fallback",
                "gate_error": str(exc),
            }
            return profile

    def _build_prompt(self, request: ResourceRequest) -> str:
        text = self.fallback._collect_text(request) if isinstance(self.fallback, HeuristicTaskGate) else str(request.state)
        return f"""你是端-边-云推理调度的本地门控单元。
请根据任务文本和节点元数据，输出严格 JSON，不要 Markdown。

枚举要求：
- realtime: hard | interactive | normal | batch
- sensitivity: public | internal | confidential | secret
- complexity: low | medium | high | extreme
- task_type: general | email | calendar | code | vision | document | recovery

判断原则：
- 涉及密钥、token、个人身份、客户隐私、内部邮件时提高 sensitivity。
- 强实时、紧急响应、低延迟交互使用 hard 或 interactive。
- 长文、多模态、复杂推理、全量分析、故障恢复提高 complexity。
- 不确定时保守提高 sensitivity，但不要无故提高 complexity。

节点：{request.node}
节点元数据：{json.dumps(request.metadata, ensure_ascii=False, default=str)}
任务文本：
{text}

输出 JSON schema：
{{
  "realtime": "...",
  "sensitivity": "...",
  "complexity": "...",
  "task_type": "...",
  "requires_trusted_workspace": true,
  "reason": "简短中文理由"
}}
"""

    def _call_model(self, prompt: str) -> Dict[str, Any]:
        from openai import OpenAI

        from engine.config import load_settings

        settings = load_settings()
        client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            organization=settings.organization,
        )
        response = client.chat.completions.create(
            model=self.model or settings.model,
            messages=[
                {"role": "system", "content": "你只输出可解析 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        text = response.choices[0].message.content or "{}"
        return self._parse_json(text)

    def _profile_from_data(self, data: Dict[str, Any], request: ResourceRequest) -> TaskProfile:
        human_approved = bool(
            request.metadata.get("human_approved")
            or request.state.get("human_approved")
            or request.state.get("cloud_audit_approved")
        )
        sensitivity = SensitivityLevel(str(data.get("sensitivity", "internal")).lower())
        return TaskProfile(
            realtime=RealtimeRequirement(str(data.get("realtime", "normal")).lower()),
            sensitivity=sensitivity,
            complexity=TaskComplexity(str(data.get("complexity", "medium")).lower()),
            task_type=str(data.get("task_type", "general")),
            requires_trusted_workspace=bool(
                data.get("requires_trusted_workspace")
                or sensitivity in (SensitivityLevel.CONFIDENTIAL, SensitivityLevel.SECRET)
            ),
            human_approved=human_approved,
            metadata={
                "node": request.node,
                "gate": "openai",
                "model_reason": data.get("reason", ""),
            },
        )

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end >= start:
            stripped = stripped[start : end + 1]
        return json.loads(stripped)
