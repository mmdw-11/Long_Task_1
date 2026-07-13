"""可信工作区内的敏感信息脱敏。"""

from __future__ import annotations

import re
from typing import Any, List, Pattern

from ._types import RedactionResult, SensitiveFinding


class SensitiveDataRedactor:
    """基于规则的本地脱敏器。

    该类不调用外部服务，适合在可信工作区内运行。后续可以替换成 DLP 服务或
    本地模型辅助检测，但默认实现必须稳定、可测试。
    """

    def __init__(self) -> None:
        self.patterns: List[tuple[str, Pattern[str]]] = [
            ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
            ("api_key", re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key)\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{12,})['\"]?")),
            ("token", re.compile(r"(?i)\b(token|bearer)\s*[:=]\s*['\"]?([A-Za-z0-9_\-\.]{16,})['\"]?")),
            ("id_card", re.compile(r"\b\d{17}[\dXx]\b")),
            ("bank_card", re.compile(r"\b\d{16,19}\b")),
            ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
            ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
        ]

    def redact(self, payload: Any) -> RedactionResult:
        findings: List[SensitiveFinding] = []
        redacted = self._redact_value(payload, "$", findings)
        return RedactionResult(original=payload, redacted=redacted, findings=findings)

    def _redact_value(
        self, value: Any, path: str, findings: List[SensitiveFinding]
    ) -> Any:
        if isinstance(value, str):
            return self._redact_text(value, path, findings)
        if isinstance(value, list):
            return [
                self._redact_value(item, f"{path}[{idx}]", findings)
                for idx, item in enumerate(value)
            ]
        if isinstance(value, tuple):
            return tuple(
                self._redact_value(item, f"{path}[{idx}]", findings)
                for idx, item in enumerate(value)
            )
        if isinstance(value, dict):
            return {
                key: self._redact_value(item, f"{path}.{key}", findings)
                for key, item in value.items()
            }
        return value

    def _redact_text(
        self, text: str, path: str, findings: List[SensitiveFinding]
    ) -> str:
        redacted = text
        for kind, pattern in self.patterns:
            redacted = self._apply_pattern(redacted, path, kind, pattern, findings)
        return redacted

    def _apply_pattern(
        self,
        text: str,
        path: str,
        kind: str,
        pattern: Pattern[str],
        findings: List[SensitiveFinding],
    ) -> str:
        counter = 0

        def repl(match: re.Match[str]) -> str:
            nonlocal counter
            counter += 1
            replacement = f"[REDACTED:{kind.upper()}:{counter}]"
            findings.append(
                SensitiveFinding(
                    kind=kind,
                    path=path,
                    replacement=replacement,
                    start=match.start(),
                    end=match.end(),
                )
            )
            if kind in {"api_key", "token"} and match.lastindex and match.lastindex >= 2:
                prefix = match.group(1)
                return f"{prefix}={replacement}"
            return replacement

        return pattern.sub(repl, text)
