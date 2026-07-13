"""LLM 记忆判断层：mem0 风格的记忆更新决策。"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Protocol


_MEMORY_UPDATE_SYSTEM_PROMPT = """\
You are an expert memory management system. Your job is to decide whether a new piece of \
information should update, replace, or be added alongside existing memories.

You will be given:
1. A NEW memory (the information to be stored)
2. A list of EXISTING memories that are semantically similar to the new one

For each existing memory, decide the relationship with the new memory.

Output a JSON object with exactly one field "actions", which is a list of objects.
Each object has:
- "id": the id of the existing memory
- "action": one of "add", "update", "delete", "noop"

Action definitions:
- "add": The new memory contains entirely new information not covered by this existing memory. \
Both should coexist. (Note: this means the new memory will be added as a new entry.)
- "update": The new memory updates, corrects, or supplements this existing memory. \
The existing memory should be replaced by the new one.
- "delete": The new memory is a duplicate of this existing memory, or the existing memory is \
now outdated/contradicted. Discard the new memory (keep existing as is or remove it).
- "noop": The existing memory and new memory are unrelated; no action needed.

Guidelines:
- If the new memory contains the same information as an existing one, choose "delete" (redundant).
- If the new memory corrects or extends an existing one, choose "update".
- If they are about different topics, choose "add".
- Be conservative: prefer "add" when in doubt.

Respond ONLY with the JSON object, no other text.
"""


class MemoryLLMJudge(Protocol):
    """Protocol for LLM-based memory update decision makers.

    Implementations should analyze the new memory against existing similar
    memories and return a list of (memory_id, action) decisions.
    """

    def judge(
        self,
        new_memory_text: str,
        existing_memories: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Decide update actions for new vs existing memories.

        :param new_memory_text: The text content of the new memory.
        :param existing_memories: List of dicts with "id" and "content" keys.
        :return: List of dicts with "id" and "action" keys.
        """
        raise NotImplementedError


class OpenAIMemoryJudge:
    """基于 OpenAI 兼容接口的记忆更新判断器。

    使用 Chat Completions API 让 LLM 判断新记忆与已有记忆的关系。

    :param api_key: OpenAI API key；默认从环境变量 ``OPENAI_API_KEY`` 读取。
    :param model: 模型名称，默认 ``gpt-4o-mini``。
    :param base_url: 可选的 API base URL（兼容自建/中转服务）。
    :param temperature: 生成温度，默认 0 以获得确定性输出。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        try:
            import openai  # type: ignore
        except ImportError:
            raise ImportError(
                "使用 OpenAIMemoryJudge 需要安装 openai：\n"
                "  pip install openai"
            )
        self._model = model
        self._temperature = temperature
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def judge(
        self,
        new_memory_text: str,
        existing_memories: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        if not existing_memories:
            return []

        existing_formatted = json.dumps(existing_memories, ensure_ascii=False, indent=2)
        user_prompt = (
            f"NEW memory:\n{new_memory_text}\n\n"
            f"EXISTING memories:\n{existing_formatted}"
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _MEMORY_UPDATE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._temperature,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or ""
        try:
            result = json.loads(content)
            return result.get("actions", [])
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from the response
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                    return result.get("actions", [])
                except json.JSONDecodeError:
                    pass
            return []
