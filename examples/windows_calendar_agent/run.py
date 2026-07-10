"""Create a Windows calendar event through an LLM-planned local tool flow.

Usage:
    python examples/windows_calendar_agent/run.py --request "明天下午3点和张三开项目会，持续1小时"
    python examples/windows_calendar_agent/run.py --request "2026-07-11 15:00-16:00 项目会" --open

The LLM only plans structured event fields. The local tool writes an ICS file.
Opening/importing the calendar event still requires user confirmation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from engine import StateGraph
from engine.config import load_settings

try:
    from examples.windows_calendar_agent.tools import (
        CalendarEvent,
        open_calendar_file,
        parse_datetime,
        write_ics,
    )
except ModuleNotFoundError:
    from tools import CalendarEvent, open_calendar_file, parse_datetime, write_ics


PLANNER_PROMPT = """你是一个 Windows 日程规划智能体。
你的任务是把用户的自然语言需求解析成严格 JSON，不要输出 Markdown。

当前时间：{now}

输出 JSON schema：
{{
  "title": "简短日程标题",
  "start": "YYYY-MM-DDTHH:MM:SS",
  "end": "YYYY-MM-DDTHH:MM:SS",
  "description": "日程说明",
  "location": "地点，没有则为空字符串"
}}

规则：
- 如果用户只说持续时间，按持续时间计算 end。
- 如果日期模糊，结合当前时间推断。
- 如果实在无法推断，仍输出 JSON，并在 description 中说明不确定点。
"""


async def plan_event(state: Dict[str, Any]) -> Dict[str, Any]:
    request = state["input"]
    now = datetime.now().replace(microsecond=0)
    settings = load_settings()
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        organization=settings.organization,
    )
    prompt = PLANNER_PROMPT.format(now=now.isoformat())
    response = await client.chat.completions.create(
        model=settings.model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"用户需求：{request}"},
        ],
        temperature=0,
    )
    text = response.choices[0].message.content or "{}"
    data = _parse_json(text)
    return {"event_plan": data, "messages": [{"agent": "calendar_planner", "content": data}]}


async def create_calendar_event(state: Dict[str, Any]) -> Dict[str, Any]:
    data = state["event_plan"]
    event = CalendarEvent(
        title=data.get("title") or "新日程",
        start=parse_datetime(data["start"]),
        end=parse_datetime(data["end"]),
        description=data.get("description", ""),
        location=data.get("location", ""),
    )
    if event.end <= event.start:
        event.end = event.start + timedelta(hours=1)
    path = write_ics(event)
    if state.get("open_calendar", False):
        open_calendar_file(path)
    return {
        "calendar_file": str(path),
        "messages": [
            {
                "agent": "calendar_executor",
                "content": f"已生成日程文件：{path}",
            }
        ],
    }


def build_graph() -> Any:
    graph = StateGraph()
    graph.add_node("calendar_planner", plan_event)
    graph.add_node("calendar_executor", create_calendar_event)
    graph.set_entry_point("calendar_planner")
    graph.add_edge("calendar_planner", "calendar_executor")
    graph.set_finish_point("calendar_executor")
    return graph.compile()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, help="Natural language calendar request.")
    parser.add_argument("--open", action="store_true", help="Open the generated .ics file.")
    args = parser.parse_args()

    graph = build_graph()
    state = await graph.ainvoke({"input": args.request, "open_calendar": args.open})
    print(json.dumps(state, ensure_ascii=False, indent=2, default=str))


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


if __name__ == "__main__":
    asyncio.run(main())
