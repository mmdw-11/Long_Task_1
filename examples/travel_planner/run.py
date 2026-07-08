"""旅行计划多 Agent 流水线 —— 基于 engine 框架。

7 个 Agent 组成线性流水线：
    leader → destination → transport → accommodation → itinerary → budget_review → final_report

运行（需在 .env 中配置 OPENAI_API_KEY）：
    PYTHONPATH=. python examples/travel_planner/run.py
    PYTHONPATH=. python examples/travel_planner/run.py --request "3人从北京去杭州4天，预算6000"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

# 确保作为脚本直接运行时能找到 engine 包
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_script_dir = str(Path(__file__).resolve().parent)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from engine import END, START, StateGraph, append_reducer
from engine.config import build_openai_agent, load_settings
from engine.node import NodeType

try:
    from .prompts import (
        ACCOMMODATION_PROMPT,
        BUDGET_REVIEW_PROMPT,
        DESTINATION_PROMPT,
        FINAL_REPORT_PROMPT,
        ITINERARY_PROMPT,
        LEADER_PROMPT,
        TRANSPORT_PROMPT,
    )
    from .tools import build_toolkit
except ImportError:
    from prompts import (
        ACCOMMODATION_PROMPT,
        BUDGET_REVIEW_PROMPT,
        DESTINATION_PROMPT,
        FINAL_REPORT_PROMPT,
        ITINERARY_PROMPT,
        LEADER_PROMPT,
        TRANSPORT_PROMPT,
    )
    from tools import build_toolkit

# ---------------------------------------------------------------------------
# 节点配置：每个 Agent 的任务模板、输出 key、可用工具
# ---------------------------------------------------------------------------
NODE_CONFIGS: list[dict] = [
    {
        "agent_id": "leader",
        "name": "leader",
        "sys_prompt": LEADER_PROMPT,
        "task": "理解以下旅行需求，提炼关键约束（出发地、目的地、天数、人数、预算、兴趣）并给出需求总结：\n{user_request}",
        "output_key": "intake_summary",
        "tools": [],
    },
    {
        "agent_id": "destination",
        "name": "destination",
        "sys_prompt": DESTINATION_PROMPT,
        "task": "基于以下需求分析，先用 web_search 检索，再用 browse_webpage 打开关键链接阅读网页内容，基于真实内容给出目的地研究：\n{intake_summary}",
        "output_key": "destination_notes",
        "tools": ["web_search", "browse_webpage"],
    },
    {
        "agent_id": "transport",
        "name": "transport",
        "sys_prompt": TRANSPORT_PROMPT,
        "task": "基于以下信息规划交通方案，先用 web_search 检索，再用 browse_webpage 打开关键链接阅读网页内容后作答：\n{intake_summary}\n{destination_notes}",
        "output_key": "transport_plan",
        "tools": ["web_search", "browse_webpage"],
    },
    {
        "agent_id": "accommodation",
        "name": "accommodation",
        "sys_prompt": ACCOMMODATION_PROMPT,
        "task": "基于以下信息规划住宿方案，先用 web_search 检索，再用 browse_webpage 打开关键链接阅读网页内容后作答：\n{destination_notes}\n{transport_plan}",
        "output_key": "accommodation_plan",
        "tools": ["web_search", "browse_webpage"],
    },
    {
        "agent_id": "itinerary",
        "name": "itinerary",
        "sys_prompt": ITINERARY_PROMPT,
        "task": "综合以下信息，生成按天行程：\n{destination_notes}\n{transport_plan}\n{accommodation_plan}",
        "output_key": "itinerary",
        "tools": [],
    },
    {
        "agent_id": "budget_review",
        "name": "budget_review",
        "sys_prompt": BUDGET_REVIEW_PROMPT,
        "task": "审查以下行程的预算与可行性，先调用 estimate_budget 得到基准预算，再用 sum_expenses 汇总开销、split_expense 计算人均分摊后作答：\n{itinerary}",
        "output_key": "budget_review",
        "tools": ["estimate_budget", "sum_expenses", "split_expense"],
    },
    {
        "agent_id": "final_report",
        "name": "final_report",
        "sys_prompt": FINAL_REPORT_PROMPT,
        "task": "综合所有 Agent 产出，合成最终旅行计划：\n{itinerary}\n{budget_review}",
        "output_key": "final_plan",
        "tools": [],
    },
]

PIPELINE_ORDER = [c["name"] for c in NODE_CONFIGS]


# ---------------------------------------------------------------------------
# 构建图并运行
# ---------------------------------------------------------------------------
def build_travel_graph(settings):
    """构建旅行计划流水线图。每个节点调用真实 OpenAI agent。"""
    from agentscope.message import Msg, TextBlock
    from agentscope.event import (
        ReplyEndEvent,
        TextBlockDeltaEvent,
        ThinkingBlockDeltaEvent,
        ThinkingBlockEndEvent,
        ThinkingBlockStartEvent,
        ToolCallStartEvent,
        ToolResultEndEvent,
        ToolResultStartEvent,
        ToolResultTextDeltaEvent,
    )

    schema = {
        "messages": append_reducer,
        "events": append_reducer,
    }
    graph = StateGraph(schema=schema)

    # 为每个 Agent 创建真实 OpenAI 实例并注册为图节点（开启流式输出）
    agents = {}
    for cfg in NODE_CONFIGS:
        toolkit = build_toolkit(cfg.get("tools", []))
        agents[cfg["name"]] = build_openai_agent(
            cfg["name"], cfg["sys_prompt"], settings, stream=True, toolkit=toolkit
        )

    for cfg in NODE_CONFIGS:
        agent = agents[cfg["name"]]
        output_key = cfg["output_key"]
        task_template = cfg["task"]

        def _make_node(ag, tpl, okey, name, aid):
            async def node(state: Dict[str, Any]) -> Dict[str, Any]:
                # 用上游 agent 的输出填充任务模板
                try:
                    safe = {k: v for k, v in state.items() if isinstance(k, str)}
                    prompt = tpl.format(**safe)
                except KeyError:
                    prompt = tpl + "\n\n当前状态：" + str(state)

                print(f"\n{'=' * 60}\n▶ [{name}] 开始执行\n{'=' * 60}", flush=True)

                text_parts: list[str] = []
                tool_calls: list[str] = []
                shown_result: dict[str, int] = {}
                result_limit = 200

                async for ev in ag.reply_stream(
                    Msg(name="user", content=[TextBlock(type="text", text=prompt)], role="user")
                ):
                    if isinstance(ev, TextBlockDeltaEvent):
                        sys.stdout.write(ev.delta)
                        sys.stdout.flush()
                        text_parts.append(ev.delta)
                    elif isinstance(ev, ThinkingBlockStartEvent):
                        sys.stdout.write("\n【思考】")
                        sys.stdout.flush()
                    elif isinstance(ev, ThinkingBlockDeltaEvent):
                        sys.stdout.write(ev.delta)
                        sys.stdout.flush()
                    elif isinstance(ev, ThinkingBlockEndEvent):
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                    elif isinstance(ev, ToolCallStartEvent):
                        tool_calls.append(ev.tool_call_name)
                        print(f"\n  → 调用工具：{ev.tool_call_name}", flush=True)
                    elif isinstance(ev, ToolResultStartEvent):
                        sys.stdout.write(f"  ← 工具结果[{ev.tool_call_name}]：")
                        sys.stdout.flush()
                        shown_result[ev.tool_call_id] = 0
                    elif isinstance(ev, ToolResultTextDeltaEvent):
                        shown = shown_result.get(ev.tool_call_id, 0)
                        if shown < result_limit:
                            take = ev.delta[: result_limit - shown]
                            sys.stdout.write(take.replace("\n", " "))
                            sys.stdout.flush()
                            shown_result[ev.tool_call_id] = shown + len(take)
                            if shown_result[ev.tool_call_id] >= result_limit:
                                sys.stdout.write(" …(已截断)")
                                sys.stdout.flush()
                    elif isinstance(ev, ToolResultEndEvent):
                        print(f"  [状态：{ev.state}]", flush=True)
                    elif isinstance(ev, ReplyEndEvent):
                        print(flush=True)

                text = "".join(text_parts).strip()
                return {
                    okey: text,
                    "messages": [{"agent": name, "text": text}],
                    "events": [{"node": name, "agent": aid, "tools": tool_calls}],
                }
            return node

        graph.add_node(
            cfg["name"],
            _make_node(agent, task_template, output_key, cfg["name"], cfg["agent_id"]),
            NodeType.AGENT,
            {"id": cfg["agent_id"], "model": settings.model},
        )

    # 连线：线性流水线
    graph.set_entry_point(PIPELINE_ORDER[0])
    for i in range(len(PIPELINE_ORDER) - 1):
        graph.add_edge(PIPELINE_ORDER[i], PIPELINE_ORDER[i + 1])
    graph.set_finish_point(PIPELINE_ORDER[-1])

    return graph.compile()


async def run_travel(request: str) -> dict:
    """执行旅行计划流水线，返回最终状态。"""
    settings = load_settings()
    print(f"使用模型：{settings.model}"
          + (f"（base_url={settings.base_url}）" if settings.base_url else ""))

    compiled = build_travel_graph(settings)

    initial_state = {
        "user_request": request,
        "events": [],
        "messages": [],
    }

    return await compiled.ainvoke(initial_state)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
DEFAULT_REQUEST = "我和朋友 2 人，8 月从上海去成都 5 天，预算 8000，希望美食和轻松行程"


def main() -> None:
    parser = argparse.ArgumentParser(description="旅行计划多 Agent 流水线（基于 engine 框架）")
    parser.add_argument("--request", default=DEFAULT_REQUEST, help="自然语言旅行需求描述")
    args = parser.parse_args()

    state = asyncio.run(run_travel(args.request))

    print("\n" + "=" * 60)
    print("最终旅行计划")
    print("=" * 60)
    print(state.get("final_plan", "（未生成）"))
    print()

    print("=" * 60)
    print("各 Agent 执行记录（含工具调用）")
    print("=" * 60)
    for evt in state.get("events", []):
        tools = evt.get("tools") or []
        tools_desc = "，调用工具：" + ", ".join(tools) if tools else "，未调用工具"
        print(f"  [{evt.get('agent', '?')}]{tools_desc}")


if __name__ == "__main__":
    main()
