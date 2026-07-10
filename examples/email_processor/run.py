"""邮件处理编排示例：edge + conditional_edge + loop。

这个示例不依赖 AgentScope / OpenAI，只使用普通 Python 函数模拟工具执行，
方便在 Python 3.10 环境下观察图编排规则：

    load_email -> analyze_email -> classify_email -> decide_actions
        -> draft_reply / create_calendar_event / escalate / finalize_email
        -> finalize_email -> load_email ... -> END

运行：
    $env:PYTHONPATH="."
    python examples\email_processor\run.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from engine import END, StateGraph, append_reducer


SAMPLE_EMAILS = [
    {
        "id": "mail-001",
        "from": "client@example.com",
        "subject": "明天下午能否安排一次项目评审会？",
        "body": "我们想在明天下午 3 点评审 Agent 编排方案，请帮忙确认并发一个会议邀请。",
    },
    {
        "id": "mail-002",
        "from": "newsletter@example.com",
        "subject": "本周 AI 行业简报",
        "body": "这是一封订阅简报，包含若干行业新闻，不需要回复。",
    },
    {
        "id": "mail-003",
        "from": "vip@example.com",
        "subject": "紧急：线上环境出现异常",
        "body": "客户反馈生产环境有严重问题，请尽快安排负责人介入并回复处理计划。",
    },
]


# --------------------------------------------------------------------------- #
# 模拟工具
# --------------------------------------------------------------------------- #
def inspect_email(email: Dict[str, str]) -> Dict[str, Any]:
    """读取邮件并提取简单信号。"""
    text = f"{email['subject']} {email['body']}"
    explicit_no_reply = any(word in text for word in ["不需要回复", "无需回复"])
    return {
        "mentions_meeting": any(word in text for word in ["会议", "评审", "安排", "邀请"]),
        "urgent": any(word in text for word in ["紧急", "严重", "尽快", "异常"]),
        "newsletter": any(word in text for word in ["简报", "订阅", "新闻"]),
        "needs_reply": (
            not explicit_no_reply
            and any(word in text for word in ["确认", "回复", "处理计划"])
        ),
    }


def classify(signals: Dict[str, Any]) -> Dict[str, str]:
    """根据提取信号给邮件分类和分级。"""
    if signals["urgent"]:
        return {"category": "incident", "priority": "P0"}
    if signals["mentions_meeting"]:
        return {"category": "meeting", "priority": "P1"}
    if signals["newsletter"]:
        return {"category": "newsletter", "priority": "P3"}
    return {"category": "general", "priority": "P2"}


def plan_actions(email: Dict[str, str], classification: Dict[str, str], signals: Dict[str, Any]) -> List[str]:
    """根据分类结果决定后续工具动作。"""
    actions: List[str] = []
    if signals["needs_reply"] or classification["priority"] in {"P0", "P1"}:
        actions.append("reply")
    if classification["category"] == "meeting":
        actions.append("calendar")
    if classification["priority"] == "P0":
        actions.append("escalate")
    return actions


def make_reply(email: Dict[str, str], classification: Dict[str, str]) -> str:
    """模拟自动回复草稿。"""
    if classification["category"] == "incident":
        return "已收到紧急问题，我们会立即升级处理，并在 30 分钟内同步初步排查计划。"
    if classification["category"] == "meeting":
        return "收到，我会确认明天下午 3 点的评审安排，并补充会议邀请。"
    return "收到，我会尽快处理。"


def make_calendar_event(email: Dict[str, str]) -> Dict[str, str]:
    """模拟创建日程。"""
    return {
        "title": email["subject"],
        "time": "明天 15:00",
        "attendees": email["from"],
    }


def make_escalation(email: Dict[str, str], classification: Dict[str, str]) -> Dict[str, str]:
    """模拟升级人工处理。"""
    return {
        "owner": "oncall-engineer",
        "reason": f"{classification['priority']} {email['subject']}",
    }


# --------------------------------------------------------------------------- #
# 图节点
# --------------------------------------------------------------------------- #
def load_email(state: Dict[str, Any]) -> Dict[str, Any]:
    emails = state["emails"]
    index = state.get("email_index", 0)
    email = emails[index]
    return {
        "current_email": email,
        "current_actions": [],
        "action_index": 0,
        "messages": [f"读取邮件 {email['id']}: {email['subject']}"],
    }


def analyze_email(state: Dict[str, Any]) -> Dict[str, Any]:
    email = state["current_email"]
    signals = inspect_email(email)
    return {
        "analysis": signals,
        "messages": [f"分析 {email['id']}: {signals}"],
    }


def classify_email(state: Dict[str, Any]) -> Dict[str, Any]:
    classification = classify(state["analysis"])
    return {
        "classification": classification,
        "messages": [
            f"分类 {state['current_email']['id']}: "
            f"{classification['category']} / {classification['priority']}"
        ],
    }


def decide_actions(state: Dict[str, Any]) -> Dict[str, Any]:
    actions = plan_actions(
        state["current_email"],
        state["classification"],
        state["analysis"],
    )
    return {
        "current_actions": actions,
        "action_index": 0,
        "messages": [f"动作决策 {state['current_email']['id']}: {actions or ['no_action']}"],
    }


def draft_reply(state: Dict[str, Any]) -> Dict[str, Any]:
    email = state["current_email"]
    reply = make_reply(email, state["classification"])
    return {
        "action_index": state["action_index"] + 1,
        "actions_log": [{"email_id": email["id"], "type": "reply", "draft": reply}],
        "messages": [f"生成回复草稿 {email['id']}"],
    }


def create_calendar_event(state: Dict[str, Any]) -> Dict[str, Any]:
    email = state["current_email"]
    event = make_calendar_event(email)
    return {
        "action_index": state["action_index"] + 1,
        "actions_log": [{"email_id": email["id"], "type": "calendar", "event": event}],
        "messages": [f"创建日程 {email['id']}: {event['time']}"],
    }


def escalate_email(state: Dict[str, Any]) -> Dict[str, Any]:
    email = state["current_email"]
    escalation = make_escalation(email, state["classification"])
    return {
        "action_index": state["action_index"] + 1,
        "actions_log": [{"email_id": email["id"], "type": "escalate", "ticket": escalation}],
        "messages": [f"升级人工处理 {email['id']}: {escalation['owner']}"],
    }


def finalize_email(state: Dict[str, Any]) -> Dict[str, Any]:
    email = state["current_email"]
    processed = {
        "id": email["id"],
        "from": email["from"],
        "subject": email["subject"],
        "classification": state["classification"],
        "actions": list(state.get("current_actions", [])),
    }
    return {
        "email_index": state.get("email_index", 0) + 1,
        "processed_emails": [processed],
        "messages": [f"完成邮件 {email['id']}"],
    }


# --------------------------------------------------------------------------- #
# 条件边路由函数
# --------------------------------------------------------------------------- #
def next_action(state: Dict[str, Any]) -> str:
    actions = state.get("current_actions", [])
    index = state.get("action_index", 0)
    if index >= len(actions):
        return "finalize"
    return actions[index]


def more_emails(state: Dict[str, Any]) -> str:
    return "more" if state.get("email_index", 0) < len(state.get("emails", [])) else "done"


def build_email_graph():
    schema = {
        "messages": append_reducer,
        "actions_log": append_reducer,
        "processed_emails": append_reducer,
    }
    graph = StateGraph(schema=schema)

    graph.add_node("load_email", load_email)
    graph.add_node("analyze_email", analyze_email)
    graph.add_node("classify_email", classify_email)
    graph.add_node("decide_actions", decide_actions)
    graph.add_node("draft_reply", draft_reply)
    graph.add_node("create_calendar_event", create_calendar_event)
    graph.add_node("escalate_email", escalate_email)
    graph.add_node("finalize_email", finalize_email)

    graph.set_entry_point("load_email")
    graph.add_edge("load_email", "analyze_email")
    graph.add_edge("analyze_email", "classify_email")
    graph.add_edge("classify_email", "decide_actions")

    action_routes = {
        "reply": "draft_reply",
        "calendar": "create_calendar_event",
        "escalate": "escalate_email",
        "finalize": "finalize_email",
    }
    graph.add_conditional_edges("decide_actions", next_action, action_routes)
    graph.add_conditional_edges("draft_reply", next_action, action_routes)
    graph.add_conditional_edges("create_calendar_event", next_action, action_routes)
    graph.add_conditional_edges("escalate_email", next_action, action_routes)

    graph.add_conditional_edges(
        "finalize_email",
        more_emails,
        {"more": "load_email", "done": END},
    )
    compiled = graph.compile()
    compiled.recursion_limit = 100
    return compiled


async def run_email_processor(show_routes: bool = False) -> Dict[str, Any]:
    compiled = build_email_graph()
    initial_state = {
        "emails": SAMPLE_EMAILS,
        "email_index": 0,
        "messages": [],
        "actions_log": [],
        "processed_emails": [],
    }

    final_state: Dict[str, Any] = {}
    async for event in compiled.astream(initial_state):
        etype = event["type"]
        if etype == "node_start":
            print(f"\n> 节点开始: {event['node']}")
        elif etype == "node_end":
            update = event.get("update") or {}
            for msg in update.get("messages", []):
                print(f"  - {msg}")
        elif etype == "route" and show_routes:
            targets = ["END" if target == END else target for target in event["targets"]]
            print(
                f"  路由: {event['node']} -> "
                f"{targets or ['END']}"
            )
        elif etype == "final":
            final_state = event["state"]

    return final_state


def main() -> None:
    parser = argparse.ArgumentParser(description="本地邮件处理编排示例")
    parser.add_argument("--show-routes", action="store_true", help="显示每一步路由决策")
    args = parser.parse_args()

    state = asyncio.run(run_email_processor(show_routes=args.show_routes))

    print("\n" + "=" * 70)
    print("处理结果")
    print("=" * 70)
    print(json.dumps(state["processed_emails"], ensure_ascii=False, indent=2))

    print("\n" + "=" * 70)
    print("工具动作日志")
    print("=" * 70)
    print(json.dumps(state["actions_log"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
