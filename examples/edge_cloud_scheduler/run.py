"""端-边-云异构资源自适应调度示例。

运行：
    python examples\edge_cloud_scheduler\run.py

该示例不调用真实远程模型，只展示调度器如何依据任务画像选择 DEVICE / EDGE /
CLOUD，并给出模型切分计划。真实系统可在节点内部读取
``__resource_allocation__`` 后转发到对应推理后端。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from engine import AdaptiveResourceScheduler, END, StateGraph, append_reducer
from engine.hooks import HookManager, RESOURCE_ALLOCATION_KEY


def make_task_node(label: str):
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        allocation = state[RESOURCE_ALLOCATION_KEY]
        decision = allocation["metadata"]["decision"]
        return {
            "decisions": [
                {
                    "task": label,
                    "tier": allocation["tier"],
                    "endpoint": allocation["endpoint"],
                    "reason": decision["reason"],
                    "profile": decision["profile"],
                    "model_split": decision["model_split"],
                }
            ]
        }

    return node


async def main() -> None:
    graph = StateGraph(schema={"decisions": append_reducer})
    graph.add_node(
        "quick_local_reply",
        make_task_node("低延迟本地回复"),
        metadata={
            "realtime": "hard",
            "sensitivity": "internal",
            "complexity": "low",
        },
    )
    graph.add_node(
        "private_mail_analysis",
        make_task_node("敏感邮件分析"),
        metadata={
            "realtime": "normal",
            "sensitivity": "confidential",
            "complexity": "high",
            "task_type": "email",
        },
    )
    graph.add_node(
        "public_deep_report",
        make_task_node("公开长文深度分析"),
        metadata={
            "realtime": "batch",
            "sensitivity": "public",
            "complexity": "high",
        },
    )

    graph.set_entry_point("quick_local_reply")
    graph.add_edge("quick_local_reply", "private_mail_analysis")
    graph.add_edge("private_mail_analysis", "public_deep_report")
    graph.add_edge("public_deep_report", END)

    compiled = graph.compile()
    compiled.hooks = HookManager(scheduler=AdaptiveResourceScheduler())
    state = await compiled.ainvoke({"input": "演示端边云资源调度"})
    print(json.dumps(state["decisions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
