"""端-边-云异构资源自适应调度示例。

运行：
    python examples\edge_cloud_scheduler\run.py
    python examples\edge_cloud_scheduler\run.py --gate llm
    python examples\edge_cloud_scheduler\run.py --gate llm --allow-fallback

默认 gate 为本地启发式规则；传入 ``--gate llm`` 时会调用 .env 中配置的
OpenAI/兼容模型生成任务画像，失败时默认直接报错。真实系统可在节点内部读取
``__resource_allocation__`` 后转发到对应推理后端。
"""

from __future__ import annotations

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from engine import (
    AdaptiveResourceScheduler,
    END,
    HeuristicTaskGate,
    OpenAITaskGate,
    ExecutorRegistry,
    InferenceRequest,
    InferenceResult,
    MutableResourceMonitor,
    ResilientInferenceRunner,
    ResourceRequest,
    ResourceStatus,
    ResourceTier,
    StateGraph,
    StaticResourceMonitor,
    append_reducer,
)
from engine.hooks import HookManager, RESOURCE_ALLOCATION_KEY
from engine.hooks import AUDIT_PACK_KEY, REDACTION_RESULT_KEY


EXECUTOR_REGISTRY = ExecutorRegistry.default()


class DemoCloudFailsOnce:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, request: InferenceRequest) -> InferenceResult:
        self.calls += 1
        if self.calls == 1:
            return InferenceResult(
                text="",
                executor=type(self).__name__,
                endpoint=request.allocation["endpoint"],
                success=False,
                error="429 rate limit",
                retryable=True,
            )
        return InferenceResult(
            text="[cloud-demo] recovered",
            executor=type(self).__name__,
            endpoint=request.allocation["endpoint"],
            model="demo-cloud",
        )


class DemoEdgeOk:
    def run(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            text="[edge-demo] " + request.prompt[:160],
            executor=type(self).__name__,
            endpoint=request.allocation["endpoint"],
            model="demo-edge",
        )


def make_task_node(label: str, request_text: str):
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        allocation = state[RESOURCE_ALLOCATION_KEY]
        decision = allocation["metadata"]["decision"]
        item = {
            "task": label,
            "tier": allocation["tier"],
            "endpoint": allocation["endpoint"],
            "reason": decision["reason"],
            "profile": decision["profile"],
            "model_split": decision["model_split"],
        }
        if state.get("show_trace"):
            trace = allocation["metadata"].get("trace")
            if trace:
                item["trace"] = {
                    "heuristic_profile": trace["heuristic_profile"],
                    "gate_profile": trace["gate_profile"],
                    "policy_hits": trace["policy_hits"],
                    "task_text": trace["task_text"],
                    "metadata": trace["metadata"],
                }
        if state.get("show_security"):
            if REDACTION_RESULT_KEY in state:
                item["redaction"] = state[REDACTION_RESULT_KEY]["summary"]
            if AUDIT_PACK_KEY in state:
                item["audit_pack"] = state[AUDIT_PACK_KEY]
        if state.get("resilient_execute"):
            runner = state["resilient_runner"]
            result = runner.run(
                resource_request=ResourceRequest(
                    node=item["task"],
                    state={**state, "input": request_text},
                    metadata={"description": request_text},
                ),
                prompt=request_text,
                metadata={"task": label, "node": item["task"]},
            )
            item["execution"] = result.to_dict()
        elif state.get("execute_inference"):
            prompt = request_text
            redacted_payload = None
            if REDACTION_RESULT_KEY in state:
                redacted_payload = state[REDACTION_RESULT_KEY].get("redacted")
                prompt = (
                    "请基于以下已脱敏的可信工作区 payload 执行任务，不要还原敏感值：\n"
                    + json.dumps(redacted_payload, ensure_ascii=False, indent=2)
                )
            result = EXECUTOR_REGISTRY.run(
                InferenceRequest(
                    prompt=prompt,
                    allocation=allocation,
                    redacted_payload=redacted_payload,
                    metadata={"task": label, "node": item["task"]},
                )
            )
            item["execution"] = result.to_dict()
        return {"decisions": [item]}

    return node


TASKS = [
    {
        "node": "quick_local_reply",
        "label": "低延迟本地回复",
        "request": "用户正在等待一句简短确认，请立即回复，不涉及隐私。",
    },
    {
        "node": "private_mail_analysis",
        "label": "敏感邮件分析",
        "request": "分析一封包含客户姓名、合同细节和内部邮件线程的复杂投诉邮件，给出处理建议。",
    },
    {
        "node": "public_deep_report",
        "label": "公开长文深度分析",
        "request": "对一份公开发布的 3 万字行业报告做深度总结、风险归纳和多维对比。",
    },
    {
        "node": "secret_key_triage",
        "label": "密钥泄露排查",
        "request": "紧急检查日志里出现的 api_key、token 和私钥片段，判断是否需要隔离并生成审计摘要。",
    },
]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="画像构建方式：heuristic 为本地规则，llm 为真实 OpenAI/兼容模型调用。",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="仅 gate=llm 时生效：模型调用失败后退回本地启发式画像。",
    )
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="输出 heuristic/gate 画像对照与策略命中记录。",
    )
    parser.add_argument(
        "--show-security",
        action="store_true",
        help="输出脱敏摘要和审计包。",
    )
    parser.add_argument(
        "--inject-secret",
        action="store_true",
        help="在全局输入中注入测试凭据，用于观察脱敏和审计包。",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="按调度结果调用对应 executor；cloud 会真实请求 .env 配置的模型服务。",
    )
    parser.add_argument(
        "--resilient-execute",
        action="store_true",
        help="使用闭环执行器：执行失败会反馈资源状态并重新调度。",
    )
    parser.add_argument(
        "--cloud-fail-once",
        action="store_true",
        help="演示用：让云端第一次执行返回 429，观察自动回退到边缘。",
    )
    parser.add_argument(
        "--cloud-rate-limited",
        action="store_true",
        help="模拟云端限流，观察高复杂任务回退到边缘。",
    )
    parser.add_argument(
        "--edge-down",
        action="store_true",
        help="模拟边缘不可用，观察敏感任务回退到端侧或审计路径。",
    )
    args = parser.parse_args()

    graph = StateGraph(schema={"decisions": append_reducer})
    for task in TASKS:
        graph.add_node(
            task["node"],
            make_task_node(task["label"], task["request"]),
            metadata={
                "description": task["request"],
            },
        )

    graph.set_entry_point(TASKS[0]["node"])
    for current, nxt in zip(TASKS, TASKS[1:]):
        graph.add_edge(current["node"], nxt["node"])
    graph.add_edge(TASKS[-1]["node"], END)

    compiled = graph.compile()
    gate = (
        OpenAITaskGate(allow_fallback=args.allow_fallback)
        if args.gate == "llm"
        else HeuristicTaskGate()
    )
    statuses = []
    if args.cloud_rate_limited:
        statuses.append(ResourceStatus(ResourceTier.CLOUD, available=True, rate_limited=True))
    if args.edge_down:
        statuses.append(ResourceStatus(ResourceTier.EDGE, available=False))
    if args.resilient_execute:
        monitor = MutableResourceMonitor(statuses)
    else:
        monitor = StaticResourceMonitor(statuses) if statuses else None
    scheduler = AdaptiveResourceScheduler(gate=gate, monitor=monitor)
    compiled.hooks = HookManager(scheduler=scheduler)
    resilient_registry = EXECUTOR_REGISTRY
    if args.cloud_fail_once:
        resilient_registry = ExecutorRegistry(
            {
                ResourceTier.CLOUD: DemoCloudFailsOnce(),
                ResourceTier.EDGE: DemoEdgeOk(),
                ResourceTier.DEVICE: DemoEdgeOk(),
            }
        )
    resilient_runner = ResilientInferenceRunner(
        scheduler=scheduler,
        registry=resilient_registry,
        max_attempts=3,
    )
    input_text = "演示端边云资源调度"
    if args.inject_secret or args.show_security:
        input_text += "；测试凭据 api_key=sk-demoSECRET123456 和联系邮箱 owner@example.com"
    state = await compiled.ainvoke({
        "input": input_text,
        "show_trace": args.show_trace,
        "show_security": args.show_security,
        "execute_inference": args.execute,
        "resilient_execute": args.resilient_execute,
        "resilient_runner": resilient_runner,
    })
    print(json.dumps(state["decisions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
