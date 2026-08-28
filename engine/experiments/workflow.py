"""小型 workflow 编排实验 runner。

本实验使用项目 Orchestrator 真实构图和执行，覆盖顺序、扇出、条件分支、父子 Agent、
循环五类结构。它用于证明“前端创建 AgentSpec -> 后端构图 -> 节点执行 -> 轨迹可检查”
这条编排链路能稳定工作。
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List

from engine import END, Orchestrator, StateGraph, add_messages

from .reports import ExperimentReport
from .types import ExperimentRow, WorkflowExample


@dataclass
class WorkflowExperimentConfig:
    """编排实验配置。"""

    recursion_limit: int = 10


async def run_workflow_experiment(
    examples: List[WorkflowExample],
    config: WorkflowExperimentConfig | None = None,
) -> ExperimentReport:
    """运行 300 条小型 workflow 编排实验。"""
    cfg = config or WorkflowExperimentConfig()
    rows: List[ExperimentRow] = []
    for example in examples:
        started = time.perf_counter()
        state = await _run_one(example, cfg)
        elapsed_ms = (time.perf_counter() - started) * 1000
        agents = [str(item.get("agent")) for item in state.get("messages") or [] if isinstance(item, dict)]
        passed = _matches_expected(example.expected_agents, agents)
        sequence_ok = _matches_expected(example.expected_agents, agents)
        branch_ok = example.pattern != "conditional" or agents[-1:] == example.expected_agents[-1:]
        loop_ok = example.pattern != "loop" or agents.count("worker") == 3
        trace_complete = len(agents) == len(example.expected_agents) and all(agents)
        rows.append(
            ExperimentRow(
                id=example.id,
                passed=passed,
                score=1.0 if passed else 0.0,
                prediction=" -> ".join(agents),
                expected=" -> ".join(example.expected_agents),
                metrics={
                    "steps": len(agents),
                    "expected_steps": len(example.expected_agents),
                    "sequence_accuracy": 1 if sequence_ok else 0,
                    "branch_accuracy": 1 if branch_ok else 0,
                    "loop_stop_success": 1 if loop_ok else 0,
                    "trace_complete": 1 if trace_complete else 0,
                    "execution_ms": elapsed_ms,
                },
                metadata={"pattern": example.pattern, "branch": example.branch},
            )
        )
    return ExperimentReport(
        name="workflow-orchestration",
        rows=rows,
        metadata={"recursion_limit": cfg.recursion_limit},
    )


async def _run_one(example: WorkflowExample, cfg: WorkflowExperimentConfig) -> Dict:
    if example.pattern == "loop":
        return await _run_loop_graph(example, cfg)
    orch = Orchestrator()
    if example.pattern == "sequential":
        planner = orch.create_agent("planner")
        executor = orch.create_agent("executor")
        reviewer = orch.create_agent("reviewer")
        orch.connect(planner, executor)
        orch.connect(executor, reviewer)
        orch.set_entry(planner)
    elif example.pattern == "fanout":
        planner = orch.create_agent("planner")
        researcher = orch.create_agent("researcher")
        writer = orch.create_agent("writer")
        reviewer = orch.create_agent("reviewer")
        orch.connect(planner, researcher)
        orch.connect(planner, writer)
        orch.connect(researcher, reviewer)
        orch.connect(writer, reviewer)
        orch.set_entry(planner)
    elif example.pattern == "conditional":
        router = orch.create_agent("router")
        left = orch.create_agent("left_worker")
        right = orch.create_agent("right_worker")
        orch.connect_conditional(router, "branch", {"L": left, "R": right})
        orch.set_entry(router)
    elif example.pattern == "parent_child":
        parent = orch.create_agent("parent")
        orch.add_sub_agent(parent, name="child_a")
        orch.add_sub_agent(parent, name="child_b")
        orch.set_entry(parent)
    else:
        raise ValueError(f"unsupported workflow pattern: {example.pattern}")
    return await orch.build_graph().ainvoke(
        {"input": example.prompt, "branch": example.branch},
        recursion_limit=cfg.recursion_limit,
    )


async def _run_loop_graph(example: WorkflowExample, cfg: WorkflowExperimentConfig) -> Dict:
    graph = StateGraph(schema={"messages": add_messages})

    async def worker(state: Dict) -> Dict:
        count = int(state.get("count") or 0) + 1
        return {
            "count": count,
            "messages": [{"agent": "worker", "content": f"{example.prompt} #{count}"}],
        }

    graph.add_node("worker", worker)
    graph.set_entry_point("worker")
    graph.add_loop("worker", lambda state: int(state.get("count") or 0) < 3)
    return await graph.compile().ainvoke({"input": example.prompt, "count": 0}, recursion_limit=cfg.recursion_limit)


def _matches_expected(expected: List[str], actual: List[str]) -> bool:
    if not expected:
        return not actual
    if expected == actual:
        return True
    # fanout 的 researcher/writer 并发顺序由图执行顺序决定，这里允许中间两个节点交换。
    if len(expected) == len(actual) and set(expected) == set(actual):
        return expected[0] == actual[0] and expected[-1] == actual[-1]
    return False
