"""End-to-end demo for context ledger, validation control, and graph resume.

Run:
    python examples/context_end_to_end_demo.py

Outputs are written to ``runs/context_demo``.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import ContextPolicy, GraphCheckpointStore, StateGraph
from engine.hooks import HookManager


OUTPUT_ROOT = Path("runs/context_demo")


async def _run_budget_pause() -> Dict[str, Any]:
    policy = ContextPolicy(
        max_context_tokens=45,
        reserved_output_tokens=10,
        long_text_threshold=120,
        summary_max_chars=80,
        repeat_node_limit=2,
    )
    ledger_store = policy.build_ledger_store(OUTPUT_ROOT / "ledger")

    graph = StateGraph()

    async def worker(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"final_answer": "this should not execute when budget pauses"}

    graph.add_node("budget_guarded_worker", worker)
    graph.set_entry_point("budget_guarded_worker")
    compiled = graph.compile()
    compiled.hooks = HookManager(
        context_ledger=ledger_store,
        context_budget=policy.build_budget_controller(),
        context_injector=policy.build_injector(),
        drift_detector=policy.build_drift_detector(),
    )

    return await compiled.ainvoke(
        {
            "run_id": "demo-budget-paused",
            "goal": "Produce a long Challenge Cup implementation report "
            + ("with strict context preservation. " * 20),
            "hard_constraints": [
                "Do not merge invalid node output.",
                "Pause before execution if context budget is exceeded.",
            ],
            "current_plan": [
                "collect context",
                "run budget guard",
                "resume only after compression or policy change",
            ],
            "input": "budget pressure " * 40,
        }
    )


async def _run_validation_reroute_and_resume() -> Dict[str, Any]:
    policy = ContextPolicy(
        max_context_tokens=5000,
        reserved_output_tokens=50,
        long_text_threshold=120,
        summary_max_chars=90,
        repeat_node_limit=2,
    )
    ledger_store = policy.build_ledger_store(OUTPUT_ROOT / "ledger")
    checkpoint_store = GraphCheckpointStore(OUTPUT_ROOT / "graph_checkpoints")

    graph = StateGraph()

    async def collect_requirements(state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "requirements_summary": "Long task requires ledger, MEMORY.md, budget, drift, validation and resume evidence. "
            * 4,
        }

    async def draft_answer(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"bad_result": "missing final_answer so validation must reroute"}

    async def repair_answer(state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "final_answer": "Recovered final answer with verified context ledger and checkpoint resume.",
            "__parent_validated__": True,
        }

    graph.add_node("collect_requirements", collect_requirements)
    graph.add_node(
        "draft_answer",
        draft_answer,
        metadata={
            "required_update_keys": ["final_answer"],
            "validation_failure_action": "reroute",
            "validation_failure_targets": ["repair_answer"],
        },
    )
    graph.add_node(
        "repair_answer",
        repair_answer,
        metadata={"required_update_keys": ["final_answer"]},
    )
    graph.set_entry_point("collect_requirements")
    graph.add_edge("collect_requirements", "draft_answer")

    compiled = graph.compile()
    compiled.hooks = HookManager(
        context_ledger=ledger_store,
        context_budget=policy.build_budget_controller(),
        context_injector=policy.build_injector(),
        drift_detector=policy.build_drift_detector(),
    )

    run_input = {
        "run_id": "demo-validation-reroute",
        "goal": "Complete a long autonomous-agent task with auditable context handling.",
        "hard_constraints": [
            "Invalid draft output must not enter GraphState.",
            "Validation failure must reroute to a repair node.",
            "Resume must load step, frontier and GraphState from checkpoint.",
        ],
        "current_plan": [
            "collect requirements",
            "draft answer",
            "validate",
            "repair if needed",
            "finish with checkpoint evidence",
        ],
        "input": "challenge cup context-management demo",
    }
    first_pass_state = await compiled.ainvoke(
        run_input,
        checkpoint_store=checkpoint_store,
    )

    resume_checkpoint = checkpoint_store.load("demo-validation-reroute", "step_0001_before")
    resume_policy = ContextPolicy(
        max_context_tokens=5000,
        reserved_output_tokens=50,
        long_text_threshold=120,
        summary_max_chars=90,
        repeat_node_limit=2,
    )
    resume_compiled = graph.compile()
    resume_compiled.hooks = HookManager(
        context_ledger=resume_policy.build_ledger_store(OUTPUT_ROOT / "ledger_resumed"),
        context_budget=resume_policy.build_budget_controller(),
        context_injector=resume_policy.build_injector(),
        drift_detector=resume_policy.build_drift_detector(),
    )
    resumed_state = await resume_compiled.ainvoke(
        checkpoint_store=GraphCheckpointStore(OUTPUT_ROOT / "graph_checkpoints_resumed"),
        resume_from=resume_checkpoint,
    )

    return {
        "first_pass": first_pass_state,
        "resumed_from": resume_checkpoint.to_dict(),
        "resumed": resumed_state,
    }


def _write_summary(result: Dict[str, Any]) -> Path:
    summary_path = OUTPUT_ROOT / "demo_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary_path


async def main() -> Dict[str, Any]:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    budget_paused = await _run_budget_pause()
    validation_resume = await _run_validation_reroute_and_resume()
    result = {
        "budget_paused": {
            "run_status": budget_paused.get("__run_status__"),
            "pause_reason": budget_paused.get("__pause_reason__"),
            "messages_executed": "messages" in budget_paused,
            "memory_md": str(OUTPUT_ROOT / "ledger" / "demo-budget-paused" / "MEMORY.md"),
        },
        "validation_reroute_resume": {
            "bad_result_merged": "bad_result" in validation_resume["first_pass"],
            "final_answer": validation_resume["first_pass"].get("final_answer"),
            "resumed_final_answer": validation_resume["resumed"].get("final_answer"),
            "resumed_from_step": validation_resume["resumed_from"]["step"],
            "resumed_from_frontier": validation_resume["resumed_from"]["frontier"],
            "resumed_state_keys": sorted(validation_resume["resumed_from"]["state"].keys()),
            "memory_md": str(OUTPUT_ROOT / "ledger" / "demo-validation-reroute" / "MEMORY.md"),
            "resumed_memory_md": str(
                OUTPUT_ROOT / "ledger_resumed" / "demo-validation-reroute" / "MEMORY.md"
            ),
            "checkpoint_final": str(
                OUTPUT_ROOT
                / "graph_checkpoints"
                / "demo-validation-reroute"
                / "final.json"
            ),
        },
    }
    _write_summary(result)
    return result


if __name__ == "__main__":
    print(json.dumps(asyncio.run(main()), ensure_ascii=False, indent=2, sort_keys=True))
