"""Demo for real LLM-backed task drift detection.

Run:
    python examples/context_drift_llm_demo.py

The script reads .env and configs/context_policy.yaml by default. It expects
semantic_drift_mode to be llm or hybrid, and TASK_DRIFT_* / OPENAI_* model
settings to point at an OpenAI-compatible chat completions endpoint.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import ContextPolicy, OpenAITaskDriftJudge, StateGraph
from engine.hooks import HookManager


DEMO_LEDGER_ROOT = PROJECT_ROOT / "runs" / "context_drift_llm_demo"


def _load_dotenv() -> None:
    env_path = PROJECT_ROOT / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
        return
    except Exception:
        pass
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


async def _run_case(
    *,
    run_id: str,
    goal: str,
    current_plan: list[str],
    input_text: str,
) -> Dict[str, Any]:
    policy_path = Path(os.environ.get("CONTEXT_POLICY_PATH", "configs/context_policy.yaml"))
    if not policy_path.is_absolute():
        policy_path = PROJECT_ROOT / policy_path
    policy = ContextPolicy.from_file(policy_path)

    graph = StateGraph()

    async def worker(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": state["input"]}

    graph.add_node("drift_probe", worker)
    graph.set_entry_point("drift_probe")

    task_drift_judge = None
    if (policy.semantic_drift_mode or "off").lower() in {"llm", "hybrid"}:
        task_drift_judge = OpenAITaskDriftJudge()
    drift_detector = policy.build_drift_detector(task_drift_judge=task_drift_judge)

    compiled = graph.compile()
    compiled.hooks = HookManager(
        context_ledger=policy.build_ledger_store(DEMO_LEDGER_ROOT),
        context_budget=policy.build_budget_controller(),
        context_injector=policy.build_injector(),
        drift_detector=drift_detector,
    )

    detector = drift_detector
    detector_info = {
        "semantic_drift_mode": getattr(detector, "semantic_drift_mode", ""),
        "task_drift_judge": type(getattr(detector, "task_drift_judge", None)).__name__
        if getattr(detector, "task_drift_judge", None) is not None
        else None,
    }

    state = await compiled.ainvoke(
        {
            "run_id": run_id,
            "goal": goal,
            "current_plan": current_plan,
            "input": input_text,
        }
    )
    return {
        "detector": detector_info,
        "drift": state.get("__context_drift__", {}),
    }


async def main() -> Dict[str, Any]:
    _load_dotenv()
    if DEMO_LEDGER_ROOT.exists():
        shutil.rmtree(DEMO_LEDGER_ROOT)
    results = {
        "on_task": await _run_case(
            run_id="llm-drift-on-task",
            goal="改进语义漂移检测，用真实 LLM 判断任务目标是否跑偏，并补充文档。",
            current_plan=[
                "配置真实 LLM judge",
                "检查上下文策略配置",
                "补充漂移检测文档",
            ],
            input_text="检查 configs/context_policy.yaml 是否已经打开 semantic_drift_mode。",
        ),
        "off_task": await _run_case(
            run_id="llm-drift-off-task",
            goal="改进语义漂移检测，用真实 LLM 判断任务目标是否跑偏，并补充文档。",
            current_plan=[
                "配置真实 LLM judge",
                "检查上下文策略配置",
                "补充漂移检测文档",
            ],
            input_text="写一份周末旅行攻略，推荐酒店和餐厅，并聊聊拍照路线。",
        ),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return results


if __name__ == "__main__":
    asyncio.run(main())
