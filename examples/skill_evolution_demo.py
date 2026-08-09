"""可直接运行的技能闭环示例：发布种子技能、主图调用、轨迹提取和审批发布。"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from engine import (
    Orchestrator,
    Skill,
    SkillEvolutionService,
    SkillManifest,
    SkillRepository,
    SkillRetriever,
    SkillStatus,
    SkillTraceStore,
)


async def main() -> None:
    workspace = Path(tempfile.mkdtemp(prefix="engine-skill-demo-"))
    repository = SkillRepository(workspace / "skill_repo")
    traces = SkillTraceStore(workspace / "runs")

    seed = Skill(
        SkillManifest(
            skill_id="travel-budget",
            name="旅行预算核验",
            version="1.0.0",
            status=SkillStatus.VALIDATED,
            task_types=["travel"],
            tags=["旅行", "预算"],
            applicable_nodes=["planner"],
        ),
        "# 旅行预算核验\n\n汇总交通、住宿和活动费用，最后断言总额不超过用户预算。",
    )
    repository.save(seed)
    repository.publish("travel-budget", "1.0.0", approved_by="demo-owner")

    orch = Orchestrator()
    orch.set_skill_retriever(SkillRetriever(repository, min_score=0.1))
    orch.set_skill_trace_store(traces)
    planner = orch.create_agent("planner", config={"task_type": "travel"})
    orch.set_entry(planner)
    state = await orch.build_graph().ainvoke(
        {"run_id": "demo-run", "task_type": "travel", "goal": "制定旅行预算", "input": "预算 5000 元"}
    )

    evolution = SkillEvolutionService(repository, traces)
    candidate = evolution.create_candidate_from_runs(
        skill_id="travel-runbook",
        name="旅行执行规程",
        run_ids=["demo-run"],
        version="1.0.0",
        task_types=["travel"],
        tags=["旅行"],
    )

    def evaluate(skill):
        return {
            "score": 0.60 if skill is None else 0.85,
            "safety_passed": True,
            "regression_passed": True,
        }

    report = evolution.validate_candidate("travel-runbook", "1.0.0", evaluate)
    if report.accepted:
        evolution.publish("travel-runbook", "1.0.0", approved_by="demo-owner")

    print("Injected skills:", state.get("__skill_context__"))
    print("Candidate:", candidate.manifest.skill_id, report.to_dict())
    print("Artifacts:", workspace)


if __name__ == "__main__":
    asyncio.run(main())
