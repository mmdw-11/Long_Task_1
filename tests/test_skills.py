"""过程性技能闭环测试：仓库、检索、主图注入、脱敏轨迹和受控发布。"""

import json

import pytest

from engine import (
    SKILL_CONTEXT_KEY,
    Orchestrator,
    Skill,
    SkillEvolutionService,
    SkillManifest,
    SkillRepository,
    SkillRetriever,
    SkillStatus,
    SkillTraceStore,
)


def _validated_skill(skill_id="travel-budget", version="1.0.0"):
    return Skill(
        SkillManifest(
            skill_id=skill_id,
            name="旅行预算核验",
            version=version,
            status=SkillStatus.VALIDATED,
            description="旅行规划和预算检查",
            task_types=["travel"],
            tags=["旅行", "预算"],
            applicable_nodes=["planner"],
        ),
        "# 旅行预算核验\n\n先汇总全部费用，再验证总额不超过预算。",
    )


def test_repository_requires_validation_and_approval_before_publish(tmp_path):
    repository = SkillRepository(tmp_path / "skills")
    skill = _validated_skill()
    repository.save(skill)

    with pytest.raises(ValueError):
        repository.publish(skill.manifest.skill_id, skill.manifest.version, approved_by="")

    published = repository.publish(skill.manifest.skill_id, skill.manifest.version, approved_by="reviewer")
    assert published.manifest.status == SkillStatus.PUBLISHED
    assert repository.get("travel-budget").manifest.approved_by == "reviewer"
    assert repository.rollback("travel-budget", "1.0.0", approved_by="operator").manifest.version == "1.0.0"

    retired = repository.retire("travel-budget", reason="dependency changed")
    assert retired.manifest.status == SkillStatus.RETIRED
    with pytest.raises(KeyError):
        repository.get("travel-budget")


def test_retriever_filters_node_and_injects_with_budget(tmp_path):
    repository = SkillRepository(tmp_path / "skills")
    repository.save(_validated_skill())
    repository.publish("travel-budget", "1.0.0", approved_by="reviewer")
    retriever = SkillRetriever(repository, top_k=2, min_score=0.1, max_chars=1000)

    state = {"goal": "为旅行计划检查预算", "task_type": "travel"}
    matches = retriever.inject(state, node="planner", metadata={})

    assert [item.skill.manifest.skill_id for item in matches] == ["travel-budget"]
    assert state[SKILL_CONTEXT_KEY][0]["version"] == "1.0.0"
    assert "先汇总全部费用" in state["__skill_context_text__"]
    assert retriever.retrieve("检查旅行预算", node="writer", task_type="travel") == []


@pytest.mark.asyncio
async def test_orchestrator_injects_skill_and_records_redacted_trace(tmp_path):
    repository = SkillRepository(tmp_path / "skills")
    repository.save(_validated_skill())
    repository.publish("travel-budget", "1.0.0", approved_by="reviewer")
    traces = SkillTraceStore(tmp_path / "runs")

    orch = Orchestrator()
    orch.set_skill_retriever(SkillRetriever(repository, min_score=0.1))
    orch.set_skill_trace_store(traces)
    planner = orch.create_agent("planner", config={"task_type": "travel"})
    orch.set_entry(planner)

    state = await orch.build_graph().ainvoke(
        {
            "run_id": "skill-run",
            "task_type": "travel",
            "goal": "规划旅行并检查预算",
            "input": "api_key=abcdefghijklmnop 请制定计划",
        }
    )

    assert state[SKILL_CONTEXT_KEY][0]["skill_id"] == "travel-budget"
    assert "旅行预算核验" in state["planner"]
    events = traces.read("skill-run")
    assert {event["event_type"] for event in events} >= {"step_start", "node_start", "node_end"}
    serialized = json.dumps(events, ensure_ascii=False)
    assert "abcdefghijklmnop" not in serialized
    assert "REDACTED" in serialized
    node_start = next(event for event in events if event["event_type"] == "node_start")
    assert node_start["skills"][0]["skill_id"] == "travel-budget"


def test_evolution_generates_validates_publishes_and_rejects_regression(tmp_path):
    repository = SkillRepository(tmp_path / "skills")
    traces = SkillTraceStore(tmp_path / "runs")
    traces.record(
        "node_end",
        state={"run_id": "source-1", "task_type": "travel", "__evaluation__": {"passed": True}},
        node="planner",
        step=1,
        payload={"result": "ok"},
    )
    service = SkillEvolutionService(repository, traces, minimum_score_gain=0.05)
    candidate = service.create_candidate_from_runs(
        skill_id="travel-flow",
        name="旅行流程",
        run_ids=["source-1"],
        version="1.0.0",
        task_types=["travel"],
        tags=["旅行"],
    )
    assert "`planner`" in candidate.content

    def passing_evaluator(skill):
        if skill is None:
            return {"score": 0.50, "safety_passed": True, "regression_passed": True}
        return {"score": 0.80, "safety_passed": True, "regression_passed": True}

    report = service.validate_candidate("travel-flow", "1.0.0", passing_evaluator)
    assert report.accepted is True
    published = service.publish("travel-flow", "1.0.0", approved_by="owner")
    assert published.manifest.status == SkillStatus.PUBLISHED

    traces.record("node_end", state={"run_id": "source-2"}, node="planner", payload={"result": "ok"})
    candidate_2 = service.create_candidate_from_runs(
        skill_id="another-flow",
        name="另一个流程",
        run_ids=["source-2"],
        version="1.0.0",
    )
    assert candidate_2.manifest.status == SkillStatus.CANDIDATE

    def failing_evaluator(skill):
        return {
            "score": 0.7 if skill is None else 0.9,
            "safety_passed": skill is None,
            "regression_passed": True,
            "findings": ["unsafe action"] if skill else [],
        }

    rejected = service.validate_candidate("another-flow", "1.0.0", failing_evaluator)
    assert rejected.accepted is False
    assert (repository.root_dir / "rejected_edits.jsonl").exists()
    assert repository.get("another-flow", "1.0.0", status=SkillStatus.REJECTED).manifest.status == SkillStatus.REJECTED


def test_candidate_update_enforces_edit_budget(tmp_path):
    repository = SkillRepository(tmp_path / "skills")
    traces = SkillTraceStore(tmp_path / "runs")
    base = Skill(
        SkillManifest("stable", "稳定技能", "1.0.0", SkillStatus.VALIDATED),
        "# Stable\n\n" + "a" * 500,
    )
    repository.save(base)
    repository.publish("stable", "1.0.0", approved_by="owner")
    traces.record("node_end", state={"run_id": "source"}, node="worker", payload={"ok": True})
    service = SkillEvolutionService(repository, traces, max_edit_ratio=0.1, max_edit_chars=30)

    with pytest.raises(ValueError, match="exceeds budget"):
        service.create_candidate_from_runs(
            skill_id="stable",
            name="稳定技能",
            run_ids=["source"],
            version="1.1.0",
            content="# Completely replaced\n" + "b" * 500,
        )
