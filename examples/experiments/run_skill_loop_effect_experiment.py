"""Run a controlled skill-loop effect experiment.

This experiment is the second layer after lifecycle smoke tests. It compares
the same calendar task set before and after publishing a skill generated from a
successful run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from engine.modules.skills import SkillRepository, SkillTraceStore
from engine.modules.workflows import RunStore, WorkflowStore
from engine.server.app import create_app


DEFAULT_TASKS = [
    {
        "id": "calendar-001",
        "task": "安排一次项目同步会议，并记录确认方式",
        "expected_terms": ["会议", "确认", "日历"],
    },
    {
        "id": "calendar-002",
        "task": "安排一次需求评审会议，并提醒参会人准备材料",
        "expected_terms": ["会议", "提醒", "材料"],
    },
    {
        "id": "calendar-003",
        "task": "安排一次代码走查会议，并写出会前检查清单",
        "expected_terms": ["会议", "检查", "清单"],
    },
    {
        "id": "calendar-004",
        "task": "安排一次周会，并说明时间冲突时如何处理",
        "expected_terms": ["周会", "冲突", "处理"],
    },
    {
        "id": "calendar-005",
        "task": "安排一次产品复盘会议，并记录输出物",
        "expected_terms": ["会议", "复盘", "输出"],
    },
    {
        "id": "calendar-006",
        "task": "安排一次客户沟通会议，并标注风险点",
        "expected_terms": ["会议", "客户", "风险"],
    },
    {
        "id": "calendar-007",
        "task": "安排一次实验结果汇报会，并列出会前准备",
        "expected_terms": ["汇报", "准备", "实验"],
    },
    {
        "id": "calendar-008",
        "task": "安排一次接口联调会议，并指定负责人",
        "expected_terms": ["会议", "联调", "负责人"],
    },
    {
        "id": "calendar-009",
        "task": "安排一次论文讨论会议，并整理议题",
        "expected_terms": ["会议", "论文", "议题"],
    },
    {
        "id": "calendar-010",
        "task": "安排一次迭代计划会议，并记录验收标准",
        "expected_terms": ["会议", "计划", "验收"],
    },
]


SEED_TASK = (
    "安排一次日历会议并沉淀成功步骤：读取参与人日历、检查时间冲突、"
    "创建提醒、输出确认清单。"
)

SKILL_GUIDANCE_TERMS = ["可复用技能", "读取参与人日历", "检查时间冲突", "创建提醒", "确认清单"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/experiments/skills/effect_calendar")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_dir)
    if args.fresh and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    client = _build_client(root)
    workflow_id = _create_calendar_workflow(client)

    round1 = _run_round(client, workflow_id, DEFAULT_TASKS, round_name="round1_no_skill")
    seed_record = _run_one(client, workflow_id, "seed", SEED_TASK, round_name="seed_for_skill")
    skill = _publish_skill_from_run(client, seed_record["run_id"])
    search_check = client.post(
        "/api/skills/search",
        json={"query": "需要安排日历会议并检查时间冲突", "node": "calendar_agent"},
    ).json()
    round2 = _run_round(client, workflow_id, DEFAULT_TASKS, round_name="round2_with_skill")

    rows = round1 + [seed_record] + round2
    summary = _summarize(rows, skill=skill, search_check=search_check)
    _write_outputs(root, rows, summary)
    print(json.dumps({"summary": str(root / "summary.json"), "rows": str(root / "rows.jsonl"), "report": str(root / "report.md")}, ensure_ascii=False, indent=2))


def _build_client(root: Path) -> TestClient:
    app = create_app(
        workflow_store=WorkflowStore(root / "workflows"),
        run_store=RunStore(root / "runs"),
        skill_repository=SkillRepository(root / "skills"),
        skill_trace_store=SkillTraceStore(root / "skill_traces"),
    )
    return TestClient(app)


def _create_calendar_workflow(client: TestClient) -> str:
    agent_id = client.post(
        "/api/agents",
        json={
            "name": "calendar_agent",
            "description": "负责日历会议安排、冲突检查、提醒创建和确认清单输出。",
            "config": {"task_type": "calendar"},
        },
    ).json()["id"]
    client.post("/api/graph/entry", json={"agent_id": agent_id})
    workflow = client.post(
        "/api/workflows",
        json={"name": "calendar skill-loop workflow", "tags": ["calendar", "skill-loop"]},
    ).json()
    return str(workflow["id"])


def _publish_skill_from_run(client: TestClient, run_id: str) -> Dict[str, Any]:
    candidate = client.post(
        "/api/skills/candidates/from-run",
        json={
            "run_id": run_id,
            "name": "日历会议安排闭环技能",
            "description": "从成功日历会议安排 run 中沉淀的可复用流程。",
            "tags": ["日历", "会议", "冲突", "提醒", "确认"],
        },
    )
    candidate.raise_for_status()
    skill_id = candidate.json()["id"]
    validation = client.post(f"/api/skills/{skill_id}/validate")
    validation.raise_for_status()
    published = client.post(f"/api/skills/{skill_id}/publish", json={"approved_by": "experiment"})
    published.raise_for_status()
    return {
        "candidate": candidate.json(),
        "validation": validation.json(),
        "published": published.json(),
    }


def _run_round(
    client: TestClient,
    workflow_id: str,
    tasks: List[Dict[str, Any]],
    *,
    round_name: str,
) -> List[Dict[str, Any]]:
    return [
        _run_one(
            client,
            workflow_id,
            str(item["id"]),
            str(item["task"]),
            expected_terms=list(item.get("expected_terms") or []),
            round_name=round_name,
        )
        for item in tasks
    ]


def _run_one(
    client: TestClient,
    workflow_id: str,
    task_id: str,
    task: str,
    *,
    expected_terms: List[str] | None = None,
    round_name: str,
) -> Dict[str, Any]:
    started = time.perf_counter()
    created = client.post(
        "/api/runs",
        json={
            "workflow_id": workflow_id,
            "input": {
                "input": task,
                "task_type": "calendar",
                "task_id": task_id,
            },
        },
    )
    created.raise_for_status()
    run_id = str(created.json()["id"])
    record = client.get(f"/api/runs/{run_id}")
    record.raise_for_status()
    data = record.json()
    elapsed = time.perf_counter() - started
    traces = client.get(f"/api/runs/{run_id}/skill-traces").json().get("events", [])
    messages = data.get("state", {}).get("messages") or []
    content = str(messages[0].get("content") if messages else data.get("state", ""))
    expected_terms = expected_terms or []
    coverage = _coverage(expected_terms, content)
    retrieved_payloads = [
        str((item.get("payload") or {}).get("skills") or "")
        for item in traces
        if item.get("event") == "skill_retrieved"
    ]
    skill_retrieved = any(payload.strip() for payload in retrieved_payloads)
    skill_injected = "可复用技能" in content
    skill_guidance_coverage = _coverage(SKILL_GUIDANCE_TERMS, content)
    return {
        "id": task_id,
        "round": round_name,
        "run_id": run_id,
        "task": task,
        "status": data.get("status"),
        "passed": data.get("status") == "succeeded",
        "elapsed_seconds": round(elapsed, 6),
        "event_count": len(data.get("events") or []),
        "message_count": len(messages),
        "skill_trace_count": len(traces),
        "skill_retrieved": skill_retrieved,
        "skill_injected": skill_injected,
        "expected_terms": expected_terms,
        "term_coverage": coverage,
        "skill_guidance_coverage": skill_guidance_coverage,
        "output_preview": content[:500],
    }


def _coverage(expected_terms: List[str], content: str) -> float:
    if not expected_terms:
        return 1.0 if content.strip() else 0.0
    hits = sum(1 for term in expected_terms if term and term in content)
    return round(hits / len(expected_terms), 6)


def _summarize(
    rows: List[Dict[str, Any]],
    *,
    skill: Dict[str, Any],
    search_check: Dict[str, Any],
) -> Dict[str, Any]:
    rounds = {}
    for name in ["round1_no_skill", "round2_with_skill"]:
        items = [row for row in rows if row["round"] == name]
        rounds[name] = {
            "total": len(items),
            "success": sum(1 for row in items if row["passed"]),
            "success_rate": _rate(items, "passed"),
            "skill_retrieval_rate": _rate(items, "skill_retrieved"),
            "skill_injection_rate": _rate(items, "skill_injected"),
            "avg_term_coverage": _avg(items, "term_coverage"),
            "avg_skill_guidance_coverage": _avg(items, "skill_guidance_coverage"),
            "avg_elapsed_seconds": _avg(items, "elapsed_seconds"),
            "avg_skill_trace_count": _avg(items, "skill_trace_count"),
        }
    return {
        "name": "skill-loop-effect-calendar",
        "task_family": "calendar",
        "skill_id": skill["published"]["id"],
        "skill_status": skill["published"]["status"],
        "validation_passed": skill["validation"]["passed"],
        "search_match_count": len(search_check.get("matches") or []),
        "rounds": rounds,
        "effect": {
            "skill_retrieval_rate_delta": round(
                rounds["round2_with_skill"]["skill_retrieval_rate"]
                - rounds["round1_no_skill"]["skill_retrieval_rate"],
                6,
            ),
            "skill_injection_rate_delta": round(
                rounds["round2_with_skill"]["skill_injection_rate"]
                - rounds["round1_no_skill"]["skill_injection_rate"],
                6,
            ),
            "term_coverage_delta": round(
                rounds["round2_with_skill"]["avg_term_coverage"]
                - rounds["round1_no_skill"]["avg_term_coverage"],
                6,
            ),
            "skill_guidance_coverage_delta": round(
                rounds["round2_with_skill"]["avg_skill_guidance_coverage"]
                - rounds["round1_no_skill"]["avg_skill_guidance_coverage"],
                6,
            ),
        },
    }


def _rate(items: List[Dict[str, Any]], key: str) -> float:
    return round(sum(1 for item in items if item.get(key)) / len(items), 6) if items else 0.0


def _avg(items: List[Dict[str, Any]], key: str) -> float:
    values = [float(item.get(key) or 0.0) for item in items]
    return round(mean(values), 6) if values else 0.0


def _write_outputs(root: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (root / "report.md").write_text(_render_markdown(summary, rows), encoding="utf-8")


def _render_markdown(summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    r1 = summary["rounds"]["round1_no_skill"]
    r2 = summary["rounds"]["round2_with_skill"]
    effect = summary["effect"]
    lines = [
        "# 技能闭环效果实验：日历任务族",
        "",
        "## 实验目的",
        "",
        "比较同一批日历会议安排任务在“无已发布技能”和“发布技能后”两种条件下的运行差异。",
        "",
        "## 汇总结果",
        "",
        "| Round | 样本数 | 成功率 | 技能命中率 | 技能注入率 | 任务词覆盖率 | 技能指导覆盖率 | 平均耗时秒 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Round 1 无技能 | {r1['total']} | {r1['success_rate']:.4f} | {r1['skill_retrieval_rate']:.4f} | {r1['skill_injection_rate']:.4f} | {r1['avg_term_coverage']:.4f} | {r1['avg_skill_guidance_coverage']:.4f} | {r1['avg_elapsed_seconds']:.4f} |",
        f"| Round 2 有技能 | {r2['total']} | {r2['success_rate']:.4f} | {r2['skill_retrieval_rate']:.4f} | {r2['skill_injection_rate']:.4f} | {r2['avg_term_coverage']:.4f} | {r2['avg_skill_guidance_coverage']:.4f} | {r2['avg_elapsed_seconds']:.4f} |",
        "",
        "## 效果变化",
        "",
        f"- 技能检索率提升：{effect['skill_retrieval_rate_delta']:.4f}",
        f"- 技能注入率提升：{effect['skill_injection_rate_delta']:.4f}",
        f"- 关键项覆盖率变化：{effect['term_coverage_delta']:.4f}",
        f"- 技能指导覆盖率提升：{effect['skill_guidance_coverage_delta']:.4f}",
        "",
        "## 技能信息",
        "",
        f"- 技能 ID：`{summary['skill_id']}`",
        f"- 技能状态：`{summary['skill_status']}`",
        f"- 验证通过：`{summary['validation_passed']}`",
        f"- 检索检查命中数：`{summary['search_match_count']}`",
        "",
        "## 明细",
        "",
        "| id | round | status | skill_hit | skill_injected | task_coverage | skill_guidance_coverage | output_preview |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["round"] == "seed_for_skill":
            continue
        lines.append(
            "| {id} | {round} | {status} | {retrieved} | {injected} | {coverage:.4f} | {skill_coverage:.4f} | {preview} |".format(
                id=_cell(row["id"]),
                round=_cell(row["round"]),
                status=_cell(str(row["status"])),
                retrieved="是" if row["skill_retrieved"] else "否",
                injected="是" if row["skill_injected"] else "否",
                coverage=float(row["term_coverage"]),
                skill_coverage=float(row["skill_guidance_coverage"]),
                preview=_cell(row["output_preview"][:120]),
            )
        )
    return "\n".join(lines) + "\n"


def _cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
