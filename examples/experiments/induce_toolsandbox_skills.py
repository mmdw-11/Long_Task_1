"""Induce structured skills exclusively from successful ToolSandbox train rollouts."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from engine.experiments.toolsandbox_skills import ToolSandboxSkill, parse_skill


def family_key(task_id: str) -> str:
    if "reminder" in task_id:
        return "reminders-time"
    if "message" in task_id or "contact" in task_id:
        return "messaging-contacts"
    if "wifi" in task_id or "cellular" in task_id or "location" in task_id or "battery" in task_id:
        return "device-state"
    return "general"


def safe_conversation(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        {"role": str(item.get("role") or ""), "content": str(item.get("content") or "")}
        for item in raw
        if item.get("role") != "system" and item.get("content")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--min-training-milestone", type=float, default=0.8)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    from openai import OpenAI
    model = args.model
    if model.startswith("deepseek-"):
        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    else:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"])
    tasks = {row["id"]: row for row in (json.loads(x) for x in args.dataset.read_text(encoding="utf-8").splitlines())}
    rows = [json.loads(x) for x in (args.collection_root / "rows.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    successful = [
        row for row in rows
        if not row.get("error")
        and float(row.get("minefield_similarity") or 0) == 0
        and float(row.get("milestone_similarity") or 0) >= args.min_training_milestone
        and tasks[row["task_id"]]["split"] == "skill_train"
    ]
    grouped = defaultdict(list)
    for row in successful:
        grouped[family_key(row["task_id"])].append(row)
    if not grouped:
        raise SystemExit("no successful skill_train trajectories")
    formal_ids = {key for key, value in tasks.items() if value["split"] == "formal_test"}
    schema_fields = list(ToolSandboxSkill.__dataclass_fields__)
    for group, group_rows in sorted(grouped.items()):
        demos, available = [], set()
        for row in group_rows:
            task = tasks[row["task_id"]]; available.update(task["tools"])
            trace_dir = args.collection_root / "trajectories" / row["key"].replace("/", "__")
            demos.append({"trajectory_key": row["key"], "family": row["family"], "tools": task["tools"],
                          "visible_successful_conversation": safe_conversation(trace_dir / "conversation.json")})
        shape = {
            "skill_id": "overridden-by-runner", "name": "string", "version": 1,
            "source_type": "overridden-by-runner", "source_trajectory_keys": [], "source_families": [],
            "applicable_when": ["string"], "not_applicable_when": ["string"],
            "required_tools": ["tool_name"], "required_slots": ["string"], "preconditions": ["string"],
            "ordered_steps": [{"order": 1, "instruction": "string", "tool": "optional existing tool or null"}],
            "canonicalization_rules": ["string"], "clarification_rules": ["string"],
            "abstention_rules": ["string"], "recovery_paths": [{"on": "string", "then": "string"}],
            "safety_rules": ["string"], "success_checks": ["string"], "status": "draft",
            "validation_report": "", "metadata": {},
        }
        prompt = (
            "Induce ONE reusable tool-use skill from only these successful TRAIN conversations. "
            "Return one JSON object, no markdown. Never include scenario IDs or hidden evaluator targets in instructions. "
            "Use only AVAILABLE_TOOLS. Each ordered_steps item has keys order, instruction, and optional tool. "
            "Record the provided trajectory keys and families exactly. status must be draft.\n"
            "Every plural field shown as an array MUST remain an array, even with one item. "
            "recovery_paths must use exactly the keys on/then.\n"
            f"EXACT_SHAPE={json.dumps(shape)}\nAVAILABLE_TOOLS={json.dumps(sorted(available))}\n"
            f"GROUP={group}\nDEMOS={json.dumps(demos, ensure_ascii=False)}"
        )
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0,
            response_format={"type": "json_object"}, max_tokens=4000,
            extra_body={"thinking": {"type": "disabled"}},
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        payload.update({"skill_id": f"auto-{group}-v1", "name": f"Induced {group} workflow", "version": 1,
                        "source_type": "successful_train_trajectories",
                        "source_trajectory_keys": [r["key"] for r in group_rows],
                        "source_families": [r["family"] for r in group_rows], "status": "draft",
                        "validation_report": "", "metadata": {
                            "generator_model": model, "training_successes": len(group_rows),
                            "training_qualification": {
                                "error_free": True, "minefield_similarity": 0,
                                "minimum_milestone_similarity": args.min_training_milestone,
                            },
                        }})
        skill = parse_skill(payload)
        if set(skill.source_trajectory_keys) & formal_ids:
            raise RuntimeError("formal leakage in skill provenance")
        unknown = set(skill.required_tools) - available
        unknown |= {str(step.get("tool")) for step in skill.ordered_steps if step.get("tool")} - available
        if unknown:
            raise RuntimeError(f"generated unknown tools: {sorted(unknown)}")
        for method, status in (("ours_no_validation", "published_unvalidated"), ("ours_full_candidates", "candidate_pending_validation")):
            clone = copy.deepcopy(skill); clone.status = status
            target = args.output_root / method; target.mkdir(parents=True, exist_ok=True)
            (target / f"{clone.skill_id}.json").write_text(json.dumps(clone.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print({"skill": skill.skill_id, "successful_sources": len(group_rows), "tools": len(available)})


if __name__ == "__main__":
    main()
