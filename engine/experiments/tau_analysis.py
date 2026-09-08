"""Offline task stratification and retrieval diagnostics for the τ experiment."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from .tau_skills import TauSkill, is_write_tool
from .types import TauToolTask


def scenario_text(task: TauToolTask, *, include_hidden_instructions: bool = False) -> str:
    """Return a retrieval probe; hidden instructions are analysis-only."""
    instructions = (task.user_scenario or {}).get("instructions") or {}
    parts = [str(instructions.get("reason_for_call") or "")]
    if include_hidden_instructions:
        parts.extend([
            str(instructions.get("task_instructions") or ""),
            str(instructions.get("unknown_info") or ""),
        ])
    return "\n".join(part for part in parts if part).strip()


def eventual_skill_families(task: TauToolTask) -> set[str]:
    """Derive analysis-only relevant families from goals plus official actions.

    A refused request still needs the skill for the requested action (for
    example, a cancellation skill must explain when cancellation is forbidden),
    so a no-write reference trajectory is not automatically ``read_or_refuse``.
    """
    writes = {
        str(action.get("name") or "")
        for action in task.reference_actions
        if is_write_tool(str(action.get("name") or ""))
    }
    # Passenger changes are intentionally covered by the generated baggage /
    # reservation-modification skill in this frozen skill library.
    if "update_reservation_passengers" in writes:
        writes.remove("update_reservation_passengers")
        writes.add("update_reservation_baggages")
    if writes:
        return writes
    query = scenario_text(task).lower()
    if task.domain == "airline":
        if re.search(r"\bcancel\w*\b", query):
            writes.add("cancel_reservation")
        if re.search(r"\b(?:book|reserve)\w*\b|\bmake\s+a\s+reservation\b|\b(?:want|need|would like)\s+to\s+fly\b", query) and not re.search(r"\bbooked\b", query):
            writes.add("book_reservation")
        if re.search(r"\b(?:change|modify|upgrade|downgrade|switch)\w*\b", query) and re.search(
            r"\b(?:flight|cabin|economy|business|passenger)\w*\b", query
        ):
            writes.add("update_reservation_flights")
        if re.search(r"\b(?:bag|bags|baggage|luggage)\b", query):
            writes.add("update_reservation_baggages")
    else:
        if re.search(r"\bcancel\w*\b", query):
            writes.add("cancel_pending_order")
        if re.search(r"\b(?:exchange|swap|replace)\w*\b", query):
            writes.add("exchange_delivered_order_items")
        if re.search(r"\b(?:return|refund)\w*\b|\bmoney\s+back\b", query):
            writes.add("return_delivered_order_items")
        if re.search(r"\baddress\b|\bsuite\b", query):
            writes.add("modify_pending_order_address")
        if re.search(r"\b(?:modify|switch|add|remove|change)\w*\b", query) and re.search(r"\b(?:item|order|speaker|bottle|laptop|camera|watch)\w*\b", query):
            writes.add("modify_pending_order_items")
    return writes or {"read_or_refuse"}


def initial_intent_families(task: TauToolTask) -> set[str]:
    """Label skills relevant to the initially stated goal, without future turns."""
    query = scenario_text(task).lower()
    result: set[str] = set()
    if task.domain == "airline":
        if re.search(r"\bcancel\w*\b", query):
            result.add("cancel_reservation")
        if re.search(r"\b(?:book|reserve)\w*\b|\bmake\s+a\s+reservation\b|\b(?:want|need|would like)\s+to\s+fly\b", query) and not re.search(r"\bbooked\b", query):
            result.add("book_reservation")
        if re.search(r"\b(?:change|modif|upgrade|downgrade|switch)\w*\b", query) and re.search(
            r"\b(?:flight|cabin|economy|business|passenger|trip)\w*\b", query
        ):
            result.add("update_reservation_flights")
        if re.search(r"\b(?:bag|bags|baggage|luggage)\b", query):
            result.add("update_reservation_baggages")
    else:
        if re.search(r"\bcancel\w*\b", query) and not re.search(r"\bnever\s+want\s+to\s+cancel\b", query):
            result.add("cancel_pending_order")
        if re.search(r"\b(?:exchange|swap|replace)\w*\b", query):
            result.add("exchange_delivered_order_items")
        if re.search(r"\b(?:want|need|would like)\b[^.!?]{0,35}\b(?:return|refund)\w*\b|\bget\s+a\s+refund\b|\bmoney\s+back\b", query) and not re.search(r"\breturn\w*\s+.*\blater\b", query):
            result.add("return_delivered_order_items")
        if re.search(r"\baddress\b|\bsuite\b", query):
            result.add("modify_pending_order_address")
        if re.search(r"\b(?:modif|upgrade|switch|add|remove|change|split)\w*\b", query) and re.search(
            r"\b(?:item|order|payment|speaker|bottle|laptop|camera|watch|t-?shirt)\w*\b", query
        ):
            result.add("modify_pending_order_items")
    return result or {"read_or_refuse"}


def task_profile(task: TauToolTask) -> dict[str, Any]:
    actions = [str(item.get("name") or "") for item in task.reference_actions if item.get("name")]
    write_actions = [name for name in actions if is_write_tool(name)]
    read_actions = [name for name in actions if not is_write_tool(name)]
    families = eventual_skill_families(task)
    instructions = (task.user_scenario or {}).get("instructions") or {}
    task_instructions = str(instructions.get("task_instructions") or "")
    constraints = len([line for line in task_instructions.splitlines() if line.strip()])
    communication_count = len(task.communicate_info) + len(task.metadata.get("nl_assertions") or [])
    transfer = "transfer_to_human_agents" in actions
    refusal = not write_actions or transfer
    multi_goal = len(families - {"read_or_refuse"}) > 1
    depth = len(actions)
    score = 0
    score += 0 if depth <= 2 else 1 if depth <= 5 else 2 if depth <= 9 else 3
    score += 1 if len(set(actions)) >= 3 else 0
    score += 1 if len(write_actions) >= 2 else 0
    score += 1 if len(write_actions) >= 4 else 0
    score += 2 if multi_goal else 0
    score += 1 if communication_count else 0
    score += 1 if refusal else 0
    score += 1 if constraints >= 3 else 0
    difficulty = "easy" if score <= 2 else "medium" if score <= 5 else "hard"
    return {
        "id": task.id,
        "domain": task.domain,
        "difficulty": difficulty,
        "difficulty_score": score,
        "action_depth": depth,
        "read_actions": len(read_actions),
        "write_actions": len(write_actions),
        "unique_tools": len(set(actions)),
        "expected_skill_families": sorted(families),
        "multi_goal": multi_goal,
        "requires_confirmation": bool(write_actions),
        "refusal_or_handoff": refusal,
        "communication_requirements": communication_count,
        "instruction_constraints": constraints,
        "scenario_chars": len(scenario_text(task, include_hidden_instructions=True)),
    }


def summarize_profiles(profiles: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(profiles)
    by_domain: dict[str, Any] = {}
    for domain in sorted({row["domain"] for row in rows}):
        selected = [row for row in rows if row["domain"] == domain]
        by_domain[domain] = {
            "tasks": len(selected),
            "difficulty": dict(sorted(Counter(row["difficulty"] for row in selected).items())),
            "multi_goal": sum(row["multi_goal"] for row in selected),
            "requires_confirmation": sum(row["requires_confirmation"] for row in selected),
            "refusal_or_handoff": sum(row["refusal_or_handoff"] for row in selected),
            "mean_action_depth": round(sum(row["action_depth"] for row in selected) / max(1, len(selected)), 3),
        }
    return {
        "tasks": len(rows),
        "domains": by_domain,
        "difficulty": dict(sorted(Counter(row["difficulty"] for row in rows).items())),
        "profile_schema": "tau_task_profile_v1",
    }


def retrieval_metrics(
    tasks: Iterable[TauToolTask],
    skills: list[TauSkill],
    ranker,
    *,
    include_hidden_instructions: bool = False,
    top_k: int = 2,
    query_suffix: str = "",
    oracle: str = "initial_intent",
) -> dict[str, Any]:
    rows = []
    for task in tasks:
        query = (scenario_text(task, include_hidden_instructions=include_hidden_instructions) + " " + query_suffix).strip()
        ranked = ranker(skills, query, task.domain)
        ranked = [(skill, score) for skill, score in ranked if score >= 0.5]
        families = [str(skill.metadata.get("action_family") or skill.id) for skill, _ in ranked[:top_k]]
        expected = (
            initial_intent_families(task)
            if oracle == "initial_intent"
            else eventual_skill_families(task)
        )
        relevant_ranks = [index + 1 for index, family in enumerate(
            str(skill.metadata.get("action_family") or skill.id) for skill, _ in ranked
        ) if family in expected]
        covered = expected & set(families)
        rows.append({
            "id": task.id,
            "domain": task.domain,
            "query_mode": "full_analysis_only" if include_hidden_instructions else "first_request_proxy",
            "oracle": oracle,
            "expected": sorted(expected),
            "selected": families,
            "hit_at_1": int(bool(families and families[0] in expected)),
            "recall_at_k": len(covered) / len(expected),
            "mrr": 1 / min(relevant_ranks) if relevant_ranks else 0.0,
        })
    count = max(1, len(rows))
    return {
        "tasks": len(rows),
        "top_k": top_k,
        "hit_at_1": sum(row["hit_at_1"] for row in rows) / count,
        "recall_at_k": sum(row["recall_at_k"] for row in rows) / count,
        "mrr": sum(row["mrr"] for row in rows) / count,
        "rows": rows,
    }
