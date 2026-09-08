"""Structured skill induction and gated publication for the τ experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .types import TauToolTask


MUTATION_PREFIXES = (
    "add_", "book_", "cancel_", "change_", "create_", "delete_", "exchange_",
    "modify_", "remove_", "refund_", "reserve_", "return_", "send_", "update_",
)


@dataclass
class TauSkill:
    id: str
    name: str
    domain: str
    applicable_when: list[str]
    not_applicable_when: list[str]
    identity_verification: list[str]
    read_steps: list[dict[str, Any]]
    write_steps: list[dict[str, Any]]
    branches: list[dict[str, Any]]
    refusal_conditions: list[str]
    handoff_conditions: list[str]
    recovery_paths: list[dict[str, Any]]
    final_state_checks: list[str]
    communicate_info: list[str]
    source_task_ids: list[str]
    status: str = "draft"
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        def lines(values: Iterable[Any]) -> str:
            rendered = []
            for value in values:
                rendered.append(f"- {json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value}")
            return "\n".join(rendered) or "- None"
        return (
            f"# {self.name}\n\n## Applicable when\n{lines(self.applicable_when)}\n\n"
            f"## Do not apply when\n{lines(self.not_applicable_when)}\n\n"
            f"## Identity verification\n{lines(self.identity_verification)}\n\n"
            f"## Read-only steps\n{lines(self.read_steps)}\n\n"
            f"## Mutating steps (explicit confirmation required)\n{lines(self.write_steps)}\n\n"
            f"## Branches and refusals\n{lines(self.branches + [{'refuse': x} for x in self.refusal_conditions])}\n\n"
            f"## Recovery and handoff\n{lines(self.recovery_paths + [{'handoff': x} for x in self.handoff_conditions])}\n\n"
            f"## Final state and communication\n{lines(self.final_state_checks + self.communicate_info)}\n"
        )


class TauSkillValidationError(ValueError):
    pass


def is_write_tool(name: str) -> bool:
    return name.startswith(MUTATION_PREFIXES)


def validate_tau_skill(
    skill: TauSkill,
    *,
    available_tools: set[str],
    test_ids: set[str] | None = None,
    replay: Callable[[TauSkill], dict[str, Any]] | None = None,
    baseline_reward: float = 0.0,
) -> dict[str, Any]:
    errors: list[str] = []
    if not skill.applicable_when or not skill.not_applicable_when:
        errors.append("applicability and exclusion rules are required")
    mentioned = [str(step.get("tool") or "") for step in skill.read_steps + skill.write_steps]
    unknown = sorted({name for name in mentioned if name and name not in available_tools})
    if unknown:
        errors.append(f"unknown tools: {unknown}")
    for step in skill.write_steps:
        tool = str(step.get("tool") or "")
        if tool and not is_write_tool(tool):
            errors.append(f"write step uses non-mutating tool: {tool}")
        if not step.get("requires_confirmation"):
            errors.append(f"write step lacks explicit confirmation: {tool or '<missing>'}")
        if not step.get("depends_on"):
            errors.append(f"write step lacks parameter/state dependency: {tool or '<missing>'}")
    leaked = sorted(set(skill.source_task_ids) & set(test_ids or set()))
    if leaked:
        errors.append(f"test task provenance leak: {leaked}")
    replay_result: dict[str, Any] | None = None
    if not errors and replay is not None:
        replay_result = replay(skill)
        if float(replay_result.get("reward", 0.0)) < baseline_reward:
            errors.append("validation reward is below no-skill baseline")
        for field in ("policy_violations", "tool_errors", "unauthorized_mutations"):
            if int(replay_result.get(field, 0)) > int(replay_result.get(f"baseline_{field}", 0)):
                errors.append(f"validation increased {field}")
    if errors:
        skill.status = "rejected"
        raise TauSkillValidationError("; ".join(errors))
    skill.status = "validated"
    return {"accepted": True, "errors": [], "replay": replay_result}


def render_generation_prompt(domain: str, tasks: list[TauToolTask], tools: list[dict[str, Any]]) -> str:
    safe_tasks = []
    for task in tasks:
        if task.split != "skill_train":
            raise ValueError("skill induction accepts skill_train tasks only")
        safe_tasks.append({
            "id": task.id,
            "scenario": task.user_scenario,
            "successful_reference_trajectory": task.reference_actions,
            "must_communicate": task.communicate_info,
        })
    schema = {
        "id": "string", "name": "string", "domain": domain,
        "applicable_when": ["string"], "not_applicable_when": ["string"],
        "identity_verification": ["string"],
        "read_steps": [{"tool": "existing tool", "depends_on": ["field"]}],
        "write_steps": [{"tool": "existing tool", "depends_on": ["field"], "requires_confirmation": True}],
        "branches": [{"condition": "string", "then": "string"}],
        "refusal_conditions": ["string"], "handoff_conditions": ["string"],
        "recovery_paths": [{"on": "tool error", "then": "string"}],
        "final_state_checks": ["string"], "communicate_info": ["string"],
        "source_task_ids": ["train id"]
    }
    return (
        "Induce one reusable customer-service tool skill from successful TRAIN trajectories. "
        "Return exactly one JSON object matching the schema. Never invent tools or fields. "
        "Separate reads from writes, require explicit confirmation immediately before every write, "
        "include refusal, handoff and recovery conditions, and do not copy scenario-specific secrets.\n"
        f"DOMAIN={domain}\nTOOLS={json.dumps(tools, ensure_ascii=False)}\n"
        f"SCHEMA={json.dumps(schema, ensure_ascii=False)}\n"
        f"TRAIN={json.dumps(safe_tasks, ensure_ascii=False)}"
    )


def group_training_tasks(tasks: Iterable[TauToolTask]) -> dict[str, list[TauToolTask]]:
    """Group successful train trajectories into reusable action families."""
    groups: dict[str, list[TauToolTask]] = {}
    for task in tasks:
        if task.split != "skill_train":
            raise ValueError("only skill_train tasks may be grouped for induction")
        writes = [
            str(action.get("name") or "") for action in task.reference_actions
            if is_write_tool(str(action.get("name") or ""))
        ]
        groups.setdefault(writes[-1] if writes else "read_or_refuse", []).append(task)
    return dict(sorted(groups.items()))


def parse_generated_skill(text: str, *, domain: str) -> TauSkill:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[1].rsplit("```", 1)[0]
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise TauSkillValidationError("generated skill must be one JSON object")
    payload["domain"] = domain
    required = {
        "id", "name", "applicable_when", "not_applicable_when", "identity_verification",
        "read_steps", "write_steps", "branches", "refusal_conditions", "handoff_conditions",
        "recovery_paths", "final_state_checks", "communicate_info", "source_task_ids",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise TauSkillValidationError(f"generated skill missing fields: {missing}")
    return TauSkill(**payload)


def save_tau_skill(skill: TauSkill, root: str | Path) -> dict[str, str]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"{skill.id}.json"
    markdown_path = root / f"{skill.id}.md"
    json_path.write_text(json.dumps(skill.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(skill.to_markdown(), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def manual_skill(domain: str, *, available_tools: Iterable[str]) -> TauSkill:
    """Complete expert baseline derived only from public domain policy/tool schema."""
    tools = set(available_tools)
    reads = sorted(name for name in tools if not is_write_tool(name))
    writes = sorted(name for name in tools if is_write_tool(name))
    return TauSkill(
        id=f"manual-{domain}-v1", name=f"Expert {domain} service workflow", domain=domain,
        applicable_when=[f"The user requests supported {domain} customer-service assistance."],
        not_applicable_when=["The request is outside this domain or policy forbids the requested operation."],
        identity_verification=["Verify the customer using only policy-approved identifiers before account-specific access."],
        read_steps=[{"tool": name, "depends_on": ["policy-required identifiers"]} for name in reads],
        write_steps=[{"tool": name, "depends_on": ["verified state", "resolved arguments"], "requires_confirmation": True} for name in writes],
        branches=[{"condition": "request is unsupported", "then": "refuse and explain the applicable policy"}],
        refusal_conditions=["Policy preconditions are not met", "The user declines the disclosed consequences"],
        handoff_conditions=["A policy or tool explicitly requires transfer to a human"],
        recovery_paths=[{"on": "tool error", "then": "re-read state, correct arguments, and retry only when safe"}],
        final_state_checks=["Re-read affected records after mutation", "Report the outcome without inventing facts"],
        communicate_info=["Disclose fees, refunds, restrictions, and next steps required by policy"],
        source_task_ids=[], status="published", metadata={"source": "official_policy_and_tool_schema"},
    )
