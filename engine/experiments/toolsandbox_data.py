"""Deterministic, leakage-aware selection for the ToolSandbox Hard experiment."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


EXTERNAL_TOOLS = {
    "search_lat_lon",
    "search_location_around_lat_lon",
    "search_weather_around_lat_lon",
    "search_stock",
    "convert_currency",
}


@dataclass(frozen=True)
class ToolSandboxTask:
    id: str
    family: str
    split: str
    primary_group: str
    categories: tuple[str, ...]
    tools: tuple[str, ...]
    milestone_count: int
    minefield_count: int
    max_messages: int
    difficulty_score: int
    source_commit: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["categories"] = list(self.categories)
        value["tools"] = list(self.tools)
        return value


_AUGMENT_SUFFIX = re.compile(
    r"_(?:3|10)_distraction_tools(?:_(?:tool|arg)_(?:name|description|type)_scrambled)*$"
)


def scenario_family(name: str) -> str:
    """Return the base scenario family used as the leakage unit."""
    return _AUGMENT_SUFFIX.sub("", name)


def _category_names(scenario: Any) -> tuple[str, ...]:
    return tuple(sorted(getattr(item, "name", str(item)) for item in scenario.categories))


def primary_group(categories: Iterable[str]) -> str | None:
    categories = set(categories)
    if {"STATE_DEPENDENCY", "MULTIPLE_TOOL_CALL"} <= categories:
        return "state_dependency"
    if {"MULTIPLE_USER_TURN", "MULTIPLE_TOOL_CALL"} <= categories:
        return "multi_tool_multi_turn"
    if "INSUFFICIENT_INFORMATION" in categories:
        return "insufficient_information"
    if "CANONICALIZATION" in categories:
        return "canonicalization"
    return None


def difficulty_score(scenario: Any, categories: Iterable[str]) -> int:
    cats = set(categories)
    milestones = len(scenario.evaluation.milestone_matcher.milestones)
    minefields = len(scenario.evaluation.minefield_matcher.milestones)
    tools = len(scenario.starting_context.tool_allow_list or ())
    return (
        milestones * 8
        + minefields * 7
        + tools * 2
        + min(int(scenario.max_messages), 50)
        + 15 * int("STATE_DEPENDENCY" in cats)
        + 12 * int("MULTIPLE_USER_TURN" in cats)
        + 10 * int("MULTIPLE_TOOL_CALL" in cats)
        + 8 * int("INSUFFICIENT_INFORMATION" in cats)
        + 6 * int("CANONICALIZATION" in cats)
    )


def select_tasks(source_root: Path) -> tuple[list[ToolSandboxTask], dict[str, Any]]:
    """Select formal/canary/pilot tasks without consulting model outcomes."""
    from tool_sandbox.common.tool_discovery import ToolBackend
    from tool_sandbox.scenarios import named_scenarios

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    scenarios = named_scenarios(ToolBackend.DEFAULT)
    candidates: dict[str, list[tuple[str, Any, tuple[str, ...], int]]] = {
        key: []
        for key in (
            "state_dependency",
            "multi_tool_multi_turn",
            "insufficient_information",
            "canonicalization",
        )
    }
    general_candidates: list[tuple[str, Any, tuple[str, ...], int, str]] = []
    external_reserve: dict[str, list[tuple[str, Any, tuple[str, ...], int]]] = {key: [] for key in candidates}
    excluded_external: list[str] = []
    for name, scenario in scenarios.items():
        if not name.endswith("_3_distraction_tools"):
            continue
        tools = set(scenario.starting_context.tool_allow_list or ())
        if tools & EXTERNAL_TOOLS:
            excluded_external.append(name)
            categories = _category_names(scenario)
            group = primary_group(categories)
            if group:
                external_reserve[group].append(
                    (name, scenario, categories, difficulty_score(scenario, categories))
                )
            continue
        categories = _category_names(scenario)
        group = primary_group(categories)
        if group:
            candidates[group].append(
                (name, scenario, categories, difficulty_score(scenario, categories))
            )
        general_candidates.append((name, scenario, categories, difficulty_score(scenario, categories), group or "general"))
    for rows in candidates.values():
        rows.sort(key=lambda row: (-row[3], row[0]))
    for rows in external_reserve.values():
        rows.sort(key=lambda row: (-row[3], row[0]))

    formal_need = {
        "state_dependency": 10,
        "multi_tool_multi_turn": 10,
        "insufficient_information": 5,
        "canonicalization": 5,
    }
    train_need = {"state_dependency": 2, "multi_tool_multi_turn": 2,
                  "insufficient_information": 4, "canonicalization": 4}
    validation_need = {"state_dependency": 1, "multi_tool_multi_turn": 1,
                       "insufficient_information": 2, "canonicalization": 2}
    selected: list[ToolSandboxTask] = []
    used_families: set[str] = set()

    def take(group: str, split: str, count: int) -> None:
        if count == 0:
            return
        taken = 0
        for name, scenario, categories, score in candidates[group]:
            family = scenario_family(name)
            if family in used_families:
                continue
            selected.append(
                ToolSandboxTask(
                    id=name,
                    family=family,
                    split=split,
                    primary_group=group,
                    categories=categories,
                    tools=tuple(sorted(scenario.starting_context.tool_allow_list or ())),
                    milestone_count=len(scenario.evaluation.milestone_matcher.milestones),
                    minefield_count=len(scenario.evaluation.minefield_matcher.milestones),
                    max_messages=int(scenario.max_messages),
                    difficulty_score=score,
                    source_commit=commit,
                )
            )
            used_families.add(family)
            taken += 1
            if taken == count:
                return
        raise RuntimeError(f"not enough family-disjoint candidates for {split}/{group}: {taken}/{count}")

    # Freeze the formal test first; API pilots can never influence its membership.
    for group, count in formal_need.items():
        take(group, "formal_test", count)
    # The two scarcest strata have exactly 13 safe families. After freezing ten
    # formal families, reserve two for training and one for validation. Canary
    # and pilot use other safe families and never consume scarce learning data.
    for group, count in train_need.items():
        take(group, "skill_train", count)
    for group, count in validation_need.items():
        take(group, "skill_validation", count)
    # Reserve unrelated scenario families for skill construction/validation.  Some
    # hard strata (notably state-dependency) have only enough unique families for
    # formal+canary+pilot, so training must come from other families to preserve
    # the strict family boundary.
    reserve_pool = general_candidates
    reserve_pool.sort(key=lambda row: (-row[3], row[0]))

    def take_reserve(split: str, count: int) -> None:
        taken = 0
        covered_tools: set[str] = set()
        while taken < count:
            eligible = [row for row in reserve_pool if scenario_family(row[0]) not in used_families]
            if not eligible:
                break
            name, scenario, categories, score, group = max(
                eligible,
                key=lambda row: (
                    len(set(row[1].starting_context.tool_allow_list or ()) - covered_tools),
                    row[3],
                    row[0],
                ),
            )
            family = scenario_family(name)
            selected.append(ToolSandboxTask(
                id=name, family=family, split=split, primary_group=group,
                categories=categories,
                tools=tuple(sorted(scenario.starting_context.tool_allow_list or ())),
                milestone_count=len(scenario.evaluation.milestone_matcher.milestones),
                minefield_count=len(scenario.evaluation.minefield_matcher.milestones),
                max_messages=int(scenario.max_messages), difficulty_score=score,
                source_commit=commit,
            ))
            used_families.add(family)
            covered_tools.update(scenario.starting_context.tool_allow_list or ())
            taken += 1
        if taken != count:
            raise RuntimeError(f"not enough reserve families for {split}: {taken}/{count}")

    take_reserve("canary", 3)
    take_reserve("pilot", 6)

    audit = {
        "source_commit": commit,
        "official_scenarios": len(scenarios),
        "candidate_counts_after_external_filter": {k: len(v) for k, v in candidates.items()},
        "excluded_external_count": len(excluded_external),
        "excluded_external_ids": sorted(excluded_external),
        "external_reserve_group_counts": {k: len(v) for k, v in external_reserve.items()},
        "split_counts": {
            split: sum(task.split == split for task in selected)
            for split in sorted({task.split for task in selected})
        },
        "family_disjoint": len({task.family for task in selected}) == len(selected),
        "formal_group_counts": {
            group: sum(task.split == "formal_test" and task.primary_group == group for task in selected)
            for group in formal_need
        },
        "selection_rule": "static difficulty descending, scenario id ascending; no rollout outcomes",
    }
    return selected, audit


def sha256_file(path: Path) -> str:
    """Hash JSONL content canonically, independent of checkout line endings.

    The frozen manifest is shared between Windows and Unix runners.  Hashing
    raw bytes made the same task rows appear to have different identities when
    Git converted LF/CRLF.  JSONL is deliberately serialized as one compact
    JSON object per LF-terminated line, so this normalization is lossless.
    """
    text = path.read_text(encoding="utf-8")
    normalized = "\n".join(text.splitlines())
    if normalized:
        normalized += "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def write_dataset(source_root: Path, output_root: Path) -> dict[str, Any]:
    tasks, audit = select_tasks(source_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks_path = output_root / "tasks.jsonl"
    tasks_path.write_text(
        "".join(json.dumps(task.to_dict(), ensure_ascii=False) + "\n" for task in tasks),
        encoding="utf-8",
    )
    manifest = {
        **audit,
        "dataset": "Apple ToolSandbox (project-derived family-disjoint split)",
        "source_url": "https://github.com/apple/ToolSandbox",
        "license_file": "LICENSE",
        "tasks_file": str(tasks_path),
        "tasks_sha256": sha256_file(tasks_path),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
