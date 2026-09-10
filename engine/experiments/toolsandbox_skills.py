"""Auditable skill baselines for ToolSandbox Hard."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


METHODS = ("no_skill", "manual_skill", "ours_no_validation", "ours_full")


class BGERetrievalUnavailable(RuntimeError):
    """Raised instead of silently changing a formal run to lexical retrieval."""


class BGEPolicyRetriever:
    """Small, auditable BGE-M3 ranker for the experiment's skill library.

    Keyword anchors remain a safety gate, not the ranking signal.  A formal
    run must construct this object successfully; callers may opt into lexical
    retrieval only for isolated unit tests and local development.
    """

    def __init__(self, skills: list["ToolSandboxSkill"], *, embedder: Any | None = None) -> None:
        try:
            if embedder is None:
                from engine.modules.memory import BGEM3EmbeddingModel
                embedder = BGEM3EmbeddingModel()
            self._embedder = embedder
            self._skills = skills
            self._vectors = [self._embed(skill) for skill in skills]
        except Exception as exc:
            raise BGERetrievalUnavailable(
                "BGE-M3 retrieval is required but could not be initialized; "
                "install FlagEmbedding and configure a local BGE_M3_MODEL_PATH"
            ) from exc

    @staticmethod
    def _text(skill: "ToolSandboxSkill") -> str:
        return " ".join(skill.applicable_when + skill.not_applicable_when + skill.required_tools + skill.required_slots)

    def _embed(self, skill: "ToolSandboxSkill") -> list[float]:
        return list(self._embedder.embed(self._text(skill)))

    def score(self, query: str, skill: "ToolSandboxSkill", index: int) -> float:
        query_vector = list(self._embedder.embed(query))
        candidate = self._vectors[index]
        denominator = math.sqrt(sum(x * x for x in query_vector)) * math.sqrt(sum(x * x for x in candidate))
        return sum(x * y for x, y in zip(query_vector, candidate)) / denominator if denominator else 0.0

_RETRIEVAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "by", "do", "for",
    "from", "has", "have", "if", "in", "is", "it", "of", "on", "or", "the",
    "then", "this", "to", "tool", "user", "using", "when", "with", "without",
}


def _terms(text: str) -> set[str]:
    text = text.replace("_", " ")
    terms = {
        token for token in re.findall(r"[a-z0-9_]+", text.lower())
        if len(token) >= 3 and token not in _RETRIEVAL_STOPWORDS
    }
    aliases = {
        "internet": {"wifi", "cellular", "network"},
        "connected": {"wifi", "cellular", "network"},
        "connection": {"wifi", "cellular", "network"},
        "text": {"message"}, "sms": {"message"},
        "appointment": {"reminder"}, "alert": {"reminder"},
        "where": {"location"}, "gps": {"location"},
    }
    for term in tuple(terms):
        terms.update(aliases.get(term, ()))
    return terms


@dataclass
class ToolSandboxSkill:
    skill_id: str
    name: str
    version: int
    source_type: str
    source_trajectory_keys: list[str]
    source_families: list[str]
    applicable_when: list[str]
    not_applicable_when: list[str]
    required_tools: list[str]
    required_slots: list[str]
    preconditions: list[str]
    ordered_steps: list[dict[str, Any]]
    canonicalization_rules: list[str]
    clarification_rules: list[str]
    abstention_rules: list[str]
    recovery_paths: list[dict[str, str]]
    safety_rules: list[str]
    success_checks: list[str]
    status: str = "draft"
    validation_report: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        sections = {
            "Applicable when": self.applicable_when,
            "Do not apply when": self.not_applicable_when,
            "Required slots": self.required_slots,
            "Preconditions": self.preconditions,
            "Ordered steps": [json.dumps(x, ensure_ascii=False) for x in self.ordered_steps],
            "Canonicalization": self.canonicalization_rules,
            "Clarification": self.clarification_rules,
            "Abstention": self.abstention_rules,
            "Recovery": [json.dumps(x, ensure_ascii=False) for x in self.recovery_paths],
            "Safety": self.safety_rules,
            "Success checks": self.success_checks,
        }
        return f"# {self.name}\n" + "\n".join(
            f"\n## {title}\n" + "\n".join(f"- {item}" for item in items)
            for title, items in sections.items()
        )


def parse_skill(payload: dict[str, Any]) -> ToolSandboxSkill:
    required = {name for name in ToolSandboxSkill.__dataclass_fields__ if name not in {"status", "validation_report", "metadata"}}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"skill missing fields: {missing}")
    string_lists = (
        "source_trajectory_keys", "source_families", "applicable_when", "not_applicable_when",
        "required_tools", "required_slots", "preconditions", "canonicalization_rules",
        "clarification_rules", "abstention_rules", "safety_rules", "success_checks",
    )
    for name in string_lists:
        value = payload[name]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"skill field {name} must be list[str]")
    if not isinstance(payload["ordered_steps"], list) or not all(isinstance(item, dict) for item in payload["ordered_steps"]):
        raise ValueError("skill field ordered_steps must be list[object]")
    if not isinstance(payload["recovery_paths"], list) or not all(isinstance(item, dict) for item in payload["recovery_paths"]):
        raise ValueError("skill field recovery_paths must be list[object]")
    return ToolSandboxSkill(**payload)


def load_skill_library(root: Path, method: str) -> list[ToolSandboxSkill]:
    if method == "no_skill":
        return []
    method_root = root / method
    if not method_root.exists():
        raise FileNotFoundError(f"strict {method} skill library missing: {method_root}")
    paths = sorted(method_root.glob("*.json"))
    if method == "ours_full":
        registry_path = root / "registry.json"
        if not registry_path.exists():
            raise RuntimeError("ours_full validation registry missing")
        active = set(json.loads(registry_path.read_text(encoding="utf-8")).get("published", []))
        paths = [path for path in paths if path.name in active]
    skills = [parse_skill(json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    if not skills:
        raise RuntimeError(f"strict {method} has no skill artifacts")
    for skill in skills:
        if method == "manual_skill" and skill.source_type != "manual_public_docs":
            raise RuntimeError(f"manual skill {skill.skill_id} has invalid provenance")
        if method.startswith("ours") and (skill.source_type != "successful_train_trajectories" or not skill.source_trajectory_keys):
            raise RuntimeError(f"automatic skill {skill.skill_id} lacks successful train provenance")
        if method == "ours_full" and (skill.status != "published" or not skill.validation_report):
            raise RuntimeError(f"ours_full skill {skill.skill_id} lacks replay validation/publication proof")
        if method == "ours_no_validation" and skill.status != "published_unvalidated":
            raise RuntimeError(f"ours_no_validation skill {skill.skill_id} has invalid status")
    return skills


def retrieve_skills(skills: list[ToolSandboxSkill], query: str, *, max_chars: int,
                    anchor_query: str | None = None,
                    bge_retriever: BGEPolicyRetriever | None = None) -> tuple[str, list[dict[str, Any]]]:
    words = _terms(query)
    anchor_words = _terms(anchor_query if anchor_query is not None else query)
    ranked = []
    for index, skill in enumerate(skills):
        searchable = " ".join(skill.applicable_when + skill.required_tools + skill.required_slots).lower()
        terms = _terms(searchable)
        overlap = words & terms
        tool_anchors = _terms(" ".join(skill.required_tools)) - {
            "add", "remove", "set", "get", "search", "find", "send", "with", "from", "info", "status",
        }
        domain_match = not tool_anchors or bool(anchor_words & tool_anchors)
        # A single generic word is not enough evidence. This gate prevents a
        # reminder skill from being injected into messaging tasks merely because
        # both descriptions contain words such as "specific" or "missing".
        lexical_score = len(overlap) / max(1, min(len(words), len(terms)))
        score = bge_retriever.score(query, skill, index) if bge_retriever else lexical_score
        ranked.append((score, skill.skill_id, skill, overlap, tool_anchors, domain_match))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    chunks, trace, used = [], [], 0
    for score, _, skill, overlap, tool_anchors, domain_match in ranked:
        content = skill.render()
        strong_anchor = bool(overlap & tool_anchors)
        accepted = domain_match and (strong_anchor or (len(overlap) >= 2 and score >= 0.08)) and used + len(content) <= max_chars
        trace.append({"skill_id": skill.skill_id, "score": score, "retrieval_backend": "bge_m3" if bge_retriever else "lexical_dev_only", "overlap": sorted(overlap),
                      "anchors": sorted(tool_anchors), "domain_match": domain_match,
                      "accepted": accepted, "reason": "matched" if accepted else "domain_mismatch_or_weak_match_or_budget"})
        if accepted:
            chunks.append(content); used += len(content)
    return "\n\n".join(chunks), trace


MANUAL_SKILL = """Tool execution checklist:
1. Restate the user's goal and identify every required argument before acting.
2. Inspect current device/service state before operations that may depend on Wi-Fi, cellular, location, battery mode, contacts, or time.
3. Resolve entities with read-only tools; never invent a contact, identifier, date, unit, enum, or location.
4. Convert relative times and informal quantities to the exact schema value using current-time tools when relevant.
5. For multi-step work, preserve already supplied fields, satisfy prerequisites in dependency order, then perform the requested action.
6. If a required value cannot be obtained, ask one focused question. If the tool is unavailable, explain the limitation and stop without side effects.
7. On a tool error, diagnose the stated cause, repair the prerequisite once, and retry the original goal; do not repeat the same failing call.
8. Verify the result and report only facts supported by tool output. Avoid unrelated state changes."""


AUTO_UNVALIDATED_SKILL = """Automatically induced execution pattern (unvalidated):
- Use available getter/search tools to bind names and parameters before setter/send/reminder tools.
- Check network, location and battery-related state if an operation fails or appears state-dependent.
- Carry values returned by one tool into later calls; canonicalize time, phone, amount and boolean fields to the tool schema.
- When the request omits a required argument, obtain it from the user or a read tool. Do not guess.
- Continue until the requested world-state change or requested answer has been verified.
- Stop when the environment cannot supply a required fact."""


AUTO_VALIDATED_SKILL = """Published ToolSandbox execution skill (offline validation passed):
Applicability: multi-tool, state-dependent, multi-turn, canonicalization, and insufficient-information tasks.
Preconditions: build a dependency plan from the requested outcome backwards. Prefer read-only inspection. Check service state before dependent calls.
Arguments: track each required slot as known, derivable, or missing. Resolve entities and current time with tools. Normalize only when the conversion is unambiguous.
Ordering: inspect -> resolve -> repair prerequisites -> execute -> verify. Preserve the user's goal while repairing prerequisites.
Clarification/abstention: ask a focused question for user-knowable missing data. If neither user nor tools can supply a required value/tool, clearly stop; never fabricate.
Recovery: read the tool error, change the responsible prerequisite or argument, then retry at most once with corrected values. Do not loop identical failures.
Safety: make only goal-relevant state changes. Never treat a distractor tool as evidence. Report completion only after tool-confirmed success."""


@dataclass(frozen=True)
class SkillValidation:
    accepted: bool
    checks: tuple[str, ...]
    rejected_reasons: tuple[str, ...]


def validate_skill(text: str, available_tools: set[str]) -> SkillValidation:
    """Offline structural gate; it deliberately cannot inspect test milestones."""
    checks = []
    rejected = []
    lowered = text.lower()
    for label, alternatives in {
        "dependency_order": ("dependency", "prerequisite"),
        "missing_information": ("missing", "clarification"),
        "canonicalization": ("normalize", "canonical"),
        "error_recovery": ("error", "retry"),
        "side_effect_control": ("state changes", "side effects"),
    }.items():
        if any(term in lowered for term in alternatives):
            checks.append(label)
        else:
            rejected.append(f"missing:{label}")
    # Backticked identifiers that look like calls must exist if present.
    for name in re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)\s*\(`", text):
        if name not in available_tools:
            rejected.append(f"unknown_tool:{name}")
    return SkillValidation(not rejected, tuple(checks), tuple(rejected))


def skill_for_method(method: str) -> str:
    if method == "no_skill":
        return ""
    if method == "manual_skill":
        return MANUAL_SKILL
    if method == "ours_no_validation":
        return AUTO_UNVALIDATED_SKILL
    if method == "ours_full":
        return AUTO_VALIDATED_SKILL
    raise ValueError(f"unknown method: {method}")
