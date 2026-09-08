"""Live official τ simulation adapter and reproducible result aggregation.

Imports from ``tau2`` are deliberately local so the legacy experiments remain
usable without the dedicated τ environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .tau_skills import TauSkill, is_write_tool


TAU_METHODS = ("no_skill", "manual_skill", "ours_no_validation", "ours_full")


@dataclass
class TauRunConfig:
    dataset: str = "data/processed/tau_skill_v1/tasks.jsonl"
    output_root: str = "runs/experiments/tau_skill_v1"
    methods: tuple[str, ...] = TAU_METHODS
    domains: tuple[str, ...] = ("retail", "airline")
    trials: int = 2
    seed: int = 42
    agent_model: str = "openai/deepseek-v4-flash"
    user_model: str = "openai/deepseek-v4-flash"
    evaluator_model: str = "openai/deepseek-v4-flash"
    max_steps: int = 200
    max_errors: int = 10
    timeout: float = 900.0
    skill_context_budget_chars: int = 12_000
    fresh: bool = False
    limit: int | None = None


def simulation_key(domain: str, method: str, trial: int, task_id: str) -> str:
    return f"{domain}/{method}/trial-{trial}/{task_id}"


def paired_trial_seed(base_seed: int, trial: int, task_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{trial}:{task_id}".encode()).hexdigest()
    return base_seed + trial * 1_000_003 + int(digest[:8], 16) % 1_000_000


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _llm_args(seed: int) -> dict[str, Any]:
    import os
    args: dict[str, Any] = {"temperature": 0, "seed": seed}
    base = os.getenv("OPENAI_BASE_URL")
    key = os.getenv("OPENAI_API_KEY")
    if base:
        args["api_base"] = base
    if key:
        args["api_key"] = key
    # DeepSeek's reasoning tokens can otherwise consume the entire short reply.
    args["extra_body"] = {"thinking": {"type": "disabled"}}
    return args


def _load_official_tasks(domain: str) -> dict[str, Any]:
    from tau2.registry import registry
    tasks = registry.get_tasks_loader(domain)(None)
    return {str(task.id): task for task in tasks}


def _skill_prompt(skill: TauSkill | None, *, budget_chars: int) -> str:
    if skill is None:
        return ""
    prefix = (
        "\n\n<retrieved_skill>\n"
        "This is reusable guidance, not task ground truth. Apply it only when its conditions match. "
        "The official policy always has priority. Never treat examples as user facts.\n"
    )
    suffix = "\n</retrieved_skill>"
    if budget_chars < len(prefix) + len(suffix) + 1:
        return ""
    body_budget = max(0, budget_chars - len(prefix) - len(suffix))
    return prefix + skill.to_markdown()[:body_budget] + suffix


def _skill_score(skill: TauSkill, query: str, domain: str) -> float:
    if skill.domain != domain:
        return float("-inf")
    query_lower = query.lower().replace("_", " ")
    # Do not let explicit irrelevant/negative clauses become positive routing
    # signals (a common failure mode for bag-of-words retrieval).
    query_lower = re.sub(
        r"\b(?:i|we|you)\s+(?:do\s+not|don't|don’t)\s+want\b[^.!?]*[.!?]?",
        " ",
        query_lower,
    )
    terms = set(query_lower.split())
    document = " ".join([skill.name, *skill.applicable_when, *skill.not_applicable_when]).lower()
    # Generated applicability prose can be overly broad, so it is only a
    # deterministic tie-breaker. Routing is driven by the audited action-family
    # metadata and explicit user intent.
    lexical = 0.05 * sum(term in document for term in terms if len(term) > 3) / max(1, len(terms))
    family = str(skill.metadata.get("action_family") or skill.id).lower().replace("-", "_")

    def has(*patterns: str) -> bool:
        return any(re.search(pattern, query_lower) for pattern in patterns)

    def positive(term: str) -> bool:
        if not re.search(rf"\b{re.escape(term)}\w*\b", query_lower):
            return False
        return not re.search(
            rf"\b(?:never|without|don['’]?t|do\s+not)\b(?:\W+(?:want|need)\w*)?(?:\W+to)?\W+{re.escape(term)}\w*\b",
            query_lower,
        )

    score = lexical
    if "read_or_refuse" in family:
        if has(r"\b(?:human\s+agent|transfer|talk\s+to\s+(?:an?\s+)?agent)\b"):
            score += 2.0
        elif has(r"\b(?:status|track|tracking|price|paid|policy|insurance|information)\b"):
            score += 1.0
    elif "book_reservation" in family:
        if (
            any(positive(term) for term in ("book", "reserve"))
            or has(r"\bmake\s+a\s+reservation\b")
            or has(r"\b(?:want|need|would like)\s+to\s+fly\b")
        ) and not has(r"\bexisting\s+reservation\b"):
            score += 2.0
    elif "cancel" in family:
        if positive("cancel"):
            score += 2.0
    elif "baggage" in family:
        if has(r"\b(?:bag|bags|baggage|luggage)\b"):
            score += 2.5
    elif "update_reservation_flights" in family:
        if has(r"\b(?:change|modif|upgrade|downgrade|switch)\w*\b") and has(r"\b(?:flight|cabin|economy|business|first class|passenger)\w*\b"):
            score += 2.0
    elif "exchange_delivered" in family:
        if any(positive(term) for term in ("exchange", "swap", "replace")):
            score += 3.0
    elif "return_delivered" in family:
        deferred_return = has(r"\breturn\w*\b[^.!?]{0,80}\b(?:later|future)\b")
        requested_return = has(
            r"\b(?:want|need|would like)\b[^.!?]{0,35}\b(?:return|refund)\w*\b",
            r"\bget\s+a\s+refund\b",
            r"\bmoney\s+back\b",
        )
        if not deferred_return and requested_return:
            score += 2.75
    elif "modify_user_address" in family:
        if has(r"\b(?:default|profile|account)\b") and has(r"\baddress\b"):
            score += 3.0
    elif "modify_pending_order_address" in family:
        if has(r"\b(?:shipping|delivery|delivered|deliver|order)\w*\b") and has(r"\baddress\b|\bsuite\b"):
            score += 2.5
    elif "modify_pending_order_items" in family:
        if has(r"\b(?:change|modif|upgrade|switch|add|remove|swap|replace|exchange)\w*\b") and has(r"\b(?:item|order|pending|watch|bottle|laptop|camera|speaker|shirt|t-?shirt)\w*\b"):
            score += 2.25
    return score


def rank_tau_skills(skills: list[TauSkill], query: str, domain: str) -> list[tuple[TauSkill, float]]:
    """Rank domain-safe skills with deterministic, auditable scores."""
    ranked = [(skill, _skill_score(skill, query, domain)) for skill in skills]
    return sorted(ranked, key=lambda item: (item[1], item[0].id), reverse=True)


def select_tau_skills(
    skills: list[TauSkill], query: str, domain: str, *, threshold: float = 0.5, limit: int = 2,
) -> list[tuple[TauSkill, float]]:
    """Select only confidently matched, unique action families."""
    selected: list[tuple[TauSkill, float]] = []
    families: set[str] = set()
    for skill, score in rank_tau_skills(skills, query, domain):
        family = str(skill.metadata.get("action_family") or skill.id)
        if score < threshold or family in families:
            continue
        selected.append((skill, score))
        families.add(family)
        if len(selected) >= limit:
            break
    return selected


def _make_agent(
    environment: Any,
    model: str,
    llm_args: dict[str, Any],
    skills: list[TauSkill],
    *,
    skill_context_budget_chars: int,
    force_skill: bool = False,
):
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import SystemMessage, UserMessage

    class SkillAwareAgent(LLMAgent):
        selected_skill: TauSkill | None = None
        selected_skills: list[TauSkill] = []
        skill_retrieval_trace: list[dict[str, Any]] = []
        retrieval_rank: int = 0
        injected_skill_chars: int = 0

        def _generate_next_message(self, message, state):
            if isinstance(message, UserMessage) and skills:
                if force_skill:
                    matches = [] if self.selected_skills else [(skills[0], float("inf"))]
                else:
                    matches = select_tau_skills(
                        skills, message.content or "", environment.get_domain_name(), limit=2,
                    )
                for matched, _score in matches:
                    if matched in self.selected_skills or self.injected_skill_chars >= skill_context_budget_chars:
                        continue
                    remaining = skill_context_budget_chars - self.injected_skill_chars
                    prompt = _skill_prompt(matched, budget_chars=remaining)
                    if not prompt:
                        continue
                    self.selected_skills.append(matched)
                    self.skill_retrieval_trace.append({"skill_id": matched.id, "score": _score})
                    self.selected_skill = self.selected_skill or matched
                    self.retrieval_rank = 1
                    self.injected_skill_chars += len(prompt)
                    state.system_messages.append(SystemMessage(role="system", content=prompt))
            generated = super()._generate_next_message(message, state)
            # Some OpenAI-compatible endpoints return explanatory prose together
            # with tool calls. τ's half-duplex protocol deliberately rejects
            # mixed messages, so retain the structured calls and discard only
            # the redundant preamble before protocol validation.
            if generated.is_tool_call() and generated.content:
                generated.content = None
            return generated

    agent = SkillAwareAgent(
        tools=environment.get_tools(), domain_policy=environment.get_policy(),
        llm=model, llm_args=llm_args,
    )
    agent.selected_skills = []
    agent.skill_retrieval_trace = []
    agent.injected_skill_chars = 0
    return agent


def _build_simulation(
    domain: str,
    task: Any,
    *,
    skills: list[TauSkill],
    config: TauRunConfig,
    seed: int,
    force_skill: bool = False,
):
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner.build import build_environment, build_user
    environment = build_environment(domain)
    agent = _make_agent(
        environment,
        config.agent_model,
        _llm_args(seed),
        skills,
        skill_context_budget_chars=config.skill_context_budget_chars,
        force_skill=force_skill,
    )
    user = build_user(
        "user_simulator", environment, task, llm=config.user_model,
        llm_args=_llm_args(seed), solo_mode=False,
    )
    return Orchestrator(
        domain=domain, agent=agent, user=user, environment=environment, task=task,
        max_steps=config.max_steps, max_errors=config.max_errors, seed=seed,
        simulation_id=f"{domain}-{task.id}-{seed}", timeout=config.timeout,
        validate_communication=True,
    )


def _message_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json", exclude_none=True)
    return dict(message)


def _usage_totals(messages: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for message in messages:
        usage = message.get("usage") or {}
        for key in totals:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                totals[key] += int(value)
    if not totals["total_tokens"]:
        totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
    return totals


def _reward_components(reward_info: Any) -> dict[str, Any]:
    payload = reward_info.model_dump(mode="json") if hasattr(reward_info, "model_dump") else dict(reward_info or {})
    info = payload.get("reward_breakdown") or payload.get("info") or {}
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    def find(names: tuple[str, ...]) -> float | None:
        queue = [payload]
        while queue:
            value = queue.pop()
            if isinstance(value, dict):
                for key, child in value.items():
                    if str(key).lower() in names and isinstance(child, (int, float, bool)):
                        return float(child)
                    queue.append(child)
            elif isinstance(value, list):
                queue.extend(value)
        return None
    return {
        "reward": float(payload.get("reward") or 0.0),
        "db_reward": find(("db_reward", "db")),
        "communicate_reward": find(("communicate_reward", "communicate")),
        "action_reward": find(("action_reward", "action")),
        "nl_assertion_reward": find(("nl_assertion_reward", "nl_assertion")),
        "raw": payload,
        "has_error_marker": "error" in serialized,
    }


def _trajectory_metrics(messages: list[dict[str, Any]], reference_actions: list[dict[str, Any]]) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    prior_text = ""
    confirmed_writes = 0
    write_count = 0
    tool_errors = 0
    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("content"):
            prior_text = str(message["content"]).lower()
        for call in message.get("tool_calls") or []:
            name = call.get("name") or (call.get("function") or {}).get("name") or ""
            args = call.get("arguments") or (call.get("function") or {}).get("arguments") or {}
            calls.append({"name": name, "arguments": args})
            if is_write_tool(name):
                write_count += 1
                if any(word in prior_text for word in ("confirm", "确认", "proceed", "go ahead", "同意")):
                    confirmed_writes += 1
        if role == "tool" and message.get("error"):
            tool_errors += 1
    ref_names = [str(item.get("name") or "") for item in reference_actions if item.get("requestor", "assistant") == "assistant"]
    call_names = [item["name"] for item in calls]
    reads = [name for name in call_names if not is_write_tool(name)]
    writes = [name for name in call_names if is_write_tool(name)]
    ref_reads = [name for name in ref_names if not is_write_tool(name)]
    ref_writes = [name for name in ref_names if is_write_tool(name)]
    def recall(gold: list[str], pred: list[str]) -> float | None:
        if not gold:
            return None
        remaining = list(pred)
        matched = 0
        for item in gold:
            if item in remaining:
                matched += 1
                remaining.remove(item)
        return matched / len(gold)
    return {
        "tool_calls": len(calls), "read_calls": len(reads), "write_calls": len(writes),
        "tool_errors": tool_errors, "write_confirmation_rate": confirmed_writes / write_count if write_count else None,
        "read_action_recall": recall(ref_reads, reads), "write_action_recall": recall(ref_writes, writes),
        "partial_action_reward": recall(ref_names, call_names),
    }


def run_one_tau_task(
    *, domain: str, task: Any, method: str, trial: int, config: TauRunConfig,
    skill: TauSkill | list[TauSkill] | None = None,
) -> dict[str, Any]:
    if method not in TAU_METHODS:
        raise ValueError(f"unknown method: {method}")
    seed = paired_trial_seed(config.seed, trial, f"{domain}:{task.id}")
    start = time.perf_counter()
    skills = [] if skill is None else skill if isinstance(skill, list) else [skill]
    orchestrator = _build_simulation(
        domain,
        task,
        skills=skills,
        config=config,
        seed=seed,
        force_skill=method == "manual_skill",
    )
    initial_hash = orchestrator.environment.get_db_hash()
    error = None
    try:
        import tau2.evaluator.evaluator_nl_assertions as nl_module
        nl_module.DEFAULT_LLM_NL_ASSERTIONS = config.evaluator_model
        nl_module.DEFAULT_LLM_NL_ASSERTIONS_ARGS = _llm_args(seed)
        from tau2.runner.simulation import run_simulation
        simulation = run_simulation(orchestrator)
        reward = _reward_components(simulation.reward_info)
        messages = [_message_dict(item) for item in simulation.messages]
        termination = str(simulation.termination_reason)
        usage = _usage_totals(messages)
        agent_cost = float(simulation.agent_cost or 0.0)
        user_cost = float(simulation.user_cost or 0.0)
        if termination.lower().endswith(("agent_error", "user_error", "infrastructure_error")):
            error = f"official_termination: {termination}"
    except Exception as exc:
        reward = {"reward": 0.0, "db_reward": None, "communicate_reward": None,
                  "action_reward": None, "nl_assertion_reward": None, "raw": {}, "has_error_marker": True}
        messages = [_message_dict(item) for item in orchestrator.get_messages()]
        termination = "exception"
        usage = {}
        agent_cost = user_cost = 0.0
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - start
    final_hash = orchestrator.environment.get_db_hash()
    reference = [item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
                 for item in (task.evaluation_criteria.actions or [])]
    trajectory = _trajectory_metrics(messages, reference)
    selected_skill = getattr(orchestrator.agent, "selected_skill", None)
    selected_skills = list(getattr(orchestrator.agent, "selected_skills", []) or [])
    selected_tools = {
        str(step.get("tool") or "")
        for candidate in selected_skills
        for step in (candidate.read_steps + candidate.write_steps)
    }
    reference_tools = {str(item.get("name") or "") for item in reference}
    retrieval_hit = bool(selected_skill and selected_tools & reference_tools)
    return {
        "key": simulation_key(domain, method, trial, str(task.id)),
        "task_id": str(task.id), "domain": domain, "method": method, "trial": trial,
        "seed": seed, "reward": reward["reward"], "db_reward": reward["db_reward"],
        "communicate_reward": reward["communicate_reward"], "action_reward": reward["action_reward"],
        "nl_assertion_reward": reward["nl_assertion_reward"], "passed": reward["reward"] == 1.0,
        "initial_state_hash": initial_hash, "final_state_hash": final_hash,
        "termination_reason": termination, "elapsed_seconds": elapsed,
        "trajectory": messages, "usage": usage, "skill_id": selected_skill.id if selected_skill else None,
        "skill_ids": [item.id for item in selected_skills],
        "skill_retrieval_trace": list(getattr(orchestrator.agent, "skill_retrieval_trace", []) or []),
        "skill_version": selected_skill.version if selected_skill else None, "error": error, **trajectory,
        "skill_context_budget_chars": config.skill_context_budget_chars,
        "skill_context_chars": int(getattr(orchestrator.agent, "injected_skill_chars", 0)),
        "skill_candidates": len(skills), "skill_retrieval_rank": 1 if retrieval_hit else 0,
        "skill_retrieval_recall_at_1": 1 if retrieval_hit else 0,
        "wrong_domain_skill_injected": bool(selected_skill and selected_skill.domain != domain),
        "official_reward": reward["raw"], "agent_cost": agent_cost, "user_cost": user_cost,
        "total_cost": agent_cost + user_cost,
        # LiteLLM reports zero for unknown/custom model aliases even though the
        # endpoint was called. Preserve that distinction instead of presenting
        # an unavailable price as a genuinely free run.
        "cost_available": bool(agent_cost or user_cost) or usage.get("total_tokens", 0) == 0,
        "task_category": (
            "refusal" if not reference else "write" if any(is_write_tool(str(item.get("name") or "")) for item in reference)
            else "read"
        ),
    }


def bootstrap_ci(values: list[float], *, seed: int = 42, samples: int = 2000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [means[math.floor(0.025 * (samples - 1))], means[math.floor(0.975 * (samples - 1))]]


def summarize_tau_rows(rows: list[dict[str, Any]], *, expected_keys: set[str] | None = None) -> dict[str, Any]:
    if expected_keys is not None:
        actual = {row["key"] for row in rows}
        missing = sorted(expected_keys - actual)
        extra = sorted(actual - expected_keys)
    else:
        missing, extra = [], []
    grouped: dict[str, Any] = {}
    for method in TAU_METHODS:
        selected = [row for row in rows if row["method"] == method]
        if not selected:
            continue
        rewards = [float(row["reward"]) for row in selected]
        by_domain = {
            domain: statistics.fmean(float(row["reward"]) for row in selected if row["domain"] == domain)
            for domain in sorted({row["domain"] for row in selected})
        }
        task_trials: dict[str, list[bool]] = {}
        for row in selected:
            task_trials.setdefault(f"{row['domain']}:{row['task_id']}", []).append(bool(row["passed"]))
        cost_available = all(bool(row.get("cost_available")) for row in selected)
        total_cost = (
            sum(float(row.get("total_cost") or 0.0) for row in selected)
            if cost_available
            else None
        )
        grouped[method] = {
            "runs": len(selected), "successes": sum(rewards), "task_success": statistics.fmean(rewards),
            "reward_ci95": bootstrap_ci(rewards), "both_trials_success": statistics.fmean(all(v) for v in task_trials.values()),
            "domain_success": by_domain, "errors": sum(bool(row.get("error")) for row in selected),
            "mean_seconds": statistics.fmean(float(row["elapsed_seconds"]) for row in selected),
            "mean_tool_calls": statistics.fmean(float(row["tool_calls"]) for row in selected),
            "mean_tool_errors": statistics.fmean(float(row.get("tool_errors") or 0) for row in selected),
            "mean_tokens": statistics.fmean(float((row.get("usage") or {}).get("total_tokens", 0)) for row in selected),
            "mean_skill_context_chars": statistics.fmean(float(row.get("skill_context_chars") or 0) for row in selected),
            "skill_injection_rate": statistics.fmean(bool(row.get("skill_ids") or row.get("skill_id")) for row in selected),
            "wrong_domain_skill_rate": statistics.fmean(bool(row.get("wrong_domain_skill_injected")) for row in selected),
            "total_cost": total_cost,
            "cost_available": cost_available,
            "cost_per_success": (
                total_cost / sum(rewards)
                if total_cost is not None and sum(rewards) else None
            ),
            "p50_seconds": statistics.median(float(row["elapsed_seconds"]) for row in selected),
            "p95_seconds": sorted(float(row["elapsed_seconds"]) for row in selected)[max(0, math.ceil(len(selected) * 0.95) - 1)],
            "category_success": {
                category: statistics.fmean(float(row["reward"]) for row in selected if row.get("task_category") == category)
                for category in sorted({row.get("task_category") for row in selected if row.get("task_category")})
            },
            "db_reward": _optional_mean(row.get("db_reward") for row in selected),
            "communicate_reward": _optional_mean(row.get("communicate_reward") for row in selected),
            "write_confirmation_rate": _optional_mean(row.get("write_confirmation_rate") for row in selected),
        }
    errors = sum(bool(row.get("error")) for row in rows)
    return {"methods": grouped, "rows": len(rows), "missing_keys": missing, "extra_keys": extra,
            "errors": errors, "complete": not missing and not extra,
            "acceptance_ready": not missing and not extra and errors == 0}


def load_skill(path: str | Path | None) -> TauSkill | None:
    if path is None:
        return None
    return TauSkill(**json.loads(Path(path).read_text(encoding="utf-8")))


def _optional_mean(values: Iterable[Any]) -> float | None:
    valid = [float(value) for value in values if isinstance(value, (int, float))]
    return statistics.fmean(valid) if valid else None
