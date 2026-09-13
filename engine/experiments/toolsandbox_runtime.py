"""DeepSeek-compatible ToolSandbox runtime and reproducible reporting."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import traceback
import random
import multiprocessing as mp
import queue as queue_module
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, cast

from .toolsandbox_skills import BGEPolicyRetriever, METHODS, ToolSandboxSkill, load_skill_library, retrieve_skills

_SCENARIO_CACHE: dict[str, Any] | None = None


class ExcessiveParallelToolCallsError(RuntimeError):
    """Agent emitted a factorially unsafe number of parallel tool calls."""


def _sanitize_tool_call_ids(response: Any) -> Any:
    """ToolSandbox embeds call IDs in Python identifiers; DeepSeek IDs may contain '-'."""
    for choice in getattr(response, "choices", ()):
        for call in getattr(choice.message, "tool_calls", None) or ():
            call.id = re.sub(r"[^A-Za-z0-9_]", "_", call.id)
            if not call.id or call.id[0].isdigit():
                call.id = "call_" + call.id
    return response


def load_tasks(path: Path, split: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row["split"] == split]


def initial_state_hash(scenario: Any) -> str:
    payload = json.dumps(
        scenario.starting_context.to_dict(serialize_console=False),
        ensure_ascii=False, sort_keys=True, default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _usage(response: Any) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0
    return int(usage.prompt_tokens or 0), int(usage.completion_tokens or 0), int(usage.total_tokens or 0)


def _trajectory_metrics(directory: Path) -> dict[str, Any]:
    conversation_path = directory / "conversation.json"
    context_path = directory / "execution_context.json"
    conversation = json.loads(conversation_path.read_text(encoding="utf-8"))
    calls = [call for message in conversation for call in message.get("tool_calls", [])]
    names = [str(call.get("function", {}).get("name") or "") for call in calls]
    read_prefixes = ("get_", "search_", "find_", "list_", "timestamp_to_")
    read_calls = sum(name.startswith(read_prefixes) for name in names)
    context = json.loads(context_path.read_text(encoding="utf-8"))
    state_payload = json.dumps(context.get("_dbs", {}), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return {
        "tool_calls": len(names), "read_tool_calls": read_calls,
        "write_tool_calls": len(names) - read_calls, "tool_names": names,
        "final_state_hash": hashlib.sha256(state_payload).hexdigest(),
        "trajectory_sha256": hashlib.sha256(conversation_path.read_bytes()).hexdigest(),
    }


def _incomplete_trajectory_metrics(directory: Path, diagnostic: str) -> dict[str, Any]:
    """Persist an auditable terminal artifact when official evaluation was interrupted."""
    context_path = directory / "execution_context.json"
    if not context_path.exists():
        return {}
    context = json.loads(context_path.read_text(encoding="utf-8"))
    state_payload = json.dumps(context.get("_dbs", {}), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    failure_path = directory / "failure_trajectory.json"
    failure_payload = {
        "status": "agent_failure_before_official_evaluation",
        "diagnostic": diagnostic,
        "execution_context": context_path.name,
    }
    failure_path.write_text(json.dumps(failure_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "final_state_hash": hashlib.sha256(state_payload).hexdigest(),
        "trajectory_sha256": hashlib.sha256(failure_path.read_bytes()).hexdigest(),
        "trajectory_artifact": failure_path.name,
    }


def _bootstrap_ci(values: list[float], seed: int = 20260909, samples: int = 2000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return [round(means[int(samples * 0.025)], 6), round(means[int(samples * 0.975)], 6)]


def _retrieval_metrics(skills: list[ToolSandboxSkill], task: dict[str, Any], events: list[dict[str, Any]],
                       budget: int) -> dict[str, Any]:
    if not skills:
        return {"expected_skill_ids": [], "retrieved_skill_ids": [], "skill_recall": None,
                "skill_mrr": None, "skill_false_positives": None, "skill_rejection_correct": None}
    _, expected_trace = retrieve_skills(skills, task["id"].replace("_", " "), max_chars=budget)
    expected = [item["skill_id"] for item in expected_trace if item["accepted"]]
    retrieved = []
    for event in events:
        for item in event.get("candidates", []):
            if item.get("accepted") and item["skill_id"] not in retrieved:
                retrieved.append(item["skill_id"])
    hits = [skill_id for skill_id in retrieved if skill_id in expected]
    rank = next((i for i, skill_id in enumerate(retrieved, 1) if skill_id in expected), 0)
    return {
        "expected_skill_ids": expected, "retrieved_skill_ids": retrieved,
        "skill_recall": len(hits) / len(expected) if expected else None,
        "skill_mrr": 1.0 / rank if rank else (0.0 if expected else None),
        "skill_false_positives": len([skill_id for skill_id in retrieved if skill_id not in expected]),
        "skill_rejection_correct": (not retrieved) if not expected else None,
    }


def stable_role_seed(task_id: str, trial: int, role: str) -> int:
    raw = f"{task_id}|{trial}|{role}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16) % 2_147_483_647


def make_roles(skills: list[ToolSandboxSkill], model: str, skill_budget_chars: int,
               *, agent_seed: int, user_seed: int, retrieval_backend: str = "bge_m3"):
    from openai import NOT_GIVEN, OpenAI
    from tool_sandbox.common.execution_context import RoleType
    from tool_sandbox.roles.execution_environment import ExecutionEnvironment
    from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent
    from tool_sandbox.roles.openai_api_user import OpenAIAPIUser

    if model.startswith("deepseek-") and os.environ.get("DEEPSEEK_API_KEY"):
        api_key = os.environ["DEEPSEEK_API_KEY"]
        base_url = "https://api.deepseek.com"
        endpoint_kind = "deepseek_direct"
    elif os.environ.get("OPENAI_BASE_URL") and os.environ.get("OPENAI_API_KEY"):
        # Project terminology calls this the cloud connection.  It is an
        # OpenAI-compatible endpoint and is preferred over any local/edge
        # fallback for the experiment.
        api_key = os.environ["OPENAI_API_KEY"]
        base_url = os.environ["OPENAI_BASE_URL"].strip().rstrip("/")
        endpoint_kind = "cloud_openai_compatible"
    elif os.environ.get("EDGE_OLLAMA_BASE_URL") and os.environ.get("EDGE_OLLAMA_API_KEY"):
        api_key = os.environ["EDGE_OLLAMA_API_KEY"]
        base_url = os.environ["EDGE_OLLAMA_BASE_URL"].strip().rstrip("/")
        endpoint_kind = "edge_openai_compatible"
    else:
        raise RuntimeError("no DeepSeek, cloud, or edge OpenAI-compatible model connection is configured")

    bge_retriever = BGEPolicyRetriever(skills) if skills and retrieval_backend == "bge_m3" else None
    if skills and retrieval_backend != "bge_m3":
        raise ValueError("ToolSandbox skill experiments require retrieval_backend='bge_m3'")

    class DeepSeekAgent(OpenAIAPIAgent):
        model_name = model

        def __init__(self):
            self.openai_client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=2)
            self.prompt_tokens = self.completion_tokens = self.total_tokens = self.calls = 0
            self.retrieval_events = []
            self.anchor_query = ""

        def model_inference(self, openai_messages, openai_tools):
            messages = list(openai_messages)
            query = "\n".join(str(item.get("content") or "") for item in messages[-4:])
            if not self.anchor_query:
                self.anchor_query = query
            skill_text, trace = retrieve_skills(
                skills, query, max_chars=skill_budget_chars, anchor_query=self.anchor_query,
                bge_retriever=bge_retriever,
            )
            self.retrieval_events.append({"call": self.calls + 1, "query": query[-2000:], "candidates": trace, "injected_chars": len(skill_text)})
            if skill_text:
                messages = [{"role": "system", "content": "Retrieved procedural skills:\n" + skill_text}] + messages
            kwargs = {"model": self.model_name, "messages": messages, "temperature": 0, "seed": agent_seed,
                      "extra_body": {"thinking": {"type": "disabled"}}}
            if openai_tools is not NOT_GIVEN:
                kwargs["tools"] = openai_tools
            response = _sanitize_tool_call_ids(self.openai_client.chat.completions.create(**kwargs))
            p, c, t = _usage(response)
            self.prompt_tokens += p; self.completion_tokens += c; self.total_tokens += t; self.calls += 1
            tool_calls = getattr(response.choices[0].message, "tool_calls", None) or ()
            if len(tool_calls) > 8:
                raise ExcessiveParallelToolCallsError(
                    f"agent emitted {len(tool_calls)} parallel tool calls; safe maximum is 8"
                )
            return response

    class DeepSeekUser(OpenAIAPIUser):
        model_name = model

        def __init__(self):
            self.openai_client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=2)
            self.prompt_tokens = self.completion_tokens = self.total_tokens = self.calls = 0

        def model_inference(self, openai_messages, openai_tools):
            kwargs = {"model": self.model_name, "messages": openai_messages, "temperature": 0, "seed": user_seed,
                      "extra_body": {"thinking": {"type": "disabled"}}}
            if openai_tools is not NOT_GIVEN:
                kwargs["tools"] = openai_tools
            response = _sanitize_tool_call_ids(self.openai_client.chat.completions.create(**kwargs))
            p, c, t = _usage(response)
            self.prompt_tokens += p; self.completion_tokens += c; self.total_tokens += t; self.calls += 1
            return response

    agent, user = DeepSeekAgent(), DeepSeekUser()
    return {
        RoleType.AGENT: agent,
        RoleType.USER: user,
        RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(),
    }, agent, user


def run_task(task: dict[str, Any], method: str, trial: int, output_root: Path, model: str,
             *, skill_root: Path, skill_budget_chars: int,
             skills_override: list[ToolSandboxSkill] | None = None) -> dict[str, Any]:
    from tool_sandbox.common.tool_discovery import ToolBackend
    from tool_sandbox.scenarios import named_scenarios

    global _SCENARIO_CACHE
    if _SCENARIO_CACHE is None:
        _SCENARIO_CACHE = named_scenarios(ToolBackend.DEFAULT)
    scenario = _SCENARIO_CACHE[task["id"]]
    key = f'{task["split"]}/{method}/trial-{trial}/{task["id"]}'
    skills = skills_override if skills_override is not None else load_skill_library(skill_root, method)
    agent_seed = stable_role_seed(task["id"], trial, "agent")
    user_seed = stable_role_seed(task["id"], trial, "user")
    roles, agent, user = make_roles(skills, model, skill_budget_chars,
                                    agent_seed=agent_seed, user_seed=user_seed)
    started = time.perf_counter()
    row = {"key": key, "task_id": task["id"], "family": task["family"], "split": task["split"],
           "primary_group": task["primary_group"], "method": method, "trial": trial,
           "initial_state_hash": initial_state_hash(scenario), "skill_library_size": len(skills),
           "skill_library_versions": {skill.skill_id: skill.version for skill in skills},
           "skill_budget_chars": skill_budget_chars, "agent_seed": agent_seed, "user_seed": user_seed,
           "model": model, "endpoint_kind": "configured_in_make_roles"}
    try:
        artifact_name = key.replace("/", "__")
        result = scenario.play_and_evaluate(roles, output_root, artifact_name)
        ev = result.evaluation_result
        row.update({"reward": float(ev.similarity), "milestone_similarity": float(ev.milestone_similarity),
                    "minefield_similarity": float(ev.minefield_similarity), "turn_count": int(ev.turn_count),
                    "error": None, "traceback": None})
        row.update(_trajectory_metrics(output_root / "trajectories" / artifact_name))
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        invalid_agent_action = (
            isinstance(exc, KeyError) and "Agent tool call" in str(exc)
        ) or isinstance(exc, ExcessiveParallelToolCallsError)
        row.update({"reward": 0.0, "milestone_similarity": 0.0, "minefield_similarity": 0.0,
                    "turn_count": scenario.max_messages,
                    "error": None if invalid_agent_action else message,
                    "failure_reason": "invalid_or_unauthorized_tool" if invalid_agent_action else None,
                    "diagnostic": message if invalid_agent_action else None,
                    "traceback": traceback.format_exc()})
    finally:
        for role in roles.values():
            role.teardown()
    artifact_dir = output_root / "trajectories" / key.replace("/", "__")
    if "final_state_hash" not in row and (artifact_dir / "conversation.json").exists() and (artifact_dir / "execution_context.json").exists():
        row.update(_trajectory_metrics(artifact_dir))
    if "final_state_hash" not in row:
        row.update(_incomplete_trajectory_metrics(
            artifact_dir, str(row.get("diagnostic") or row.get("error") or "incomplete trajectory")
        ))
    row.update({"seconds": round(time.perf_counter() - started, 6),
                "agent_calls": agent.calls, "user_calls": user.calls,
                "agent_tokens": agent.total_tokens, "user_tokens": user.total_tokens,
                "total_tokens": agent.total_tokens + user.total_tokens,
                "skill_retrieval_events": agent.retrieval_events,
                "skill_injections": sum(bool(x["injected_chars"]) for x in agent.retrieval_events),
                "skill_injected_chars": sum(x["injected_chars"] for x in agent.retrieval_events)})
    row.update(_retrieval_metrics(skills, task, agent.retrieval_events, skill_budget_chars))
    return row


def _run_task_worker(queue: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    try:
        queue.put((True, run_task(*args, **kwargs)))
    except BaseException as exc:  # child-process boundary must report all exits
        queue.put((False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


def run_task_with_hard_timeout(
    task: dict[str, Any], method: str, trial: int, output_root: Path, model: str, *,
    skill_root: Path, skill_budget_chars: int, timeout_seconds: float,
) -> dict[str, Any]:
    """Run one task in a killable child process; required on Windows where signals cannot bound work."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    args = (task, method, trial, output_root, model)
    kwargs = {"skill_root": skill_root, "skill_budget_chars": skill_budget_chars}
    process = ctx.Process(target=_run_task_worker, args=(queue, args, kwargs))
    process.start()
    try:
        # Read while the child is alive. Joining first can deadlock when the
        # serialized trajectory is larger than the multiprocessing pipe buffer.
        ok, payload = queue.get(timeout=timeout_seconds)
    except queue_module.Empty:
        process.terminate()
        process.join(10)
        return {
            "key": f'{task["split"]}/{method}/trial-{trial}/{task["id"]}',
            "task_id": task["id"], "family": task["family"], "split": task["split"],
            "primary_group": task["primary_group"], "method": method, "trial": trial,
            "reward": 0.0, "milestone_similarity": 0.0, "minefield_similarity": 0.0,
            "error": f"TaskTimeoutError: exceeded {timeout_seconds:g} seconds",
            "failure_reason": "task_hard_timeout", "seconds": timeout_seconds,
            "initial_state_hash": None, "final_state_hash": None, "trajectory_sha256": None,
            "total_tokens": 0,
        }
    process.join(10)
    if process.is_alive():
        process.terminate()
        process.join(10)
    if not ok:
        return {
            "key": f'{task["split"]}/{method}/trial-{trial}/{task["id"]}',
            "task_id": task["id"], "family": task["family"], "split": task["split"],
            "primary_group": task["primary_group"], "method": method, "trial": trial,
            "reward": 0.0, "milestone_similarity": 0.0, "minefield_similarity": 0.0,
            "error": f"TaskWorkerError: {payload}", "failure_reason": "task_worker_exception",
            "seconds": 0.0, "total_tokens": 0,
        }
    return cast(dict[str, Any], payload)


def repair_incomplete_failure_artifacts(rows: list[dict[str, Any]], output_root: Path) -> int:
    """Backfill hashes only for valid agent failures with an official saved execution context."""
    repaired = 0
    for row in rows:
        if row.get("final_state_hash") and row.get("trajectory_sha256"):
            continue
        if row.get("failure_reason") != "invalid_or_unauthorized_tool":
            continue
        directory = output_root / "trajectories" / row["key"].replace("/", "__")
        metrics = _incomplete_trajectory_metrics(directory, str(row.get("diagnostic") or "invalid tool call"))
        if metrics:
            row.update(metrics)
            repaired += 1
    return repaired


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("error")]
    methods = {}
    for method in METHODS:
        subset = [row for row in valid if row["method"] == method]
        events = [event for row in subset for event in row.get("skill_retrieval_events", [])]
        accepted = [candidate for event in events for candidate in event.get("candidates", []) if candidate.get("accepted")]
        def avg_optional(field):
            values = [float(row[field]) for row in subset if row.get(field) is not None]
            return round(sum(values) / len(values), 6) if values else None
        methods[method] = {
            "runs": len(subset), "successes": sum(row["reward"] == 1 for row in subset),
            "task_success": round(sum(row["reward"] == 1 for row in subset) / len(subset), 6) if subset else None,
            "mean_reward": round(sum(row["reward"] for row in subset) / len(subset), 6) if subset else None,
            "reward_95ci": _bootstrap_ci([float(row["reward"]) for row in subset]),
            "task_success_95ci": _bootstrap_ci([float(row["reward"] == 1) for row in subset]),
            "mean_tokens": round(sum(row["total_tokens"] for row in subset) / len(subset), 2) if subset else None,
            "mean_seconds": round(sum(row["seconds"] for row in subset) / len(subset), 2) if subset else None,
            "mean_skill_injected_chars": round(sum(row.get("skill_injected_chars", 0) for row in subset) / len(subset), 2) if subset else None,
            "skill_injection_rate": round(sum(bool(row.get("skill_injections")) for row in subset) / len(subset), 6) if subset else None,
            "accepted_skill_events": len(accepted),
            "mean_tool_calls": round(sum(row.get("tool_calls", 0) for row in subset) / len(subset), 3) if subset else None,
            "mean_read_tool_calls": round(sum(row.get("read_tool_calls", 0) for row in subset) / len(subset), 3) if subset else None,
            "mean_write_tool_calls": round(sum(row.get("write_tool_calls", 0) for row in subset) / len(subset), 3) if subset else None,
            "skill_retrieval_recall": avg_optional("skill_recall"),
            "skill_retrieval_mrr": avg_optional("skill_mrr"),
            "mean_skill_false_positives": avg_optional("skill_false_positives"),
            "inapplicable_skill_rejection_rate": avg_optional("skill_rejection_correct"),
        }
    comparisons = {}
    baseline = {(row["task_id"], row["trial"]): row for row in valid
                if row["method"] == "no_skill" and "task_id" in row and "trial" in row}
    for method in METHODS[1:]:
        pairs = [(baseline[(row["task_id"], row["trial"])], row) for row in valid
                 if row["method"] == method and "task_id" in row and "trial" in row
                 and (row["task_id"], row["trial"]) in baseline]
        deltas = [candidate["reward"] - base["reward"] for base, candidate in pairs]
        comparisons[f"{method}_vs_no_skill"] = {
            "paired_runs": len(pairs), "mean_reward_delta": round(sum(deltas) / len(deltas), 6) if deltas else None,
            "reward_delta_95ci": _bootstrap_ci(deltas) if deltas else None,
        }
    return {"expected_runs": len(rows), "unique_keys": len({row["key"] for row in rows}),
            "errors": len(rows) - len(valid), "methods": methods,
            "paired_comparisons": comparisons,
            "group_method": {f"{g}/{m}": round(sum(r["reward"] for r in valid if r["primary_group"] == g and r["method"] == m) /
                max(1, sum(1 for r in valid if r["primary_group"] == g and r["method"] == m)), 6)
                for g in sorted({r["primary_group"] for r in rows}) for m in METHODS}}


def save_results(rows: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = summarize(rows)
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
