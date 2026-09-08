"""End-to-end long-memory evaluation.

Every method is evaluated with the same fixed QA solver: history is written or
made visible, the method returns its context, and the solver answers from that
context only. Retrieval metrics are secondary diagnostics; when a dataset does
not expose gold evidence, they use answer-text containment and are labelled as
such in the row metadata.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from engine.config import load_settings
from engine.modules.bge_local import resolve_bge_m3_cache_dir, resolve_bge_m3_model_path
from engine.modules.context.budget import rough_token_count
from engine.modules.memory import (
    BGEM3EmbeddingModel,
    HybridTieredMemoryStore,
    HashingEmbeddingModel,
    MemoryContext,
    MemoryItem,
    MemoryScope,
    RetrievalMode,
    build_default_memory_judge,
)

from .reports import ExperimentReport
from .types import ExperimentRow, MemoryExample


QA_SYSTEM_PROMPT = """Answer the question using only the supplied memory context.
First identify the relevant facts internally, reconcile repeated or conflicting
facts by preferring the latest explicitly supported information, and combine
all required facts for multi-part questions. Do not expose that reasoning.
Return the shortest direct answer only. Reply exactly INSUFFICIENT_INFORMATION
only when the supplied evidence truly cannot determine the answer; do not use
it merely because the answer requires a simple inference, date calculation, or
combination of multiple supplied facts."""

QA_JUDGE_SYSTEM_PROMPT = """You are an impartial judge for a long-term memory QA benchmark.
Return exactly yes or no. Do not explain your decision."""


@dataclass
class MemoryExperimentConfig:
    backend: str = "engine"
    top_k: int = 5
    candidate_k: int = 20
    context_token_budget: int = 12000
    evidence_chunk_chars: int = 1800
    evidence_chunk_overlap_chars: int = 240
    max_chunks_per_session: int = 2
    output_root: str = "runs/experiments/memory"
    clean: bool = True
    use_llm_judge: bool = True
    mem0_infer: bool = True
    mem0_threshold: float = 0.0
    retrieval_mode: str = RetrievalMode.HYBRID.value
    cascade_read: bool = True
    enable_memory_update: bool = True
    long_text_threshold: int = 2000
    qa_solver: str = "llm"
    qa_model: str = ""
    qa_judge_model: str = ""
    qa_timeout_seconds: float = 90.0
    embedding_backend: str = "bge_m3"
    bge_batch_size: int = 32
    history_chunk_chars: int = 0


def run_memory_experiment(
    examples: List[MemoryExample], config: MemoryExperimentConfig | None = None
) -> ExperimentReport:
    cfg = config or MemoryExperimentConfig()
    if cfg.qa_solver not in {"llm", "extractive"}:
        raise ValueError("qa_solver must be 'llm' or 'extractive'")
    if cfg.backend in {"engine", "ours"}:
        return _run_engine_memory(examples, cfg)
    if cfg.backend == "no_memory":
        return _run_context_baseline(examples, cfg, full_context=False)
    if cfg.backend == "full_context":
        return _run_context_baseline(examples, cfg, full_context=True)
    if cfg.backend == "full_context_budgeted":
        return _run_context_baseline(examples, cfg, full_context=True, budgeted=True)
    if cfg.backend == "mem0":
        return _run_mem0_memory(examples, cfg)
    raise ValueError(f"unsupported memory backend: {cfg.backend}")


def _run_engine_memory(examples: List[MemoryExample], cfg: MemoryExperimentConfig) -> ExperimentReport:
    root = Path(cfg.output_root) / "engine_store"
    if cfg.clean and root.exists():
        shutil.rmtree(root)
    judge = build_default_memory_judge() if cfg.use_llm_judge else None
    embedding_model = _build_embedding_model(cfg)
    # Full-corpus BGE encoding is prohibitively expensive for utterance-heavy
    # dialogue histories.  In the no-Judge retrieval ablation, use the cheap
    # hybrid index for candidate recall and BGE-M3 only to rerank candidates.
    # This is explicit in output metadata, rather than a hidden fallback.
    bge_reranker = embedding_model if cfg.embedding_backend == "bge_m3" else None
    index_embedding_model = HashingEmbeddingModel() if bge_reranker is not None else embedding_model
    store = HybridTieredMemoryStore(
        root,
        embedding_model=index_embedding_model,
        memory_llm_judge=judge,
        enable_memory_update=cfg.enable_memory_update,
        long_text_threshold=cfg.long_text_threshold,
    )
    rows: List[ExperimentRow] = []
    started = time.perf_counter()
    try:
        for example in examples:
            indexed_example = _with_chunked_memories(example, cfg.history_chunk_chars)
            context = _engine_memory_context(example)
            write_started = time.perf_counter()
            _write_example_memories(store, indexed_example, context, cfg)
            if cfg.use_llm_judge:
                if judge is not None and getattr(judge, "parse_error_count", 0):
                    raise RuntimeError(
                        "memory update judge returned malformed JSON; stopped to avoid further API spend"
                    )
            write_seconds = time.perf_counter() - write_started
            retrieval_started = time.perf_counter()
            mode = RetrievalMode(cfg.retrieval_mode)
            candidate_k = max(cfg.top_k, cfg.candidate_k)
            if cfg.cascade_read:
                retrieved = store.cascade_read(example.question, context=context, top_k=candidate_k, retrieval_mode=mode)
            else:
                retrieved = store.read(example.question, scope=MemoryScope.PROJECT, context=context, top_k=candidate_k, retrieval_mode=mode)
            retrieval_seconds = time.perf_counter() - retrieval_started
            retrieved_sessions = [_engine_retrieved_text(store, item) for item in retrieved]
            if bge_reranker is not None:
                rerank_started = time.perf_counter()
                retrieved_sessions = _rerank_sessions_with_bge(
                    example.question, retrieved_sessions, bge_reranker
                )
                retrieval_seconds += time.perf_counter() - rerank_started
            # Candidate@K is only a recall stage. The answer generator must
            # never see evidence beyond the declared final Top-K.
            visible, evidence_stats = _select_evidence_chunks(
                example.question,
                retrieved_sessions[:cfg.top_k],
                token_budget=cfg.context_token_budget,
                chunk_chars=cfg.evidence_chunk_chars,
                overlap_chars=cfg.evidence_chunk_overlap_chars,
                max_chunks_per_session=cfg.max_chunks_per_session,
            )
            rows.append(_evaluate_example(
                example, visible, cfg,
                write_seconds=write_seconds, retrieval_seconds=retrieval_seconds,
                retrieval_applicable=True,
                metadata={
                    "backend": cfg.backend,
                    "source": example.source,
                    "expanded_archives": sum(bool(item.raw_ref) for item in retrieved),
                    "candidate_sessions": len(retrieved_sessions),
                    "final_evidence_sessions": min(len(retrieved_sessions), cfg.top_k),
                    "history_units_original": len(example.memories),
                    "history_units_indexed": len(indexed_example.memories),
                    "candidate_retriever": "hashing_hybrid" if bge_reranker is not None else cfg.embedding_backend,
                    "reranker": "bge_m3" if bge_reranker is not None else "none",
                    "evidence_selection": evidence_stats,
                },
                retrieval_evidence_texts=retrieved_sessions,
            ))
    finally:
        action_counts = Counter(store.memory_update_action_counts)
        store.close()
    return ExperimentReport(
        name=f"memory-{cfg.backend}", rows=rows,
        metadata={
            "backend": cfg.backend, "top_k": cfg.top_k, "candidate_k": cfg.candidate_k,
            "embedding_backend": cfg.embedding_backend,
            "embedding_model": type(embedding_model).__name__,
            "candidate_retriever": "hashing_hybrid" if bge_reranker is not None else cfg.embedding_backend,
            "reranker": "bge_m3" if bge_reranker is not None else "none",
            "history_chunk_chars": cfg.history_chunk_chars,
            "context_token_budget": cfg.context_token_budget, "scope": MemoryScope.PROJECT.value,
            "llm_judge_enabled": judge is not None, "retrieval_mode": cfg.retrieval_mode,
            "memory_judge_calls": int(getattr(judge, "call_count", 0)) if judge is not None else 0,
            "memory_judge_parse_errors": int(getattr(judge, "parse_error_count", 0)) if judge is not None else 0,
            "memory_judge_thinking_disabled": bool(getattr(judge, "thinking_disabled", False)) if judge is not None else False,
            "cascade_read": cfg.cascade_read, "enable_memory_update": cfg.enable_memory_update,
            "archive_expand_policy": "expand_candidates_then_select_evidence_chunks",
            "qa_solver": cfg.qa_solver, "qa_model": _reported_model(cfg.qa_model),
            "storage_bytes": _directory_size(root), "memory_update_actions": dict(action_counts),
            "memory_update_effective_actions": int(action_counts.get("UPDATE", 0) + action_counts.get("DELETE", 0)),
            "seconds": round(time.perf_counter() - started, 4),
        },
    )


def _run_context_baseline(
    examples: List[MemoryExample], cfg: MemoryExperimentConfig, *, full_context: bool, budgeted: bool = False
) -> ExperimentReport:
    rows: List[ExperimentRow] = []
    started = time.perf_counter()
    for example in examples:
        retrieval_started = time.perf_counter()
        visible = list(example.memories) if full_context else []
        if budgeted:
            visible = _recent_context_within_budget(visible, cfg.context_token_budget)
        retrieval_seconds = time.perf_counter() - retrieval_started
        rows.append(_evaluate_example(
            example, visible, cfg, write_seconds=0.0, retrieval_seconds=retrieval_seconds,
            retrieval_applicable=False,
            metadata={
                "backend": cfg.backend,
                "source": example.source,
                "retrieval_metrics": "not_applicable_for_context_control",
                "context_policy": "recent_history_truncation" if budgeted else "unbounded_full_history",
            },
        ))
    return ExperimentReport(
        name=f"memory-{cfg.backend}", rows=rows,
        metadata={
            "backend": cfg.backend, "top_k": cfg.top_k, "context_token_budget": cfg.context_token_budget,
            "qa_solver": cfg.qa_solver,
            "qa_model": _reported_model(cfg.qa_model), "storage_bytes": 0,
            "evaluator_scope": "end-to-end QA; retrieval metrics are N/A for context controls",
            "seconds": round(time.perf_counter() - started, 4),
        },
    )


def _run_mem0_memory(examples: List[MemoryExample], cfg: MemoryExperimentConfig) -> ExperimentReport:
    root = Path(cfg.output_root) / "mem0_store"
    if cfg.clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    os.environ["MEM0_DIR"] = str(root / "home")
    settings = load_settings()
    from mem0 import Memory

    memory = Memory.from_config(_mem0_config(root, settings))
    _disable_mem0_thinking(memory)
    rows: List[ExperimentRow] = []
    started = time.perf_counter()
    for example in examples:
        indexed_example = _with_chunked_memories(example, cfg.history_chunk_chars)
        user_id = _mem0_user_id(indexed_example)
        write_started = time.perf_counter()
        for index, text in enumerate(indexed_example.memories):
            memory.add([{"role": "user", "content": text}], user_id=user_id,
                       metadata={"source": example.source, "example_id": example.id, "index": index}, infer=cfg.mem0_infer)
        write_seconds = time.perf_counter() - write_started
        retrieval_started = time.perf_counter()
        result = memory.search(indexed_example.question, top_k=cfg.top_k, filters={"user_id": user_id}, threshold=cfg.mem0_threshold)
        retrieval_seconds = time.perf_counter() - retrieval_started
        result_items = _mem0_results(result)
        rows.append(_evaluate_example(
            indexed_example, [_mem0_memory_text(item) for item in result_items], cfg,
            write_seconds=write_seconds, retrieval_seconds=retrieval_seconds, retrieval_applicable=True,
            metadata={
                "backend": "mem0",
                "source": example.source,
                "retrieval_provenance": "mem0_metadata_index_to_source_session",
                "history_units_original": len(example.memories),
                "history_units_indexed": len(indexed_example.memories),
            },
            retrieval_evidence_texts=_deduplicated_mem0_source_sessions(result_items, indexed_example),
        ))
    return ExperimentReport(
        name="memory-mem0", rows=rows,
        metadata={
            "backend": "mem0", "top_k": cfg.top_k, "infer": cfg.mem0_infer,
            "embedding_backend": "bge_m3", "embedding_model": "BGEM3EmbeddingModel",
            "history_chunk_chars": cfg.history_chunk_chars,
            "thinking_disabled": True,
            "threshold": cfg.mem0_threshold, "qa_solver": cfg.qa_solver,
            "qa_model": _reported_model(cfg.qa_model), "storage_bytes": _directory_size(root),
            "seconds": round(time.perf_counter() - started, 4),
        },
    )


def _evaluate_example(
    example: MemoryExample, retrieved: List[str], cfg: MemoryExperimentConfig, *,
    write_seconds: float, retrieval_seconds: float, retrieval_applicable: bool, metadata: Dict[str, Any],
    retrieval_evidence_texts: List[str] | None = None,
) -> ExperimentRow:
    context = "\n\n".join(retrieved)
    answer_started = time.perf_counter()
    prediction, qa_usage, qa_error = _answer_question(example.question, context, cfg)
    qa_seconds = time.perf_counter() - answer_started
    qa_exact = _exact_match(prediction, example.answer)
    judge_started = time.perf_counter()
    qa_correct, judge_usage, judge_response, judge_error = _judge_qa_answer(
        example, prediction, cfg, qa_error=qa_error
    )
    judge_seconds = time.perf_counter() - judge_started
    # Keep lexical overlap independent of the semantic judge. QA accuracy is
    # the semantic outcome; turning every judge-accepted answer into F1=1.0
    # makes the two metrics redundant and hides partial-answer behavior.
    answer_f1 = _answer_f1(prediction, example.answer)
    metrics: Dict[str, Any] = {
        "qa_acc": int(qa_correct), "qa_exact_match": int(qa_exact),
        "answer_f1": answer_f1, "retrieved": len(retrieved),
        "write_ms": write_seconds * 1000, "retrieval_ms": retrieval_seconds * 1000,
        "qa_ms": qa_seconds * 1000,
        "time_ms": (write_seconds + retrieval_seconds + qa_seconds) * 1000,
        "end_to_end_ms": (write_seconds + retrieval_seconds + qa_seconds + judge_seconds) * 1000,
        "qa_judge_ms": judge_seconds * 1000,
        "qa_judge_input_tokens": int(judge_usage.get("prompt_tokens") or 0),
        "qa_judge_output_tokens": int(judge_usage.get("completion_tokens") or 0),
        "qa_judge_tokens": int(judge_usage.get("total_tokens") or 0),
        "context_tokens": rough_token_count(context),
        "qa_input_tokens": int(qa_usage.get("prompt_tokens") or rough_token_count(_qa_prompt(example.question, context))),
        "qa_output_tokens": int(qa_usage.get("completion_tokens") or rough_token_count(prediction)),
        "tokens": int(qa_usage.get("total_tokens") or rough_token_count(_qa_prompt(example.question, context)) + rough_token_count(prediction)),
    }
    if retrieval_applicable:
        rank, evidence_recall, mode = _evidence_metrics(
            example, retrieval_evidence_texts if retrieval_evidence_texts is not None else retrieved
        )
        metrics.update({
            "rank": rank, "mrr": 1.0 / rank if rank else 0.0,
            "hit_at_1": int(rank == 1), "hit_at_3": int(0 < rank <= 3),
            "hit_at_5": int(0 < rank <= 5), "evidence_recall": evidence_recall,
            # These diagnostics separate retrieval-caused QA failures from
            # generator/judge failures. They are especially important for
            # multi-evidence questions where Hit@5 alone is insufficient.
            "gold_evidence_complete": int(evidence_recall >= 1.0),
            "qa_failure_with_complete_evidence": int(not qa_correct and evidence_recall >= 1.0),
            "qa_failure_with_incomplete_evidence": int(not qa_correct and evidence_recall < 1.0),
        })
        metadata = {**metadata, "evidence_metric": mode}
    if qa_error:
        metadata = {**metadata, "qa_error": qa_error}
    if cfg.qa_solver == "llm":
        metadata = {
            **metadata,
            "qa_evaluator": "longmemeval_llm_judge",
            "qa_judge_model": _reported_model(cfg.qa_judge_model or os.environ.get("OPENAI_JUDGE_MODEL", "")),
            "qa_judge_response": judge_response,
        }
    if judge_error:
        metadata = {**metadata, "qa_judge_error": judge_error}
    return ExperimentRow(
        id=example.id, passed=qa_correct, score=answer_f1, prediction=prediction, expected=example.answer,
        metrics=metrics, metadata={**metadata, "trajectory_id": example.trajectory_id},
    )


def _answer_question(question: str, context: str, cfg: MemoryExperimentConfig) -> Tuple[str, Dict[str, Any], str]:
    if cfg.qa_solver == "extractive":
        return _extractive_answer(context), {}, ""
    try:
        from openai import OpenAI
        settings = load_settings()
        client = OpenAI(api_key=settings.api_key, base_url=settings.base_url, organization=settings.organization, timeout=cfg.qa_timeout_seconds)
        response = client.chat.completions.create(
            model=cfg.qa_model or settings.model,
            messages=[{"role": "system", "content": QA_SYSTEM_PROMPT}, {"role": "user", "content": _qa_prompt(question, context)}],
            temperature=0,
            # Keep generation and judging in the same non-thinking evaluation mode.
            extra_body={"thinking": {"type": "disabled"}},
        )
        usage = getattr(response, "usage", None)
        usage_data = json.loads(usage.model_dump_json()) if usage is not None else {}
        return (response.choices[0].message.content or "").strip(), usage_data, ""
    except Exception as exc:
        return "", {}, f"{type(exc).__name__}: {exc}"


def _qa_prompt(question: str, context: str) -> str:
    return f"Memory context:\n{context or '[empty]'}\n\nQuestion: {question}\nAnswer:"


def _reported_model(requested_model: str) -> str:
    """Persist the actual configured model name without exposing credentials."""
    if requested_model:
        return requested_model
    try:
        return load_settings().model
    except Exception:
        return "UNRESOLVED_MODEL"


def _judge_qa_answer(
    example: MemoryExample, prediction: str, cfg: MemoryExperimentConfig, *, qa_error: str = ""
) -> Tuple[bool, Dict[str, Any], str, str]:
    """Apply the LongMemEval-style semantic judge to an answer.

    The extractive solver is only used by offline smoke tests, so it keeps the
    deterministic exact-match evaluator. Reported LLM runs use the same judge
    for every memory backend.
    """
    if cfg.qa_solver == "extractive":
        return _exact_match(prediction, example.answer), {}, "exact_match", ""
    if qa_error:
        return False, {}, "", "skipped because QA generation failed"
    try:
        from openai import OpenAI

        settings = load_settings()
        client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            organization=settings.organization,
            timeout=cfg.qa_timeout_seconds,
        )
        response = client.chat.completions.create(
            model=cfg.qa_judge_model or os.environ.get("OPENAI_JUDGE_MODEL") or settings.model,
            messages=[
                {"role": "system", "content": QA_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": _qa_judge_prompt(example, prediction)},
            ],
            temperature=0,
            max_tokens=10,
            extra_body={"thinking": {"type": "disabled"}},
        )
        verdict = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        usage_data = json.loads(usage.model_dump_json()) if usage is not None else {}
        if not verdict:
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            return False, usage_data, "", f"empty judge response (finish_reason={finish_reason})"
        return _yes_verdict(verdict), usage_data, verdict, ""
    except Exception as exc:
        return False, {}, "", f"{type(exc).__name__}: {exc}"


def _qa_judge_prompt(example: MemoryExample, prediction: str) -> str:
    question_type = str(example.metadata.get("question_type") or "").lower()
    if example.id.endswith("_abs") or "abstention" in question_type:
        criterion = "The response is correct if it clearly recognizes that the question cannot be answered from the available information."
    elif "temporal" in question_type:
        criterion = "The response is correct if it contains an answer equivalent to the reference. For day-count calculations, an off-by-one answer is acceptable."
    elif "knowledge-update" in question_type or "knowledge_update" in question_type:
        criterion = "The response is correct if it includes the updated information required by the reference; mentioning obsolete information as additional context is allowed."
    elif "preference" in question_type:
        criterion = "The response is correct if it gives a reasonable answer grounded in the user's stated preference; it need not repeat every detail of the reference."
    else:
        criterion = "The response is correct if it contains or is semantically equivalent to the reference answer. Extra explanation is allowed, but a merely partial answer is not."
    return (
        f"Question type: {question_type or 'unspecified'}\n"
        f"Question: {example.question}\n"
        f"Reference answer: {example.answer}\n"
        f"Candidate response: {prediction}\n\n"
        f"Criterion: {criterion}\n"
        "Is the candidate response correct? Answer yes or no."
    )


def _yes_verdict(text: str) -> bool:
    return bool(re.match(r"^\s*yes\b", str(text), flags=re.IGNORECASE))


def _extractive_answer(context: str) -> str:
    """Offline smoke-test solver; do not use it for reported QA numbers."""
    if not context.strip():
        return "INSUFFICIENT_INFORMATION"
    patterns = [r"(?:是|为|使用|需要)\s*([\u4e00-\u9fffA-Za-z0-9_-]+)", r"([\u4e00-\u9fffA-Za-z0-9_-]+)\s*(?:。|，|,|$)"]
    for pattern in patterns:
        match = re.search(pattern, context)
        if match:
            return match.group(1)
    return context.splitlines()[0][:80]


def _evidence_metrics(example: MemoryExample, retrieved: List[str]) -> Tuple[int, float, str]:
    evidence = example.evidence or [example.answer]
    mode = "gold_evidence" if example.evidence else "answer_text_fallback"
    ranks = [_text_rank(item, retrieved) for item in evidence if _norm(item)]
    positive = [rank for rank in ranks if rank]
    return (min(positive) if positive else 0, len(positive) / len(ranks) if ranks else 0.0, mode)


def _select_evidence_chunks(
    question: str,
    sessions: List[str],
    *,
    token_budget: int,
    chunk_chars: int,
    overlap_chars: int,
    max_chunks_per_session: int,
) -> Tuple[List[str], Dict[str, int]]:
    """Select QA evidence from retrieved sessions under a fixed context budget."""
    if token_budget <= 0:
        return list(sessions), {
            "candidate_sessions": len(sessions),
            "selected_chunks": len(sessions),
            "selected_tokens": rough_token_count("\n".join(sessions)),
        }
    candidates: List[Tuple[float, int, int, str]] = []
    for session_rank, session in enumerate(sessions):
        chunks = _split_evidence_chunks(session, chunk_chars, overlap_chars)
        scored = [(_chunk_relevance(question, chunk), chunk_index, chunk) for chunk_index, chunk in enumerate(chunks)]
        for score, chunk_index, chunk in sorted(scored, key=lambda item: (-item[0], item[1]))[:max(1, max_chunks_per_session)]:
            # Session retrieval provides semantic recall; this score chooses a
            # compact evidence span inside each retrieved raw session.
            candidates.append((score + 1.0 / (session_rank + 1), session_rank, chunk_index, chunk))
    selected: List[str] = []
    used_tokens = 0
    for _, session_rank, chunk_index, chunk in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
        remaining = token_budget - used_tokens
        if remaining <= 0:
            break
        chunk_tokens = rough_token_count(chunk)
        if chunk_tokens > remaining:
            chunk = chunk[: max(1, remaining * 4)]
            chunk_tokens = rough_token_count(chunk)
        selected.append(f"[session_rank={session_rank + 1}]\n{chunk}")
        used_tokens += chunk_tokens
    selected = _fit_context_to_budget(selected, token_budget)
    return selected, {
        "candidate_sessions": len(sessions),
        "selected_chunks": len(selected),
        "selected_tokens": rough_token_count("\n\n".join(selected)),
    }


def _recent_context_within_budget(memories: List[str], token_budget: int) -> List[str]:
    """Conventional fixed-budget Full-Context baseline using recent history."""
    if token_budget <= 0:
        return list(memories)
    selected: List[str] = []
    used_tokens = 0
    for memory in reversed(memories):
        remaining = token_budget - used_tokens
        if remaining <= 0:
            break
        memory_tokens = rough_token_count(memory)
        if memory_tokens > remaining:
            memory = memory[-max(1, remaining * 4):]
            memory_tokens = rough_token_count(memory)
        selected.append(memory)
        used_tokens += memory_tokens
    return _fit_context_to_budget(list(reversed(selected)), token_budget)


def _fit_context_to_budget(parts: List[str], token_budget: int) -> List[str]:
    """Defensively enforce the reported budget including separators and labels."""
    if token_budget <= 0:
        return list(parts)
    fitted = list(parts)
    while fitted and rough_token_count("\n\n".join(fitted)) > token_budget:
        if len(fitted) > 1:
            fitted.pop()
            continue
        text = fitted[0]
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if rough_token_count(text[:mid]) <= token_budget:
                low = mid
            else:
                high = mid - 1
        fitted = [text[:low]] if low else []
    return fitted


def _split_evidence_chunks(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    if len(text) <= chunk_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    step = max(1, chunk_chars - max(0, overlap_chars))
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end), text.rfind("。", start, end))
            if boundary > start + chunk_chars // 2:
                end = boundary + 1
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(start + step, end - max(0, overlap_chars))
    return chunks


def _chunk_relevance(question: str, chunk: str) -> float:
    query_terms = set(_retrieval_terms(question))
    chunk_terms = set(_retrieval_terms(chunk))
    if not query_terms or not chunk_terms:
        return 0.0
    return len(query_terms & chunk_terms) / len(query_terms)


def _retrieval_terms(text: str) -> List[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]{2,}", str(text).lower())


def _text_rank(target: str, texts: List[str]) -> int:
    target_norm = _norm(target)
    for index, text in enumerate(texts, 1):
        text_norm = _norm(text)
        if target_norm in text_norm or text_norm in target_norm:
            return index
    return 0


def _exact_match(prediction: str, answer: str) -> bool:
    return _norm(prediction) == _norm(answer)


def _answer_f1(prediction: str, answer: str) -> float:
    """Token-level F1, reported independently from semantic QA accuracy."""
    predicted = _answer_tokens(prediction)
    expected = _answer_tokens(answer)
    if not predicted or not expected:
        return float(bool(predicted == expected))
    hits = sum((Counter(predicted) & Counter(expected)).values())
    if not hits:
        return 0.0
    precision, recall = hits / len(predicted), hits / len(expected)
    lexical_f1 = round(2 * precision * recall / (precision + recall), 6)
    return lexical_f1


def _build_embedding_model(cfg: MemoryExperimentConfig):
    """Construct the declared Ours embedder; never silently fall back."""
    backend = cfg.embedding_backend.lower().strip()
    if backend == "bge_m3":
        return BGEM3EmbeddingModel(batch_size=cfg.bge_batch_size)
    if backend == "hashing":
        return HashingEmbeddingModel()
    raise ValueError("embedding_backend must be 'bge_m3' or 'hashing'")


def _write_example_memories(
    store: HybridTieredMemoryStore,
    example: MemoryExample,
    context: MemoryContext,
    cfg: MemoryExperimentConfig,
) -> None:
    """Write one history, batching BGE embeddings when updates are disabled.

    A no-Judge ablation has no stateful update decision to preserve, so issuing
    one model call per session is needless and makes a normal LoCoMo-scale
    evaluation impractically slow.  The stored index text is identical to the
    regular write path.
    """
    texts = list(example.memories)
    if cfg.use_llm_judge or not hasattr(store.embedding_model, "embed_batch"):
        for index, memory in enumerate(texts):
            store.append(
                memory, MemoryScope.PROJECT, context=context,
                tags=[example.source or "memory", f"memory-{index}"],
            )
        return

    tags_by_item = [[example.source or "memory", f"memory-{index}"] for index in range(len(texts))]
    summaries = [_summary_for_index(memory) for memory in texts]
    index_texts = [f"{summary} {' '.join(tags)} {{}}" for summary, tags in zip(summaries, tags_by_item)]
    embeddings = store.embedding_model.embed_batch(index_texts)
    if len(embeddings) != len(texts):
        raise RuntimeError("BGE-M3 batch embedding count does not match input memories")
    for memory, tags, summary, embedding in zip(texts, tags_by_item, summaries, embeddings):
        store.write(
            MemoryItem(
                content=memory,
                scope=MemoryScope.PROJECT,
                scope_id=context.id_for(MemoryScope.PROJECT),
                tags=tags,
                summary=summary,
                embedding=embedding,
            ),
            context=context,
        )


def _summary_for_index(text: str, max_chars: int = 1200) -> str:
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    return normalized if len(normalized) <= max_chars else normalized[: max_chars - 3].rstrip() + "..."


def _with_chunked_memories(example: MemoryExample, chunk_chars: int) -> MemoryExample:
    """Coalesce utterance-level histories into auditable retrieval units."""
    if chunk_chars <= 0:
        return example
    chunks: List[str] = []
    current: List[str] = []
    current_chars = 0
    for memory in example.memories:
        addition = len(memory) + (1 if current else 0)
        if current and current_chars + addition > chunk_chars:
            chunks.append("\n".join(current))
            current, current_chars = [], 0
        current.append(memory)
        current_chars += addition
    if current:
        chunks.append("\n".join(current))
    return MemoryExample(
        id=example.id, question=example.question, answer=example.answer,
        memories=chunks, evidence=list(example.evidence),
        trajectory_id=example.trajectory_id, source=example.source,
        metadata={**example.metadata, "history_unit": "coalesced_text_chunk", "history_chunk_chars": chunk_chars},
    )


def _rerank_sessions_with_bge(question: str, sessions: List[str], model: BGEM3EmbeddingModel) -> List[str]:
    """BGE-M3 rerank of a fixed first-stage candidate set."""
    if not sessions:
        return []
    # Keep reranking tractable on CPU while preserving the most query-relevant
    # local evidence.  Full source chunks remain available after ranking.
    rerank_texts = [_query_focused_window(question, session) for session in sessions]
    vectors = model.embed_batch([question, *rerank_texts])
    query = vectors[0]

    def cosine(vector: List[float]) -> float:
        return sum(a * b for a, b in zip(query, vector))

    return [
        session
        for _, _, session in sorted(
            ((cosine(vector), index, session) for index, (session, vector) in enumerate(zip(sessions, vectors[1:]))),
            key=lambda item: (-item[0], item[1]),
        )
    ]


def _query_focused_window(question: str, text: str, max_chars: int = 512) -> str:
    if len(text) <= max_chars:
        return text
    terms = [term for term in _retrieval_terms(question) if len(term) > 1]
    lowered = text.lower()
    positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
    if not positions:
        return text[:max_chars]
    center = min(positions)
    start = max(0, center - max_chars // 3)
    end = min(len(text), start + max_chars)
    return text[start:end]


def _answer_tokens(text: str) -> List[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", str(text).lower())


def _mem0_config(root: Path, settings: Any) -> Dict[str, Any]:
    return {
        "llm": {"provider": "deepseek", "config": {"api_key": settings.api_key, "model": os.environ.get("OPENAI_JUDGE_MODEL") or settings.model, "deepseek_base_url": settings.base_url, "temperature": 0.0}},
        "embedder": {"provider": "huggingface", "config": {"model": resolve_bge_m3_model_path("BAAI/bge-m3"), "embedding_dims": 1024, "model_kwargs": {"cache_folder": resolve_bge_m3_cache_dir(), "local_files_only": True}}},
        "vector_store": {"provider": "qdrant", "config": {"collection_name": "memory_experiment", "path": str(root / "qdrant"), "embedding_model_dims": 1024, "on_disk": True}},
        "history_db_path": str(root / "history.db"),
    }


def _disable_mem0_thinking(memory: Any) -> None:
    """Force non-thinking DeepSeek calls for Mem0's strict JSON extraction.

    mem0ai 2.0.19 filters provider kwargs and its DeepSeek config has no
    ``extra_body`` field, so setting this in ``_mem0_config`` is ineffective.
    Wrap only this Memory instance's OpenAI-compatible request method instead
    of modifying the installed third-party package.
    """
    try:
        completions = memory.llm.client.chat.completions
        original_create = completions.create
    except AttributeError as exc:
        raise RuntimeError("cannot configure Mem0 DeepSeek thinking mode") from exc

    def create_non_thinking(*args: Any, **kwargs: Any) -> Any:
        extra_body = dict(kwargs.get("extra_body") or {})
        thinking = dict(extra_body.get("thinking") or {})
        thinking["type"] = "disabled"
        extra_body["thinking"] = thinking
        kwargs["extra_body"] = extra_body
        return original_create(*args, **kwargs)

    completions.create = create_non_thinking


def _mem0_results(result: Any) -> List[Dict[str, Any]]:
    raw = (result.get("results") or result.get("memories") or []) if isinstance(result, dict) else (result or [])
    return [item for item in raw if isinstance(item, dict)]


def _mem0_memory_text(item: Dict[str, Any]) -> str:
    return str(item.get("memory") or item.get("text") or item.get("content") or item.get("data") or "")


def _mem0_source_text(item: Dict[str, Any], example: MemoryExample) -> str:
    """Map an extracted Mem0 fact back to its source session for retrieval metrics."""
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    try:
        index = int(metadata.get("index"))
    except (TypeError, ValueError):
        return _mem0_memory_text(item)
    if 0 <= index < len(example.memories):
        return example.memories[index]
    return _mem0_memory_text(item)


def _deduplicated_mem0_source_sessions(
    result_items: List[Dict[str, Any]], example: MemoryExample,
) -> List[str]:
    """Evaluate Mem0 at source-session grain, matching the Ours unit."""
    unique: List[str] = []
    seen: set[str] = set()
    for item in result_items:
        source_text = _mem0_source_text(item, example)
        key = _norm(source_text)
        if key and key not in seen:
            unique.append(source_text)
            seen.add(key)
    return unique


def _mem0_user_id(example: MemoryExample) -> str:
    return "exp-" + "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in example.id)


def _engine_memory_context(example: MemoryExample) -> MemoryContext:
    """Isolate each benchmark question's history in its own project scope."""
    source = example.source or "memory-exp"
    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in example.id)
    return MemoryContext(
        task_id=example.id,
        project_id=f"{source}-{safe_id}",
        global_id="memory-exp",
    )


def _engine_retrieved_text(store: HybridTieredMemoryStore, item: Any) -> str:
    """Hydrate archived Top-K items before QA and evidence evaluation."""
    return store.expand(item) if item.raw_ref else str(item.content)


def _norm(text: str) -> str:
    return "".join(str(text).lower().split())


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0
