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
    HybridTieredMemoryStore,
    MemoryContext,
    MemoryScope,
    RetrievalMode,
    build_default_memory_judge,
)

from .reports import ExperimentReport
from .types import ExperimentRow, MemoryExample


QA_SYSTEM_PROMPT = """You answer a question using only the supplied memory context.
If the context does not contain enough information, reply exactly: INSUFFICIENT_INFORMATION.
Do not use outside knowledge. Return only the shortest answer, with no explanation."""


@dataclass
class MemoryExperimentConfig:
    backend: str = "engine"
    top_k: int = 5
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
    qa_timeout_seconds: float = 90.0


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
    if cfg.backend == "mem0":
        return _run_mem0_memory(examples, cfg)
    raise ValueError(f"unsupported memory backend: {cfg.backend}")


def _run_engine_memory(examples: List[MemoryExample], cfg: MemoryExperimentConfig) -> ExperimentReport:
    root = Path(cfg.output_root) / "engine_store"
    if cfg.clean and root.exists():
        shutil.rmtree(root)
    judge = build_default_memory_judge() if cfg.use_llm_judge else None
    store = HybridTieredMemoryStore(
        root,
        memory_llm_judge=judge,
        enable_memory_update=cfg.enable_memory_update,
        long_text_threshold=cfg.long_text_threshold,
    )
    rows: List[ExperimentRow] = []
    started = time.perf_counter()
    try:
        for example in examples:
            context = MemoryContext(task_id=example.id, project_id=example.source or "memory-exp", global_id="memory-exp")
            write_started = time.perf_counter()
            for index, memory in enumerate(example.memories):
                store.append(memory, MemoryScope.PROJECT, context=context, tags=[example.source or "memory", f"memory-{index}"])
            write_seconds = time.perf_counter() - write_started
            retrieval_started = time.perf_counter()
            mode = RetrievalMode(cfg.retrieval_mode)
            if cfg.cascade_read:
                retrieved = store.cascade_read(example.question, context=context, top_k=cfg.top_k, retrieval_mode=mode)
            else:
                retrieved = store.read(example.question, scope=MemoryScope.PROJECT, context=context, top_k=cfg.top_k, retrieval_mode=mode)
            retrieval_seconds = time.perf_counter() - retrieval_started
            rows.append(_evaluate_example(
                example, [str(item.content) for item in retrieved], cfg,
                write_seconds=write_seconds, retrieval_seconds=retrieval_seconds,
                retrieval_applicable=True,
                metadata={"backend": cfg.backend, "source": example.source},
            ))
    finally:
        action_counts = Counter(store.memory_update_action_counts)
        store.close()
    return ExperimentReport(
        name=f"memory-{cfg.backend}", rows=rows,
        metadata={
            "backend": cfg.backend, "top_k": cfg.top_k, "scope": MemoryScope.PROJECT.value,
            "llm_judge_enabled": judge is not None, "retrieval_mode": cfg.retrieval_mode,
            "cascade_read": cfg.cascade_read, "enable_memory_update": cfg.enable_memory_update,
            "qa_solver": cfg.qa_solver, "qa_model": cfg.qa_model or "OPENAI_MODEL",
            "storage_bytes": _directory_size(root), "memory_update_actions": dict(action_counts),
            "seconds": round(time.perf_counter() - started, 4),
        },
    )


def _run_context_baseline(
    examples: List[MemoryExample], cfg: MemoryExperimentConfig, *, full_context: bool
) -> ExperimentReport:
    rows: List[ExperimentRow] = []
    started = time.perf_counter()
    for example in examples:
        retrieval_started = time.perf_counter()
        visible = list(example.memories) if full_context else []
        retrieval_seconds = time.perf_counter() - retrieval_started
        rows.append(_evaluate_example(
            example, visible, cfg, write_seconds=0.0, retrieval_seconds=retrieval_seconds,
            retrieval_applicable=False,
            metadata={"backend": cfg.backend, "source": example.source, "retrieval_metrics": "not_applicable_for_context_control"},
        ))
    return ExperimentReport(
        name=f"memory-{cfg.backend}", rows=rows,
        metadata={
            "backend": cfg.backend, "top_k": cfg.top_k, "qa_solver": cfg.qa_solver,
            "qa_model": cfg.qa_model or "OPENAI_MODEL", "storage_bytes": 0,
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
    rows: List[ExperimentRow] = []
    started = time.perf_counter()
    for example in examples:
        user_id = _mem0_user_id(example)
        write_started = time.perf_counter()
        for index, text in enumerate(example.memories):
            memory.add([{"role": "user", "content": text}], user_id=user_id,
                       metadata={"source": example.source, "example_id": example.id, "index": index}, infer=cfg.mem0_infer)
        write_seconds = time.perf_counter() - write_started
        retrieval_started = time.perf_counter()
        result = memory.search(example.question, top_k=cfg.top_k, filters={"user_id": user_id}, threshold=cfg.mem0_threshold)
        retrieval_seconds = time.perf_counter() - retrieval_started
        rows.append(_evaluate_example(
            example, [_mem0_memory_text(item) for item in _mem0_results(result)], cfg,
            write_seconds=write_seconds, retrieval_seconds=retrieval_seconds, retrieval_applicable=True,
            metadata={"backend": "mem0", "source": example.source},
        ))
    return ExperimentReport(
        name="memory-mem0", rows=rows,
        metadata={
            "backend": "mem0", "top_k": cfg.top_k, "infer": cfg.mem0_infer,
            "threshold": cfg.mem0_threshold, "qa_solver": cfg.qa_solver,
            "qa_model": cfg.qa_model or "OPENAI_MODEL", "storage_bytes": _directory_size(root),
            "seconds": round(time.perf_counter() - started, 4),
        },
    )


def _evaluate_example(
    example: MemoryExample, retrieved: List[str], cfg: MemoryExperimentConfig, *,
    write_seconds: float, retrieval_seconds: float, retrieval_applicable: bool, metadata: Dict[str, Any],
) -> ExperimentRow:
    context = "\n\n".join(retrieved)
    answer_started = time.perf_counter()
    prediction, qa_usage, qa_error = _answer_question(example.question, context, cfg)
    qa_seconds = time.perf_counter() - answer_started
    qa_exact = _exact_match(prediction, example.answer)
    answer_f1 = _answer_f1(prediction, example.answer)
    metrics: Dict[str, Any] = {
        "qa_acc": int(qa_exact), "answer_f1": answer_f1, "retrieved": len(retrieved),
        "write_ms": write_seconds * 1000, "retrieval_ms": retrieval_seconds * 1000,
        "qa_ms": qa_seconds * 1000, "time_ms": (write_seconds + retrieval_seconds + qa_seconds) * 1000,
        "context_tokens": rough_token_count(context),
        "qa_input_tokens": int(qa_usage.get("prompt_tokens") or rough_token_count(_qa_prompt(example.question, context))),
        "qa_output_tokens": int(qa_usage.get("completion_tokens") or rough_token_count(prediction)),
        "tokens": int(qa_usage.get("total_tokens") or rough_token_count(_qa_prompt(example.question, context)) + rough_token_count(prediction)),
    }
    if retrieval_applicable:
        rank, evidence_recall, mode = _evidence_metrics(example, retrieved)
        metrics.update({
            "rank": rank, "mrr": 1.0 / rank if rank else 0.0,
            "hit_at_1": int(rank == 1), "hit_at_3": int(0 < rank <= 3),
            "hit_at_5": int(0 < rank <= 5), "evidence_recall": evidence_recall,
        })
        metadata = {**metadata, "evidence_metric": mode}
    if qa_error:
        metadata = {**metadata, "qa_error": qa_error}
    return ExperimentRow(
        id=example.id, passed=qa_exact, score=answer_f1, prediction=prediction, expected=example.answer,
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
        )
        usage = getattr(response, "usage", None)
        usage_data = json.loads(usage.model_dump_json()) if usage is not None else {}
        return (response.choices[0].message.content or "").strip(), usage_data, ""
    except Exception as exc:
        return "", {}, f"{type(exc).__name__}: {exc}"


def _qa_prompt(question: str, context: str) -> str:
    return f"Memory context:\n{context or '[empty]'}\n\nQuestion: {question}\nAnswer:"


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
    predicted = _answer_tokens(prediction)
    expected = _answer_tokens(answer)
    if not predicted or not expected:
        return float(bool(predicted == expected))
    hits = sum((Counter(predicted) & Counter(expected)).values())
    if not hits:
        return 0.0
    precision, recall = hits / len(predicted), hits / len(expected)
    return round(2 * precision * recall / (precision + recall), 6)


def _answer_tokens(text: str) -> List[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", str(text).lower())


def _mem0_config(root: Path, settings: Any) -> Dict[str, Any]:
    return {
        "llm": {"provider": "deepseek", "config": {"api_key": settings.api_key, "model": os.environ.get("OPENAI_JUDGE_MODEL") or settings.model, "deepseek_base_url": settings.base_url, "temperature": 0.0}},
        "embedder": {"provider": "huggingface", "config": {"model": resolve_bge_m3_model_path("BAAI/bge-m3"), "embedding_dims": 1024, "model_kwargs": {"cache_folder": resolve_bge_m3_cache_dir(), "local_files_only": True}}},
        "vector_store": {"provider": "qdrant", "config": {"collection_name": "memory_experiment", "path": str(root / "qdrant"), "embedding_model_dims": 1024, "on_disk": True}},
        "history_db_path": str(root / "history.db"),
    }


def _mem0_results(result: Any) -> List[Dict[str, Any]]:
    raw = (result.get("results") or result.get("memories") or []) if isinstance(result, dict) else (result or [])
    return [item for item in raw if isinstance(item, dict)]


def _mem0_memory_text(item: Dict[str, Any]) -> str:
    return str(item.get("memory") or item.get("text") or item.get("content") or item.get("data") or "")


def _mem0_user_id(example: MemoryExample) -> str:
    return "exp-" + "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in example.id)


def _norm(text: str) -> str:
    return "".join(str(text).lower().split())


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0
