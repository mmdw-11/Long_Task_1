"""长期记忆实验 runner。

本实验用于 LongMemEval 主实验和 LoCoMo 补充实验。流程是：把样本中的历史记忆写入
指定后端，再用问题检索 top-k 记忆，检查标准答案是否能被检索出来。

当前实现两个真实后端：
- engine：本项目 HybridTieredMemoryStore，默认启用 DeepSeek 记忆更新判断。
- mem0：mem0 OSS，本地 Qdrant + 本地 BGE-M3/HuggingFace embedding + DeepSeek LLM。
"""

from __future__ import annotations

import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from engine.config import load_settings
from engine.modules.bge_local import resolve_bge_m3_cache_dir, resolve_bge_m3_model_path
from engine.modules.memory import (
    HybridTieredMemoryStore,
    MemoryContext,
    MemoryScope,
    RetrievalMode,
    build_default_memory_judge,
)
from engine.modules.context.budget import rough_token_count

from .reports import ExperimentReport
from .types import ExperimentRow, MemoryExample


@dataclass
class MemoryExperimentConfig:
    """长期记忆实验配置。"""

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


def run_memory_experiment(
    examples: List[MemoryExample],
    config: MemoryExperimentConfig | None = None,
) -> ExperimentReport:
    """运行长期记忆检索实验。"""
    cfg = config or MemoryExperimentConfig()
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
    started = time.time()
    for example in examples:
        ctx = MemoryContext(task_id=example.id, project_id=example.source or "memory-exp", global_id="memory-exp")
        # 长期记忆实验写 PROJECT 层，才能覆盖跨 run 的持久记忆与 LLM 更新判断。
        write_started = time.perf_counter()
        for index, memory in enumerate(example.memories):
            store.append(
                memory,
                MemoryScope.PROJECT,
                context=ctx,
                tags=[example.source or "memory", f"memory-{index}"],
            )
        write_seconds = time.perf_counter() - write_started
        retrieval_started = time.perf_counter()
        mode = RetrievalMode(cfg.retrieval_mode)
        if cfg.cascade_read:
            retrieved = store.cascade_read(
                example.question,
                context=ctx,
                top_k=cfg.top_k,
                retrieval_mode=mode,
            )
        else:
            retrieved = store.read(
                example.question,
                scope=MemoryScope.PROJECT,
                context=ctx,
                top_k=cfg.top_k,
                retrieval_mode=mode,
            )
        retrieval_seconds = time.perf_counter() - retrieval_started
        retrieved_text = "\n".join(str(item.content) for item in retrieved)
        rank = _answer_rank(example.answer, [str(item.content) for item in retrieved])
        passed = rank > 0
        rows.append(
            ExperimentRow(
                id=example.id,
                passed=passed,
                score=1.0 / rank if rank else 0.0,
                prediction=retrieved_text,
                expected=example.answer,
                metrics={
                    "hit": 1 if passed else 0,
                    "rank": rank or 0,
                    "mrr": 1.0 / rank if rank else 0.0,
                    "retrieved": len(retrieved),
                    "hit_at_1": 1 if rank == 1 else 0,
                    "hit_at_3": 1 if 0 < rank <= 3 else 0,
                    "hit_at_5": 1 if 0 < rank <= 5 else 0,
                    "recall_at_k": 1 if rank else 0,
                    "write_ms": write_seconds * 1000,
                    "retrieval_ms": retrieval_seconds * 1000,
                    "tokens": rough_token_count(retrieved_text),
                },
                metadata={"source": example.source, "backend": cfg.backend},
            )
        )
    judge_enabled = store.memory_llm_judge is not None
    action_counts = Counter(store.memory_update_action_counts)
    store.close()
    return ExperimentReport(
        name=f"memory-{cfg.backend}",
        rows=rows,
        metadata={
            "backend": cfg.backend,
            "top_k": cfg.top_k,
            "scope": MemoryScope.PROJECT.value,
            "llm_judge_enabled": judge_enabled,
            "retrieval_mode": cfg.retrieval_mode,
            "cascade_read": cfg.cascade_read,
            "enable_memory_update": cfg.enable_memory_update,
            "storage_bytes": _directory_size(root),
            "memory_update_actions": dict(action_counts),
            "seconds": round(time.time() - started, 4),
        },
    )


def _run_context_baseline(
    examples: List[MemoryExample],
    cfg: MemoryExperimentConfig,
    *,
    full_context: bool,
) -> ExperimentReport:
    """Run model-free controls for retrieval cost and answer containment.

    ``full_context`` exposes every history item to the evaluator; ``no_memory``
    exposes none.  They intentionally do not claim generative QA quality.
    """
    rows: List[ExperimentRow] = []
    started = time.perf_counter()
    for example in examples:
        retrieval_started = time.perf_counter()
        retrieved = list(example.memories) if full_context else []
        retrieval_seconds = time.perf_counter() - retrieval_started
        rank = _answer_rank(example.answer, retrieved)
        text = "\n".join(retrieved)
        rows.append(
            ExperimentRow(
                id=example.id,
                passed=rank > 0,
                score=1.0 / rank if rank else 0.0,
                prediction=text,
                expected=example.answer,
                metrics={
                    "hit": 1 if rank else 0,
                    "rank": rank or 0,
                    "mrr": 1.0 / rank if rank else 0.0,
                    "retrieved": len(retrieved),
                    "hit_at_1": 1 if rank == 1 else 0,
                    "hit_at_3": 1 if 0 < rank <= 3 else 0,
                    "hit_at_5": 1 if 0 < rank <= 5 else 0,
                    "recall_at_k": 1 if rank else 0,
                    "write_ms": 0.0,
                    "retrieval_ms": retrieval_seconds * 1000,
                    "tokens": rough_token_count(text),
                },
                metadata={"source": example.source, "backend": cfg.backend},
            )
        )
    return ExperimentReport(
        name=f"memory-{cfg.backend}",
        rows=rows,
        metadata={
            "backend": cfg.backend,
            "top_k": cfg.top_k,
            "evaluator_scope": "answer containment, not generative QA",
            "storage_bytes": 0,
            "seconds": round(time.perf_counter() - started, 4),
        },
    )


def _run_mem0_memory(examples: List[MemoryExample], cfg: MemoryExperimentConfig) -> ExperimentReport:
    """运行真实 mem0 对照实验。

    mem0 需要三块配置：LLM、embedding、vector store。这里全部在本地实验目录内落盘，
    LLM 复用项目云端 DeepSeek，embedding 复用本地 BGE-M3，向量库存到本地 Qdrant。
    """
    root = Path(cfg.output_root) / "mem0_store"
    if cfg.clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    os.environ["MEM0_DIR"] = str(root / "home")
    settings = load_settings()

    from mem0 import Memory

    memory = Memory.from_config(_mem0_config(root, settings))
    rows: List[ExperimentRow] = []
    started = time.time()
    for example in examples:
        user_id = _mem0_user_id(example)
        for index, text in enumerate(example.memories):
            memory.add(
                [{"role": "user", "content": text}],
                user_id=user_id,
                metadata={"source": example.source, "example_id": example.id, "index": index},
                infer=cfg.mem0_infer,
            )
        result = memory.search(
            example.question,
            top_k=cfg.top_k,
            filters={"user_id": user_id},
            threshold=cfg.mem0_threshold,
        )
        items = _mem0_results(result)
        retrieved_texts = [_mem0_memory_text(item) for item in items]
        rank = _answer_rank(example.answer, retrieved_texts)
        rows.append(
            ExperimentRow(
                id=example.id,
                passed=rank > 0,
                score=1.0 / rank if rank else 0.0,
                prediction="\n".join(retrieved_texts),
                expected=example.answer,
                metrics={
                    "hit": 1 if rank else 0,
                    "rank": rank or 0,
                    "mrr": 1.0 / rank if rank else 0.0,
                    "retrieved": len(items),
                },
                metadata={"source": example.source, "backend": cfg.backend},
            )
        )
    return ExperimentReport(
        name="memory-mem0",
        rows=rows,
        metadata={
            "backend": "mem0",
            "top_k": cfg.top_k,
            "infer": cfg.mem0_infer,
            "threshold": cfg.mem0_threshold,
            "seconds": round(time.time() - started, 4),
        },
    )


def _mem0_config(root: Path, settings: Any) -> Dict[str, Any]:
    model_path = resolve_bge_m3_model_path("BAAI/bge-m3")
    cache_dir = resolve_bge_m3_cache_dir()
    return {
        "llm": {
            "provider": "deepseek",
            "config": {
                "api_key": settings.api_key,
                "model": os.environ.get("OPENAI_JUDGE_MODEL") or settings.model,
                "deepseek_base_url": settings.base_url,
                "temperature": 0.0,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": model_path,
                "embedding_dims": 1024,
                "model_kwargs": {
                    "cache_folder": cache_dir,
                    "local_files_only": True,
                },
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "memory_experiment",
                "path": str(root / "qdrant"),
                "embedding_model_dims": 1024,
                "on_disk": True,
            },
        },
        "history_db_path": str(root / "history.db"),
    }


def _mem0_results(result: Any) -> List[Dict[str, Any]]:
    if isinstance(result, dict):
        raw = result.get("results") or result.get("memories") or []
    else:
        raw = result or []
    return [item for item in raw if isinstance(item, dict)]


def _mem0_memory_text(item: Dict[str, Any]) -> str:
    return str(item.get("memory") or item.get("text") or item.get("content") or item.get("data") or "")


def _mem0_user_id(example: MemoryExample) -> str:
    return "exp-" + "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in example.id)


def _answer_rank(answer: str, retrieved: List[str]) -> int:
    answer_norm = _norm(answer)
    if not answer_norm:
        return 0
    for index, text in enumerate(retrieved, 1):
        if answer_norm in _norm(text):
            return index
    return 0


def _norm(text: str) -> str:
    return "".join(str(text).lower().split())


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _memory_action_counts(path: Path) -> Counter:
    """Best-effort action accounting from the append-only memory audit."""
    counts: Counter = Counter()
    if not path.exists():
        return counts
    import json

    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except (TypeError, ValueError):
            continue
        action = payload.get("action") or (payload.get("metadata") or {}).get("memory_update_action")
        if action:
            counts[str(action).upper()] += 1
    return counts
