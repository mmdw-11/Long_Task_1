"""Run the memory/context-governance long-task controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.experiments import (
    LONG_TASK_METHODS,
    LongTaskExperimentConfig,
    build_long_task_dataset,
    load_long_task_dataset,
    run_long_task_experiment,
    save_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="", help="prepare_datasets.py 生成的状态保持 JSONL")
    parser.add_argument("--limit", type=int, default=0, help="仅运行前 N 条；0 表示全部")
    parser.add_argument("--size", type=int, default=100, help="仅未传 --dataset 时使用")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=[*LONG_TASK_METHODS, "all"], default="all")
    parser.add_argument("--output-dir", default="runs/experiments/long_task")
    parser.add_argument("--max-context-tokens", type=int, default=16384)
    parser.add_argument("--reserved-output-tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--evidence-token-budget", type=int, default=12000)
    parser.add_argument("--qa-solver", choices=["llm", "extractive"], default="llm")
    parser.add_argument("--qa-model", default="")
    parser.add_argument("--qa-judge-model", default="")
    parser.add_argument("--trajectory-judge", choices=["auto", "llm", "heuristic"], default="auto")
    parser.add_argument("--goal-similarity-threshold", type=float, default=0.8)
    parser.add_argument("--recovery-window", type=int, default=3)
    parser.add_argument("--embedding-backend", choices=["hashing", "bge_m3"], default="bge_m3")
    parser.add_argument("--bge-batch-size", type=int, default=32)
    args = parser.parse_args()

    examples = (
        load_long_task_dataset(args.dataset, limit=args.limit)
        if args.dataset
        else build_long_task_dataset(size=args.limit or args.size, seed=args.seed)
    )
    methods = list(LONG_TASK_METHODS) if args.method == "all" else [args.method]
    for method in methods:
        report = run_long_task_experiment(
            examples,
            LongTaskExperimentConfig(
                method=method,
                output_root=args.output_dir,
                max_context_tokens=args.max_context_tokens,
                reserved_output_tokens=args.reserved_output_tokens,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                evidence_token_budget=args.evidence_token_budget,
                qa_solver=args.qa_solver,
                qa_model=args.qa_model,
                qa_judge_model=args.qa_judge_model,
                trajectory_judge=args.trajectory_judge,
                goal_similarity_threshold=args.goal_similarity_threshold,
                recovery_window=args.recovery_window,
                embedding_backend=args.embedding_backend,
                bge_batch_size=args.bge_batch_size,
            ),
        )
        paths = save_report(report, Path(args.output_dir) / method)
        print(method, paths)


if __name__ == "__main__":
    main()
