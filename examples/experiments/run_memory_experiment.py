"""运行长期记忆实验。

示例：
    python examples/experiments/run_memory_experiment.py --dataset data/longmemeval.jsonl --source longmemeval
    python examples/experiments/run_memory_experiment.py --dataset data/locomo.json --source locomo --limit 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.experiments import MemoryExperimentConfig, load_memory_dataset, run_memory_experiment, save_report
from examples.experiments.resume import filter_remaining, load_existing_rows, merge_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", choices=["longmemeval", "locomo", "custom"], default="longmemeval")
    parser.add_argument(
        "--backend",
        choices=["no_memory", "full_context", "full_context_budgeted", "engine", "mem0"],
        default="engine",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20, help="Ours 检索的候选会话数")
    parser.add_argument("--context-token-budget", type=int, default=12000, help="Ours 送入 QA 的证据上下文预算")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", default="runs/experiments/memory")
    parser.add_argument("--run-name", default="", help="结果子目录；不同方法必须使用不同名称")
    parser.add_argument("--mem0-raw", action="store_true", help="mem0 直接写原文，不让 LLM 抽取记忆")
    parser.add_argument("--no-llm-judge", action="store_true", help="engine 后端关闭 DeepSeek 记忆更新判断")
    parser.add_argument(
        "--retrieval-mode",
        choices=["dense", "sparse", "hybrid", "hybrid_temporal"],
        default="hybrid",
    )
    parser.add_argument("--project-only", action="store_true", help="仅检索 PROJECT 层，不做层级级联")
    parser.add_argument("--append-only", action="store_true", help="关闭 UPDATE/DELETE 合并策略")
    parser.add_argument("--qa-solver", choices=["llm", "extractive"], default="llm")
    parser.add_argument("--qa-model", default="", help="默认使用 .env 中的 OPENAI_MODEL")
    parser.add_argument("--fresh", action="store_true", help="忽略已有 rows.jsonl，完整重跑")
    args = parser.parse_args()

    examples = load_memory_dataset(args.dataset, source=args.source, limit=args.limit)
    default_name = args.backend
    if args.backend == "engine" and args.no_llm_judge:
        default_name = "ours_no_judge"
    final_dir = Path(args.output_dir) / args.source / (args.run_name or default_name)
    existing = load_existing_rows(final_dir, fresh=args.fresh)
    remaining = filter_remaining(examples, existing)
    if not remaining:
        print({"status": "already_complete", "rows": len(existing), "output_dir": str(final_dir)})
        return
    report = run_memory_experiment(
        remaining,
        MemoryExperimentConfig(
            backend=args.backend,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            context_token_budget=args.context_token_budget,
            output_root=str(final_dir),
            clean=args.fresh or not existing,
            use_llm_judge=not args.no_llm_judge,
            mem0_infer=not args.mem0_raw,
            retrieval_mode=args.retrieval_mode,
            cascade_read=not args.project_only,
            enable_memory_update=not args.append_only,
            qa_solver=args.qa_solver,
            qa_model=args.qa_model,
        ),
    )
    report = merge_reports(report.name, existing, report)
    paths = save_report(report, final_dir)
    print(paths)


if __name__ == "__main__":
    main()
