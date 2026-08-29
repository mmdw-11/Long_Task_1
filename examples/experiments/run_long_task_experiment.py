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
    parser.add_argument("--size", type=int, default=100, help="仅未传 --dataset 时使用")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=[*LONG_TASK_METHODS, "all"], default="all")
    parser.add_argument("--output-dir", default="runs/experiments/long_task")
    parser.add_argument("--max-context-tokens", type=int, default=320)
    args = parser.parse_args()

    examples = load_long_task_dataset(args.dataset) if args.dataset else build_long_task_dataset(size=args.size, seed=args.seed)
    methods = list(LONG_TASK_METHODS) if args.method == "all" else [args.method]
    for method in methods:
        report = run_long_task_experiment(
            examples,
            LongTaskExperimentConfig(
                method=method,
                output_root=args.output_dir,
                max_context_tokens=args.max_context_tokens,
            ),
        )
        paths = save_report(report, Path(args.output_dir) / method)
        print(method, paths)


if __name__ == "__main__":
    main()
