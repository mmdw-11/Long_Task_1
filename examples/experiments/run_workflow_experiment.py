"""生成并运行 300 条小型 workflow 编排实验。

示例：
    python examples/experiments/run_workflow_experiment.py --size 300
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.experiments import build_workflow_dataset, run_workflow_experiment, save_report


async def _amain() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-out", default="runs/experiments/workflow/workflow_dataset.jsonl")
    parser.add_argument("--output-dir", default="runs/experiments/workflow")
    args = parser.parse_args()

    examples = build_workflow_dataset(size=args.size, seed=args.seed, output=args.dataset_out)
    report = await run_workflow_experiment(examples)
    paths = save_report(report, args.output_dir)
    print(paths)


if __name__ == "__main__":
    asyncio.run(_amain())
