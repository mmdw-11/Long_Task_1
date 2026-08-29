"""Prepare the fixed inputs for the four retained experiments.

Raw public datasets are never modified.  This script writes normalized JSONL
files under data/processed so every method receives exactly the same records.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.experiments import (
    build_long_task_dataset_from_memory,
    build_skill_reuse_dataset,
    load_memory_dataset,
    memory_examples_to_rows,
    sample_memory_examples,
    save_jsonl,
    long_task_examples_to_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and sample experiment datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    memory = subparsers.add_parser("memory", help="prepare LongMemEval or LoCoMo QA records")
    memory.add_argument("--input", required=True)
    memory.add_argument("--source", choices=["longmemeval", "locomo"], required=True)
    memory.add_argument("--output", required=True)
    memory.add_argument("--count", type=int, required=True)
    memory.add_argument("--seed", type=int, default=42)

    state = subparsers.add_parser("state", help="derive state-governance tasks from normalized LoCoMo")
    state.add_argument("--input", required=True)
    state.add_argument("--output", required=True)
    state.add_argument("--count", type=int, default=100)
    state.add_argument("--seed", type=int, default=42)

    skills = subparsers.add_parser("skills", help="create the balanced 30-item repeated-task dataset")
    skills.add_argument("--output", required=True)
    skills.add_argument("--count", type=int, default=30)
    skills.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "memory":
        examples = sample_memory_examples(
            load_memory_dataset(args.input, source=args.source), count=args.count, seed=args.seed
        )
        path = save_jsonl(args.output, memory_examples_to_rows(examples))
        print({"output": str(path), "questions": len(examples), "trajectories": len({item.trajectory_id or item.id for item in examples})})
    elif args.command == "state":
        source = load_memory_dataset(args.input, source="locomo")
        examples = build_long_task_dataset_from_memory(source, count=args.count, seed=args.seed)
        path = save_jsonl(args.output, long_task_examples_to_rows(examples))
        print({"output": str(path), "tasks": len(examples)})
    else:
        examples = build_skill_reuse_dataset(size=args.count, seed=args.seed)
        path = save_jsonl(args.output, [
            {"id": item.id, "task": item.task, "trajectory": item.trajectory,
             "expected_steps": item.expected_steps, "task_type": item.task_type,
             "source": item.source, "metadata": item.metadata}
            for item in examples
        ])
        print({"output": str(path), "tasks": len(examples)})


if __name__ == "__main__":
    main()
