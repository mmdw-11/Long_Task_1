"""Run the minimum formal memory ablation matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.experiments import MemoryExperimentConfig, load_memory_dataset, run_memory_experiment, save_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", choices=["longmemeval", "locomo", "custom"], default="longmemeval")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", default="runs/experiments/memory_ablation")
    args = parser.parse_args()
    examples = load_memory_dataset(args.dataset, source=args.source, limit=args.limit)

    matrix = []
    for top_k in (1, 3, 5, 10):
        matrix.append((f"top_k_{top_k}", dict(top_k=top_k, use_llm_judge=False)))
    for mode in ("dense", "sparse", "hybrid", "hybrid_temporal"):
        matrix.append((f"retrieval_{mode}", dict(retrieval_mode=mode, use_llm_judge=False)))
    matrix.extend(
        [
            ("judge_off", dict(use_llm_judge=False)),
            ("judge_on", dict(use_llm_judge=True)),
            ("project_only", dict(cascade_read=False, use_llm_judge=False)),
            ("cascade_read", dict(cascade_read=True, use_llm_judge=False)),
            ("append_only", dict(enable_memory_update=False, use_llm_judge=False)),
            ("update_merge", dict(enable_memory_update=True, use_llm_judge=True)),
            ("cold_archive_off", dict(long_text_threshold=10**9, use_llm_judge=False)),
            ("cold_archive_on", dict(long_text_threshold=1, use_llm_judge=False)),
        ]
    )
    for name, overrides in matrix:
        report = run_memory_experiment(
            examples,
            MemoryExperimentConfig(
                output_root=str(Path(args.output_dir) / name),
                clean=True,
                **overrides,
            ),
        )
        print(name, save_report(report, Path(args.output_dir) / name))


if __name__ == "__main__":
    main()
