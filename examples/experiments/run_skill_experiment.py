"""运行 SkillEvolBench 小子集技能实验。

示例：
    python examples/experiments/run_skill_experiment.py --dataset data/skillevolbench_sample.jsonl --limit 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.experiments import SkillExperimentConfig, load_skill_dataset, run_skill_experiment, save_report
from examples.experiments.resume import filter_remaining, load_existing_rows, merge_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output-dir", default="runs/experiments/skills")
    parser.add_argument("--fresh", action="store_true", help="忽略已有 rows.jsonl，完整重跑")
    args = parser.parse_args()

    examples = load_skill_dataset(args.dataset, limit=args.limit)
    existing = load_existing_rows(args.output_dir, fresh=args.fresh)
    remaining = filter_remaining(examples, existing)
    if not remaining:
        print({"status": "already_complete", "rows": len(existing), "output_dir": args.output_dir})
        return
    report = run_skill_experiment(
        remaining,
        SkillExperimentConfig(output_root=args.output_dir, clean=args.fresh or not existing),
    )
    report = merge_reports(report.name, existing, report)
    paths = save_report(report, args.output_dir)
    print(paths)


if __name__ == "__main__":
    main()
