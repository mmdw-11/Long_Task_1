"""Build offline difficulty strata for the frozen τ test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.experiments.tau_analysis import summarize_profiles, task_profile
from engine.experiments.tau_data import load_tau_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/tau_skill_v1/tasks.jsonl")
    parser.add_argument("--output", default="runs/experiments/tau_skill_v1/offline/task_profiles.json")
    args = parser.parse_args()
    profiles = [task_profile(task) for task in load_tau_tasks(args.dataset, split="test")]
    payload = {"summary": summarize_profiles(profiles), "profiles": profiles}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print({"output": str(output), **payload["summary"]})


if __name__ == "__main__":
    main()

