"""Generate and validate the frozen 40-case dynamic-topology task suite."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.experiments.dynamic_topology import build_dynamic_topology_dataset, dataset_manifest
from engine.experiments.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="examples/experiments/sample_data/dynamic_topology_tasks.jsonl")
    parser.add_argument("--manifest", default="examples/experiments/sample_data/dynamic_topology_tasks.manifest.json")
    args = parser.parse_args()
    rows = build_dynamic_topology_dataset(args.output)
    manifest = dataset_manifest(rows)
    write_json(args.manifest, manifest)
    print(json.dumps({"dataset_path": args.output, "manifest_path": args.manifest, **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
