"""Run reproducible low-entropy communication and temporal-memory ablations."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.experiments import run_communication_ablation, run_temporal_memory_ablation, save_report


def main() -> None:
    root=Path("runs/experiments/communication_memory")
    for mode in ("free_text_append","structured_no_gate","structured_gate_compress"):
        save_report(run_communication_ablation(mode),root/mode)
    save_report(run_temporal_memory_ablation(temporal=False),root/"tiered_memory")
    save_report(run_temporal_memory_ablation(temporal=True),root/"temporal_evidence")
    print(root.resolve())


if __name__ == "__main__": main()
