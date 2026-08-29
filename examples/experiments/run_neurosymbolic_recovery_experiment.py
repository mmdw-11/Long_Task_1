from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.experiments.neurosymbolic_recovery import run_neurosymbolic_recovery_experiment


if __name__ == "__main__":
    target = Path("artifacts/experiments/neurosymbolic_recovery")
    result = run_neurosymbolic_recovery_experiment(target, seed=42)
    print(target.resolve())
    print(result)
