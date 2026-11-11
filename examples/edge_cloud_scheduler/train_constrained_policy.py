"""Train a small constrained policy in a controlled full-action-feedback environment.

Input JSONL: {"limits": {...}, "steps": [{"profile": {...},
"estimates": {"edge": {"fee": .1}}, "outcomes": {"edge":
{"reward": 1., "fee": .1, "time_ms": 40., "success": true}}}]}
Outcomes must come from a controlled benchmark or be explicitly labelled as
synthetic. They are feedback, not inputs to the policy. Ordinary single-action
deployment logs cannot be treated as on-policy training episodes.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.modules.execution import InferenceResult
from engine.modules.scheduling import AdaptiveResourceScheduler, ConstrainedPlacementPolicy, ResourceRequest, PlacementInfeasible
from engine.modules.scheduling.constrained import BUDGET_KEY, TRAJECTORY_KEY
from engine.modules.scheduling.gate import HeuristicTaskGate


def train(episodes, output, *, epochs=1, seed=0):
    if not episodes or epochs < 1:
        raise ValueError("training requires episodes and at least one epoch")
    random.seed(seed)
    model = ConstrainedPlacementPolicy(seed=seed)
    reports = []
    for _ in range(epochs):
        for episode in episodes:
            if not episode.get("steps"):
                raise ValueError("empty training episode")
            state = {BUDGET_KEY: {"limits": dict(episode.get("limits") or {}), "spent": {}}}
            scheduler = AdaptiveResourceScheduler(gate=HeuristicTaskGate(), placement_policy=model, placement_mode="constrained")
            for index, step in enumerate(episode["steps"]):
                outcomes = step["outcomes"]
                request = ResourceRequest(node=f"train-{index}", state=state, metadata={
                    **step.get("profile", {}), "allowed_tiers": list(outcomes),
                    "placement_sample": True, "placement_estimates": step.get("estimates") or {},
                })
                try:
                    allocation = scheduler.acquire(request)
                except PlacementInfeasible:
                    if state.get(TRAJECTORY_KEY):
                        state[TRAJECTORY_KEY][-1]["reward"] += float(episode.get("infeasible_reward", -1.))
                    break
                outcome = outcomes[allocation.tier.value]
                result = InferenceResult("", "controlled-feedback", allocation.endpoint,
                    success=bool(outcome.get("success", True)),
                    metadata={"fee": outcome["fee"]} if "fee" in outcome else {})
                scheduler.record_outcome(request, allocation, result, elapsed_ms=outcome["time_ms"])
                state[TRAJECTORY_KEY][-1]["reward"] = float(outcome["reward"])
            trajectory = state.get(TRAJECTORY_KEY) or []
            if not trajectory:
                reports.append({"skipped": "no feasible sampled action"})
                continue
            trajectory[-1]["terminal"] = True
            reports.append(model.learn_episode(trajectory, state[BUDGET_KEY]["limits"]))
    if not model.updates:
        raise ValueError("no trainable episodes; no policy artifact was produced")
    model.save(output)
    return {"updates": model.updates, "multipliers": model.multipliers,
            "training_environment": "controlled_full_action_feedback", "episodes": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    episodes = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(train(episodes, args.output, epochs=args.epochs, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
