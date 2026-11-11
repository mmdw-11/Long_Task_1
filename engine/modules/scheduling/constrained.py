"""Small finite-horizon constrained actor--critic, with no ML runtime dependency.

The existing BGE task gate supplies semantic task profiles. This policy consumes
those profiles plus live resources/budgets. It is trained explicitly on sampled
episodes, never on HTTP success or unverified model self-scores.
"""

from __future__ import annotations

import json
import math
import random
import copy
from pathlib import Path
from typing import Any

from ._types import COMPLEXITY_SCORE, SENSITIVITY_SCORE

BUDGET_KEY = "__placement_budget__"
TRAJECTORY_KEY = "__placement_trajectory__"
COSTS = ("fee", "time_ms", "failures")
FEATURES = 14


def merge_budget(old, new):
    """Merge independent attempt charges without dropping parallel-node costs."""
    if not old:
        return copy.deepcopy(new)
    result = copy.deepcopy(new)
    limits = dict(old.get("limits") or {})
    for key, value in (new.get("limits") or {}).items():
        limits[key] = min(finite(limits[key]), finite(value)) if key in limits else finite(value)
    result["limits"] = limits
    result["charges"] = {**(old.get("charges") or {}), **(new.get("charges") or {})}
    result["base_spent"] = dict(old.get("base_spent") or old.get("spent") or {}) if not old.get("charges") else dict(old.get("base_spent") or {})
    result["unpriced_call"] = bool(old.get("unpriced_call") or new.get("unpriced_call"))
    recalculate_spent(result)
    return result


def merge_trajectory(old, new):
    items = {item["decision_id"]: item for item in (old or [])}
    items.update({item["decision_id"]: item for item in (new or [])})
    return list(items.values())


def recalculate_spent(budget):
    spent = dict(budget.get("base_spent") or {})
    intervals = []
    for charge in (budget.get("charges") or {}).values():
        for key, value in charge["costs"].items():
            if key != "time_ms":
                spent[key] = spent.get(key, 0.) + value
        intervals.append(charge["interval"])
    end = None
    duration = 0.
    for start, finish in sorted(intervals):
        duration += max(0., finish - max(start, end if end is not None else start))
        end = max(finish, end) if end is not None else finish
    spent["time_ms"] = spent.get("time_ms", 0.) + duration
    budget["spent"] = spent


class PlacementInfeasible(RuntimeError):
    """No action satisfies the active hard constraints or remaining budget."""


def finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("placement values must be finite")
    return number


def budget_for(request):
    budget = request.state.setdefault(BUDGET_KEY, {
        "limits": dict(request.metadata.get("placement_budget") or {}), "spent": {},
    })
    limits = budget.setdefault("limits", {})
    spent = budget.setdefault("spent", {})
    for key in COSTS:
        if key in limits and finite(limits[key]) < 0:
            raise ValueError("negative placement budget")
        if key in spent and finite(spent[key]) < 0:
            raise ValueError("negative placement consumption")
    return budget


def action_features(profile, resource, status, budget):
    remaining = [max(0., 1. - finite(budget["spent"].get(key, 0)) / max(1e-9, finite(budget["limits"][key])))
                 if key in budget["limits"] else 1. for key in COSTS]
    latency = status.latency_ms if status and status.latency_ms is not None else resource.latency_ms
    route = str(profile.metadata.get("route_tier") or "")
    return [
        1., COMPLEXITY_SCORE[profile.complexity] / 4., SENSITIVITY_SCORE[profile.sensitivity] / 3.,
        float(profile.realtime.value == "hard"), *remaining,
        float(route == resource.tier.value),
        float(resource.tier.value == "device"), float(resource.tier.value == "edge"),
        float(resource.trusted), min(10., max(0., finite(latency)) / 1000.),
        min(10., max(0., finite(resource.cost_weight))),
        min(1., max(0., finite(status.load or 0.))) if status else 0.,
    ]


class ConstrainedPlacementPolicy:
    """Tanh MLP actor, separate reward/cost value heads, episode dual updates."""

    def __init__(self, *, hidden: int = 8, seed: int = 0):
        if not 1 <= hidden <= 128:
            raise ValueError("hidden size must be between 1 and 128")
        rng = random.Random(seed)
        self.hidden = hidden
        self.weights = [[rng.uniform(-.1, .1) for _ in range(FEATURES)] for _ in range(hidden)]
        self.output = [rng.uniform(-.1, .1) for _ in range(hidden)]
        self.values = {key: [0.] * FEATURES for key in ("reward", *COSTS)}
        self.multipliers = {key: 0. for key in COSTS}
        self.updates = 0

    @property
    def version(self):
        return self.updates

    def _forward(self, features):
        if len(features) != FEATURES:
            raise ValueError("invalid placement feature size")
        x = [finite(value) for value in features]
        hidden = [math.tanh(sum(a * b for a, b in zip(row, x))) for row in self.weights]
        return hidden, sum(a * b for a, b in zip(self.output, hidden))

    def probabilities(self, features):
        if not features:
            raise PlacementInfeasible("no safe placement action")
        scores = {key: finite(self._forward(value)[1]) for key, value in features.items()}
        maximum = max(scores.values())
        weights = {key: math.exp(value - maximum) for key, value in scores.items()}
        total = sum(weights.values())
        return {key: value / total for key, value in weights.items()}

    def choose(self, features, *, sample=False, rng=None):
        probabilities = self.probabilities(features)
        action = (rng or random).choices(list(probabilities), weights=list(probabilities.values()))[0] if sample else max(probabilities, key=probabilities.get)
        return action, probabilities

    def learn_episode(self, steps, limits, *, actor_lr=.01, value_lr=.01, dual_lr=.05):
        """Update once from a complete on-policy episode, with normalized costs.

        Each step has features, action, probabilities, policy_version,
        sampled=True, a verifier-supplied reward, and measured costs.
        Deterministic deployment logs and trajectories from older policies are
        rejected; a training caller must explicitly close the entire episode.
        """
        if not steps:
            raise ValueError("empty placement episode")
        if steps[-1].get("terminal") is not True:
            raise ValueError("training requires a complete terminal episode")
        for step in steps:
            if step.get("reward") is None or step.get("costs_complete", True) is not True:
                raise ValueError("training requires verified reward and complete costs")
            if step.get("policy_version") != self.version or step.get("sampled") is not True:
                raise ValueError("training requires current sampled policy episodes")
            expected = self.probabilities(step["features"])
            if step["action"] not in expected or set(step.get("probabilities", {})) != set(expected) or any(abs(finite(step["probabilities"][key]) - value) > 1e-6 for key, value in expected.items()):
                raise ValueError("behavior probabilities do not match the policy")
            finite(step["reward"])
            for key in COSTS:
                if finite(step["costs"].get(key, 0)) < 0:
                    raise ValueError("negative measured cost")
        scales = {key: max(1e-9, finite(limits[key])) if key in limits else 1. for key in COSTS}
        if any(finite(value) < 0 for value in limits.values()):
            raise ValueError("negative episode limit")
        actor_grad = [[0.] * FEATURES for _ in self.weights]
        output_grad = [0.] * self.hidden
        value_grad = {key: [0.] * FEATURES for key in self.values}
        returns = {key: 0. for key in self.values}
        totals = {key: sum(finite(step["costs"].get(key, 0)) for step in steps) for key in COSTS}
        for step in reversed(steps):
            features = step["features"]
            # Average safe-action features is a state-only baseline.
            state = [sum(values[index] for values in features.values()) / len(features) for index in range(FEATURES)]
            returns["reward"] += finite(step["reward"])
            for key in COSTS:
                returns[key] += finite(step["costs"].get(key, 0)) / scales[key]
            advantages = {}
            for key, weights in self.values.items():
                advantages[key] = returns[key] - sum(a * b for a, b in zip(weights, state))
                for index in range(FEATURES):
                    value_grad[key][index] += max(-10., min(10., advantages[key])) * state[index] / len(steps)
            advantage = max(-10., min(10., advantages["reward"] - sum(self.multipliers[key] * advantages[key] for key in COSTS)))
            for action, values in features.items():
                hidden, _ = self._forward(values)
                coefficient = (float(action == step["action"]) - step["probabilities"][action]) * advantage / len(steps)
                for unit in range(self.hidden):
                    output_grad[unit] += coefficient * hidden[unit]
                    for index in range(FEATURES):
                        actor_grad[unit][index] += coefficient * self.output[unit] * (1. - hidden[unit] ** 2) * values[index]
        for unit in range(self.hidden):
            self.output[unit] += actor_lr * output_grad[unit]
            for index in range(FEATURES):
                self.weights[unit][index] += actor_lr * actor_grad[unit][index]
        for key in self.values:
            self.values[key] = [value + value_lr * grad for value, grad in zip(self.values[key], value_grad[key])]
        for key in COSTS:
            if key in limits:
                self.multipliers[key] = max(0., self.multipliers[key] + dual_lr * (totals[key] / scales[key] - 1.))
        self.updates += 1
        return {"policy_version": self.version, "costs": totals, "multipliers": dict(self.multipliers)}

    def save(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"schema_version": 1, "features": FEATURES, "hidden": self.hidden,
            "weights": self.weights, "output": self.output, "values": self.values,
            "multipliers": self.multipliers, "updates": self.updates}), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or data.get("features") != FEATURES or not 1 <= int(data["hidden"]) <= 128:
            raise ValueError("unsupported placement policy")
        model = cls(hidden=int(data["hidden"]))
        if len(data["weights"]) != model.hidden or len(data["output"]) != model.hidden or any(len(row) != FEATURES for row in data["weights"]) or set(data["values"]) != set(model.values) or any(len(row) != FEATURES for row in data["values"].values()) or set(data["multipliers"]) != set(COSTS):
            raise ValueError("invalid placement policy shape")
        for value in [*data["output"], *data["multipliers"].values(), *(value for row in data["weights"] for value in row), *(value for row in data["values"].values() for value in row)]:
            finite(value)
        if any(value < 0 for value in data["multipliers"].values()) or int(data["updates"]) < 0:
            raise ValueError("invalid placement policy state")
        model.weights, model.output, model.values = data["weights"], data["output"], data["values"]
        model.multipliers, model.updates = data["multipliers"], int(data["updates"])
        return model
