"""Train and demo the learned binary router.

Examples:
    python examples/router_learning/run.py --save runs/router_learning/router.json
    python examples/router_learning/run.py --predict "帮我写一个复杂代码重构方案"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine import (
    BinaryTextRouterModel,
    FallbackCascadeTeacher,
    LearnedTaskGate,
    PseudoCascadeTeacher,
    RealCascadeTeacher,
    ResourceRequest,
    build_route_dataset,
    default_training_texts,
    train_router,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="runs/router_learning/router.json")
    parser.add_argument("--predict", default="")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--teacher", choices=["pseudo", "real", "fallback"], default="real")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    if args.teacher == "pseudo":
        teacher = PseudoCascadeTeacher()
    elif args.teacher == "fallback":
        teacher = FallbackCascadeTeacher(
            primary=RealCascadeTeacher(timeout_seconds=args.timeout)
        )
    else:
        teacher = RealCascadeTeacher(timeout_seconds=args.timeout)
    dataset = build_route_dataset(default_training_texts(), teacher=teacher)
    result = train_router(dataset)
    model = result["model"]
    save_path = Path(args.save)
    model.save(save_path)

    print(json.dumps({"metrics": result["metrics"], "train_size": result["train_size"], "test_size": result["test_size"]}, ensure_ascii=False, indent=2))

    if args.predict:
        loaded = BinaryTextRouterModel.load(save_path)
        gate = LearnedTaskGate(loaded, threshold=args.threshold)
        profile = gate.evaluate(ResourceRequest(node="demo", state={"input": args.predict}))
        print(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
