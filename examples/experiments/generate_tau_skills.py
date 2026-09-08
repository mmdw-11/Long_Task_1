"""Generate strict JSON τ skills from train trajectories and gate publication."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.experiments.tau_data import load_tau_tasks
from engine.experiments.tau_skills import (
    group_training_tasks, parse_generated_skill, render_generation_prompt, save_tau_skill, validate_tau_skill,
)


def _generate(prompt: str, model: str) -> str:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL"))
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0,
        max_tokens=4000, response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
    )
    return response.choices[0].message.content or ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/tau_skill_v1/tasks.jsonl")
    parser.add_argument("--output-root", default="runs/experiments/tau_skill_v1/skills")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--domain", choices=["retail", "airline"], required=True)
    parser.add_argument("--method", choices=["ours_no_validation", "ours_full", "all"], required=True)
    parser.add_argument("--max-groups", type=int, help="smoke-only cap; omit for the complete library")
    args = parser.parse_args()

    from tau2.runner.build import build_environment
    environment = build_environment(args.domain)
    schemas = [tool.openai_schema for tool in environment.get_tools()]
    train = load_tau_tasks(args.dataset, split="skill_train", domains=[args.domain])
    test_ids = {item.id for item in load_tau_tasks(args.dataset, split="test", domains=[args.domain])}
    available = {tool.name for tool in environment.get_tools()}
    methods = ["ours_no_validation", "ours_full"] if args.method == "all" else [args.method]
    groups = list(group_training_tasks(train).items())
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    for family, family_tasks in groups:
        skill = parse_generated_skill(
            _generate(render_generation_prompt(args.domain, family_tasks, schemas), args.model),
            domain=args.domain,
        )
        skill.id = f"{args.domain}-{family}".replace("_", "-")
        skill.metadata.update({"action_family": family, "generator_model": args.model})
        validate_tau_skill(skill, available_tools=available, test_ids=test_ids)
        for method in methods:
            clone = parse_generated_skill(json.dumps(skill.to_dict()), domain=args.domain)
            clone.status = "published" if method == "ours_no_validation" else "validated"
            if method == "ours_full":
                clone.metadata["publication_blocked_pending_replay"] = True
            paths = save_tau_skill(clone, Path(args.output_root) / method)
            print({"skill": clone.id, "family": family, "method": method, "status": clone.status, **paths})


if __name__ == "__main__":
    main()
