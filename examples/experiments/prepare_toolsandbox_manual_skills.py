"""Build the expert baseline solely from public ToolSandbox schemas and rules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.experiments.toolsandbox_skills import ToolSandboxSkill


SPECS = {
    "device-state": {
        "terms": ["wifi", "cellular", "location", "battery", "setting"],
        "pre": ["Read the requested setting and all documented prerequisite settings before mutation.", "Low-battery mode can block service activation; disable it only when necessary for the user's requested outcome."],
        "steps": ["Call the matching get_*_status tool.", "Repair only the prerequisite named by a tool error.", "Call the requested set_*_status tool.", "Read the final status to verify."],
    },
    "messaging-contacts": {
        "terms": ["message", "contact", "phone", "recipient", "sender"],
        "pre": ["Resolve a person to a unique stored contact before using a phone number.", "Check cellular service before sending a message."],
        "steps": ["Search messages or contacts using the user's exact known fields.", "If multiple entities match, ask the user to disambiguate.", "Carry returned IDs/phone numbers without alteration.", "For sending, satisfy cellular prerequisites and verify the returned message."],
    },
    "reminders-time": {
        "terms": ["reminder", "time", "date", "latest", "oldest", "upcoming", "yesterday", "week"],
        "pre": ["Use the current-time tool before interpreting relative dates.", "Identify a unique reminder before modifying or removing it."],
        "steps": ["Read current time when the request is relative.", "Search reminders with supported filters.", "Resolve latest/oldest by returned timestamps, not list position.", "Canonicalize to an absolute timestamp.", "Perform the requested reminder operation and verify."],
    },
    "insufficient-information": {
        "terms": ["missing", "unknown", "insufficient", "clarify", "cannot", "which"],
        "pre": ["List required schema fields and mark each as user-known, tool-derivable, or unavailable."],
        "steps": ["Use read-only tools for derivable fields.", "Ask one focused question for a user-knowable missing field.", "If a required tool or fact is unavailable, explain the limitation and end without state changes."],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    target = args.output_root / "manual_skill"
    target.mkdir(parents=True, exist_ok=True)
    for key, spec in SPECS.items():
        skill = ToolSandboxSkill(
            skill_id=f"manual-{key}-v1", name=f"Expert ToolSandbox {key} procedure", version=1,
            source_type="manual_public_docs", source_trajectory_keys=[], source_families=[],
            applicable_when=[f"Visible request or tool feedback concerns: {', '.join(spec['terms'])}."],
            not_applicable_when=["No visible request, available tool, or tool feedback matches this procedure."],
            required_tools=[], required_slots=list(spec["terms"]), preconditions=list(spec["pre"]),
            ordered_steps=[{"order": i + 1, "instruction": value} for i, value in enumerate(spec["steps"])],
            canonicalization_rules=["Use tool-returned IDs exactly.", "Convert relative time only after reading current time."],
            clarification_rules=["Ask only for required fields that cannot be derived with available read tools."],
            abstention_rules=["Do not guess unavailable facts, entities, arguments, or tools."],
            recovery_paths=[{"on": "tool error", "then": "inspect the stated prerequisite or argument, correct it once, and retry"}],
            safety_rules=["Make no unrelated state changes.", "Do not call a distractor tool merely because its name is similar."],
            success_checks=["Verify the requested state or returned entity using tool output before claiming success."],
            status="published", metadata={"basis": "public tool schemas and documented ToolSandbox behavior"},
        )
        (target / f"{skill.skill_id}.json").write_text(json.dumps(skill.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print({"method": "manual_skill", "skills": len(SPECS), "output": str(target)})


if __name__ == "__main__":
    main()
