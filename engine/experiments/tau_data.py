"""Freeze and audit official tau2-bench tasks without importing tau2.

The converter intentionally reads the pinned checkout as ordinary JSON.  This
lets dataset preparation and leakage tests run in the project's normal Python
environment; only live simulations require the dedicated ``.venv-tau``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .types import TauToolTask


TAU_COMMIT = "672227c6b6676edc20d57ea53b7000262aae77b9"
TAU_REPOSITORY = "https://github.com/sierra-research/tau2-bench"
TAU_LICENSE = "MIT"
SUPPORTED_DOMAINS = ("retail", "airline")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_bucket(value: str, seed: int, modulo: int = 10_000) -> int:
    raw = f"tau-skill-v1:{seed}:{value}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % modulo


def split_train_ids(ids: Iterable[str], seed: int = 42, validation_ratio: float = 0.2) -> dict[str, list[str]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between zero and one")
    ordered = sorted({str(item) for item in ids}, key=lambda item: (stable_bucket(item, seed), item))
    count = max(1, round(len(ordered) * validation_ratio)) if ordered else 0
    return {"skill_validation": ordered[:count], "skill_train": ordered[count:]}


def deterministic_sample(ids: Iterable[str], count: int, seed: int, namespace: str) -> list[str]:
    values = sorted({str(item) for item in ids})
    if count < 0 or count > len(values):
        raise ValueError(f"cannot sample {count} from {len(values)} {namespace} tasks")
    return sorted(values, key=lambda item: (stable_bucket(f"{namespace}:{item}", seed), item))[:count]


def _domain_dir(checkout: Path, domain: str) -> Path:
    if domain not in SUPPORTED_DOMAINS:
        raise ValueError(f"unsupported tau domain: {domain}")
    return checkout / "data" / "tau2" / "domains" / domain


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _action_kinds(actions: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    reads: list[str] = []
    writes: list[str] = []
    for action in actions:
        name = str(action.get("name") or "")
        if not name:
            continue
        # Official retail/airline tools consistently use these read prefixes.
        if name.startswith(("get_", "find_", "search_", "list_", "calculate_")):
            reads.append(name)
        else:
            writes.append(name)
    return list(dict.fromkeys(reads)), list(dict.fromkeys(writes))


def _task_row(
    raw: dict[str, Any], *, domain: str, split: str, policy: str,
    source_commit: str, task_hash: str, db_hash: str, tool_schemas: list[dict[str, Any]],
) -> TauToolTask:
    criteria = raw.get("evaluation_criteria") or {}
    actions = list(criteria.get("actions") or [])
    reads, writes = _action_kinds(actions)
    initial_payload = json.dumps(raw.get("initial_state"), ensure_ascii=False, sort_keys=True)
    initial_hash = hashlib.sha256(f"{db_hash}:{initial_payload}".encode("utf-8")).hexdigest()
    return TauToolTask(
        id=f"{domain}:{raw['id']}",
        domain=domain,
        split=split,
        user_scenario=dict(raw.get("user_scenario") or {}),
        domain_policy=policy,
        tool_schemas=tool_schemas,
        initial_state_ref=f"{domain}:initial:{initial_hash}",
        reference_actions=actions,
        reward_basis=[str(item) for item in criteria.get("reward_basis") or []],
        communicate_info=[str(item) for item in criteria.get("communicate_info") or []],
        env_assertions=list(criteria.get("env_assertions") or []),
        trajectory=actions,
        tool_calls=actions,
        read_actions=reads,
        write_actions=writes,
        confirmation_boundaries=writes,
        source_commit=source_commit,
        source_hash=task_hash,
        metadata={
            "official_task_id": str(raw["id"]),
            "description": raw.get("description"),
            "nl_assertions": criteria.get("nl_assertions") or [],
            "annotations": raw.get("annotations"),
        },
    )


def prepare_tau_dataset(
    checkout: str | Path,
    output_dir: str | Path,
    *, domains: Iterable[str] = SUPPORTED_DOMAINS,
    test_count_per_domain: int = 20,
    seed: int = 42,
    validation_ratio: float = 0.2,
    source_commit: str = TAU_COMMIT,
    tool_schema_loader: Callable[[str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    checkout = Path(checkout)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[TauToolTask] = []
    split_ids: dict[str, dict[str, list[str]]] = {}
    source_files: dict[str, dict[str, Any]] = {}

    if tool_schema_loader is None:
        try:
            from tau2.runner.build import build_environment
        except ImportError as exc:
            raise RuntimeError("prepare tau data inside .venv-tau so official tool schemas can be frozen") from exc
        tool_schema_loader = lambda name: [tool.openai_schema for tool in build_environment(name).get_tools()]

    for domain in domains:
        domain_dir = _domain_dir(checkout, domain)
        tasks_path = domain_dir / "tasks.json"
        splits_path = domain_dir / "split_tasks.json"
        policy_path = domain_dir / "policy.md"
        db_path = domain_dir / "db.json"
        for path in (tasks_path, splits_path, policy_path, db_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        raw_tasks = _read_json(tasks_path)
        if not isinstance(raw_tasks, list):
            raise ValueError(f"unknown task schema in {tasks_path}")
        by_id = {str(item.get("id")): item for item in raw_tasks}
        if len(by_id) != len(raw_tasks) or "None" in by_id:
            raise ValueError(f"duplicate or missing IDs in {tasks_path}")
        official = _read_json(splits_path)
        train_ids = [str(item) for item in official.get("train") or []]
        test_ids = [str(item) for item in official.get("test") or []]
        if set(train_ids) & set(test_ids):
            raise ValueError(f"official train/test overlap in {domain}")
        unknown = (set(train_ids) | set(test_ids)) - set(by_id)
        if unknown:
            raise ValueError(f"split references unknown {domain} IDs: {sorted(unknown)}")
        derived = split_train_ids(train_ids, seed=seed, validation_ratio=validation_ratio)
        selected_test = deterministic_sample(test_ids, test_count_per_domain, seed, domain)
        splits = {**derived, "test": selected_test}
        split_ids[domain] = splits
        policy = policy_path.read_text(encoding="utf-8")
        tool_schemas = tool_schema_loader(domain)
        if not tool_schemas or any(not item.get("function", {}).get("name") for item in tool_schemas):
            raise ValueError(f"missing or unknown official tool schema for {domain}")
        task_hash = sha256_file(tasks_path)
        db_hash = sha256_file(db_path)
        for split, ids in splits.items():
            for task_id in ids:
                all_rows.append(_task_row(
                    by_id[task_id], domain=domain, split=split, policy=policy,
                    source_commit=source_commit, task_hash=task_hash, db_hash=db_hash,
                    tool_schemas=tool_schemas,
                ))
        source_files[domain] = {
            "tasks_sha256": task_hash,
            "split_sha256": sha256_file(splits_path),
            "policy_sha256": sha256_file(policy_path),
            "db_sha256": db_hash,
            "official_train": len(train_ids),
            "official_test": len(test_ids),
        }

    scenario_hashes: dict[str, set[str]] = {}
    for item in all_rows:
        serialized = json.dumps(item.user_scenario, ensure_ascii=False, sort_keys=True).encode("utf-8")
        scenario_hashes.setdefault(item.split, set()).add(hashlib.sha256(serialized).hexdigest())
    leakage = {
        "id_overlap": bool(
            {item.id for item in all_rows if item.split != "test"}
            & {item.id for item in all_rows if item.split == "test"}
        ),
        "scenario_overlap": bool(
            (scenario_hashes.get("skill_train", set()) | scenario_hashes.get("skill_validation", set()))
            & scenario_hashes.get("test", set())
        ),
    }
    if any(leakage.values()):
        raise ValueError(f"tau split leakage detected: {leakage}")

    rows_path = output_dir / "tasks.jsonl"
    with rows_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in sorted(all_rows, key=lambda row: (row.domain, row.split, row.id)):
            handle.write(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n")

    stats: dict[str, Any] = {}
    for domain in domains:
        domain_rows = [item for item in all_rows if item.domain == domain]
        stats[domain] = {
            "splits": {name: len(ids) for name, ids in split_ids[domain].items()},
            "reference_action_depth": {
                "min": min((len(item.reference_actions) for item in domain_rows), default=0),
                "max": max((len(item.reference_actions) for item in domain_rows), default=0),
                "mean": round(sum(len(item.reference_actions) for item in domain_rows) / max(1, len(domain_rows)), 3),
            },
            "read_action_count": sum(len(item.read_actions) for item in domain_rows),
            "write_action_count": sum(len(item.write_actions) for item in domain_rows),
            "rejection_tasks": sum(not item.write_actions for item in domain_rows if item.split == "test"),
            "reward_basis": sorted({basis for item in domain_rows for basis in item.reward_basis}),
            "tool_count": len(tool_schemas),
        }
    manifest = {
        "schema_version": "tau_skill_v1",
        "repository": TAU_REPOSITORY,
        "source_commit": source_commit,
        "license": TAU_LICENSE,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "validation_ratio": validation_ratio,
        "test_count_per_domain": test_count_per_domain,
        "source_files": source_files,
        "splits": split_ids,
        "leakage": leakage,
        "statistics": stats,
        "tasks_sha256": sha256_file(rows_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    test_index_path = output_dir / "test_index.json"
    test_index_path.write_text(
        json.dumps({domain: split_ids[domain]["test"] for domain in domains}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"tasks": str(rows_path), "manifest": str(manifest_path), "test_index": str(test_index_path), **manifest}


def load_tau_tasks(path: str | Path, *, split: str | None = None, domains: Iterable[str] | None = None) -> list[TauToolTask]:
    allowed = set(domains or SUPPORTED_DOMAINS)
    rows: list[TauToolTask] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            item = TauToolTask(**payload)
            if (split is None or item.split == split) and item.domain in allowed:
                rows.append(item)
    ids = [item.id for item in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task IDs in selected tau dataset")
    return rows
