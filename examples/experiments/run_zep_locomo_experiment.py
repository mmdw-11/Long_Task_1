"""Evaluate Zep Cloud v4 on the frozen LoCoMo splits with the shared QA judge."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.experiments import ExperimentReport, MemoryExperimentConfig, load_memory_dataset, save_report
from engine.experiments.io import read_json_or_jsonl, write_json
from engine.experiments.memory import _evaluate_example, _fit_context_to_budget


def _history_episodes(conversation):
    """Keep speaker attribution and each source session's original date."""
    for name, turns in conversation.items():
        if not re.fullmatch(r"session_\d+", name):
            continue
        reference_time = datetime.strptime(
            conversation[f"{name}_date_time"], "%I:%M %p on %d %B, %Y",
        ).replace(tzinfo=timezone.utc).isoformat()
        lines = [
            f"{turn['speaker']}: {turn['text']}"
            for turn in turns if turn.get("text")
        ]
        chunk = ""
        for line in lines:
            if chunk and len(chunk) + len(line) + 1 > 1800:
                yield reference_time, chunk
                chunk = ""
            chunk = f"{chunk}\n{line}" if chunk else line
        if chunk:
            yield reference_time, chunk


def _wait_for_ingestion(client, batch_uuid, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        batch = client.batch.get(batch_uuid)
        if batch.status == "succeeded":
            return
        if batch.status in {"partial", "failed", "invalid", "canceled"}:
            raise RuntimeError(f"Zep ingestion ended with status {batch.status}; see batch {batch_uuid}")
        time.sleep(5)
    raise TimeoutError(f"Zep ingestion is still pending; see batch {batch_uuid}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-locomo", required=True, type=Path)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--final", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ingestion-timeout", type=int, default=3600)
    args = parser.parse_args()
    api_key = os.environ.get("ZEP_API_KEY")
    if not api_key:
        parser.error("configure ZEP_API_KEY in the server environment")

    from zep_cloud import BatchItemInput
    from zep_cloud.client import Zep

    splits = {
        name: load_memory_dataset(path, source="locomo")
        for name, path in (("pilot", args.pilot), ("final", args.final))
    }
    ids = sorted({example.trajectory_id for examples in splits.values() for example in examples})
    raw = {str(item["sample_id"]): item["conversation"] for item in read_json_or_jsonl(args.raw_locomo)}
    episodes = {conv_id: list(_history_episodes(raw[conv_id])) for conv_id in ids}
    if not all(episodes.values()):
        raise ValueError("one or more LoCoMo conversations have no usable episodes")

    client = Zep(api_key=api_key)
    namespace = hashlib.sha256(
        str(args.output_dir.resolve()).encode() + hashlib.sha256(args.raw_locomo.read_bytes()).digest()
        + "|".join(ids).encode()
    ).hexdigest()[:24]
    graphs = {
        conv_id: client.graph.create(
            name=f"TIDE LoCoMo {conv_id}", idempotency_key=f"{namespace}-graph-{index}",
        ).uuid_
        for index, conv_id in enumerate(ids)
    }
    batch_uuid = client.batch.create(
        metadata={"description": "TIDE LoCoMo benchmark"}, idempotency_key=f"{namespace}-batch",
    ).uuid_
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "zep_ingestion.json", {"graphs": graphs, "batch_uuid": batch_uuid})
    items = [
        BatchItemInput(
            type="graph_episode", data_type="text", graph_uuid=graphs[conv_id],
            reference_time=stamp, data=text, source_description="Public LoCoMo transcript",
        )
        for conv_id in ids for stamp, text in episodes[conv_id]
    ]
    batch = client.batch.get(batch_uuid)
    if batch.status == "draft":
        for offset in range(0, len(items), 100):
            client.batch.add_items(
                batch_uuid, items=items[offset:offset + 100],
                idempotency_key=f"{namespace}-items-{offset}",
            )
        client.batch.process(batch_uuid, idempotency_key=f"{namespace}-process")
    print(f"Waiting for Zep ingestion of {len(items)} episodes", flush=True)
    _wait_for_ingestion(client, batch_uuid, args.ingestion_timeout)

    cfg = MemoryExperimentConfig(top_k=5, context_token_budget=12000)
    for split, examples in splits.items():
        rows = []
        for example in examples:
            started = time.perf_counter()
            result = client.graph.search_edges(
                graphs[example.trajectory_id], query=example.question, page_size=cfg.top_k,
            )
            facts = [edge.fact for edge in (result.items or []) if edge.fact]
            visible = _fit_context_to_budget(facts, cfg.context_token_budget)
            rows.append(_evaluate_example(
                example, visible, cfg, write_seconds=0.0, retrieval_applicable=True,
                retrieval_seconds=time.perf_counter() - started,
                metadata={"backend": "zep_cloud_graph_edges", "source": "locomo"},
            ))
        report = ExperimentReport(
            name="memory-zep", rows=rows,
            metadata={"top_k": cfg.top_k, "context_token_budget": cfg.context_token_budget,
                      "graph_count": len(graphs), "ingestion": "shared offline batch"},
        )
        print(save_report(report, args.output_dir / split / "locomo" / "zep"), flush=True)
        if any(row.metadata.get("qa_error") or row.metadata.get("qa_judge_error") for row in rows):
            raise RuntimeError(f"{split} contains QA provider errors; inspect rows.jsonl")


if __name__ == "__main__":
    main()
