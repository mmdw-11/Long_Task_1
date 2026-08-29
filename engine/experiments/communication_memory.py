"""Deterministic ablations for low-entropy communication and temporal memory."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from engine.modules.communication import CommunicationManager, CommunicationPolicy, MessageCapsule
from engine.modules.communication.compressor import rough_tokens
from engine.modules.memory import Fact, HybridTieredMemoryStore, TemporalEvidenceMemoryStore

from .reports import ExperimentReport
from .types import ExperimentRow


COMMUNICATION_CASES = [
    {"id":"relevant","goal":"repair tests","claim":"pytest failure is in ledger","recipient":"developer","critical":["pytest","ledger"]},
    {"id":"duplicate","goal":"repair tests","claim":"pytest failure is in ledger","recipient":"developer","critical":["pytest","ledger"]},
    {"id":"role_noise","goal":"write report","claim":"database migration detail","recipient":"writer","critical":["migration"]},
    {"id":"constraint","goal":"release","claim":"","recipient":"planner","constraint_delta":{"deadline":"Friday"},"critical":["Friday"]},
    {"id":"state_change","goal":"release","claim":"task status changed to blocked","recipient":"planner","critical":["blocked"]},
]


def run_communication_ablation(mode: str = "structured_gate_compress") -> ExperimentReport:
    manager = CommunicationManager(CommunicationPolicy(min_contribution=.28, max_capsule_tokens=80))
    rows=[]; started=time.perf_counter()
    for case in COMMUNICATION_CASES:
        raw=f"Goal: {case['goal']}\nClaim: {case['claim']}\nConstraint: {case.get('constraint_delta',{})}" + ("\nRepeated natural-language explanation that every agent already received."*80)
        begin=time.perf_counter(); delivered_text=""; edges=1; pruned=0
        if mode == "free_text_append": delivered_text=raw
        else:
            capsule=MessageCapsule(sender="source",recipients=[case["recipient"]],goal=case["goal"],claim=case["claim"],constraint_delta=case.get("constraint_delta",{}))
            if mode == "structured_no_gate": delivered_text=str(capsule.to_dict())
            else:
                envelopes,_=manager.route(capsule,candidates=[case["recipient"]]); pruned=0 if envelopes else 1; edges=len(envelopes)
                delivered_text=str(envelopes[0].capsule.to_dict()) if envelopes else ""
        fidelity=sum(1 for key in case["critical"] if key.lower() in delivered_text.lower())/len(case["critical"])
        rows.append(ExperimentRow(id=case["id"],passed=fidelity==1.0 or pruned==1,score=fidelity, prediction=delivered_text,expected=",".join(case["critical"]),metrics={"tokens":rough_tokens(delivered_text),"raw_tokens":rough_tokens(raw),"compression_ratio":rough_tokens(delivered_text)/max(1,rough_tokens(raw)),"delivered_edges":edges,"pruned":pruned,"critical_fidelity":fidelity,"latency_ms":(time.perf_counter()-begin)*1000},metadata={"mode":mode}))
    return ExperimentReport(f"communication-{mode}",rows,{"mode":mode,"seed":42,"seconds":time.perf_counter()-started,"manager_metrics":manager.metrics})


def run_temporal_memory_ablation(*, temporal: bool = True) -> ExperimentReport:
    rows=[]; started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="agent-graph-temporal-") as root:
        base=HybridTieredMemoryStore(Path(root),enable_memory_update=False); store=TemporalEvidenceMemoryStore(base,clock=lambda:30)
        store.upsert_fact(Fact(subject="project",predicate="status",object="planned",valid_from=1))
        store.upsert_fact(Fact(subject="project",predicate="status",object="running",valid_from=10),replace=True)
        store.upsert_fact(Fact(subject="project",predicate="owner",object="Alice",valid_from=1))
        store.upsert_fact(Fact(subject="project",predicate="owner",object="Bob",valid_from=1))
        cases=[("historical",5,"planned"),("current",20,"running"),("conflict",20,"disputed")]
        for cid,instant,expected in cases:
            begin=time.perf_counter()
            if temporal:
                hits=store.retrieve_facts(f"project {'owner' if cid=='conflict' else 'status'}",as_of=instant,include_conflicts=cid=="conflict",top_k=5)
                relevant = [x for x in hits if x.fact.predicate == ("owner" if cid == "conflict" else "status")]
                prediction=" ".join(str(x.fact.object) for x in relevant); conflict_ok=cid!="conflict" or len(relevant)==2
            else:
                all_rows=base._get_conn().execute("SELECT object_json FROM facts ORDER BY updated_at DESC LIMIT 1").fetchall(); prediction=" ".join(str(x[0]) for x in all_rows); conflict_ok=False if cid=="conflict" else True
            passed=(expected in prediction.lower()) if cid!="conflict" else conflict_ok
            rows.append(ExperimentRow(id=cid,passed=passed,score=1.0 if passed else 0.0,prediction=prediction,expected=expected,metrics={"temporal_accuracy":1 if passed and cid!="conflict" else 0,"conflict_accuracy":1 if passed and cid=="conflict" else 0,"stale_fact":1 if cid=="current" and "planned" in prediction else 0,"latency_ms":(time.perf_counter()-begin)*1000},metadata={"temporal":temporal}))
        base.close()
    return ExperimentReport("temporal-memory" if temporal else "tiered-memory-baseline",rows,{"temporal":temporal,"seed":42,"seconds":time.perf_counter()-started})
