"""Temporal evidence memory layered on top of HybridTieredMemoryStore."""
from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..communication import EvidenceRef, MessageCapsule
from .store import HybridTieredMemoryStore
from ._utils import _cosine, _sparse_score, _tokenize


class FactStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"
    DISPUTED = "disputed"


@dataclass
class TemporalEvidence:
    content: str
    source: str
    source_type: str = "agent"
    confidence: float = 0.5
    observed_at: float = field(default_factory=time.time)
    task_id: str = ""
    agent_id: str = ""
    trace_id: str = ""
    capsule_digest: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"ev-{uuid.uuid4().hex}")

    def to_dict(self) -> Dict[str, Any]: return asdict(self)
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalEvidence": return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Fact:
    subject: str
    predicate: str
    object: Any
    valid_from: float = field(default_factory=time.time)
    valid_to: Optional[float] = None
    status: FactStatus = FactStatus.ACTIVE
    confidence: float = 0.5
    version: int = 1
    supersedes_id: str = ""
    conflict_set_id: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    task_id: str = ""
    agent_id: str = ""
    trace_id: str = ""
    id: str = field(default_factory=lambda: f"fact-{uuid.uuid4().hex}")
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self); data["status"] = self.status.value; return data
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fact":
        payload = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        payload["status"] = FactStatus(payload.get("status", "active")); return cls(**payload)


@dataclass
class FactSearchResult:
    fact: Fact
    score: float
    score_breakdown: Dict[str, float]
    evidence: List[TemporalEvidence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]: return {"fact": self.fact.to_dict(), "score": self.score, "score_breakdown": self.score_breakdown, "evidence": [x.to_dict() for x in self.evidence]}


@dataclass
class TemporalRetrievalWeights:
    semantic: float = 0.30
    sparse: float = 0.15
    temporal: float = 0.16
    source: float = 0.13
    evidence: float = 0.10
    scope: float = 0.09
    recency: float = 0.07
    conflict_penalty: float = 0.20


class TemporalEvidenceMemoryStore:
    """Composition layer: shares the base store's SQLite connection and archive."""
    SCHEMA_VERSION = 1

    def __init__(self, base: HybridTieredMemoryStore, *, source_reliability: Optional[Dict[str, float]] = None, weights: Optional[TemporalRetrievalWeights] = None, clock: Callable[[], float] = time.time) -> None:
        self.base = base; self.clock = clock; self.weights = weights or TemporalRetrievalWeights()
        self.source_reliability = {"tool": .9, "human": .95, "agent": .7, "node_update": .6, **(source_reliability or {})}
        self._init_db()

    def _init_db(self) -> None:
        c = self.base._get_conn()
        c.executescript("""
        CREATE TABLE IF NOT EXISTS temporal_schema(version INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS temporal_evidence(id TEXT PRIMARY KEY, content TEXT NOT NULL, source TEXT NOT NULL, source_type TEXT NOT NULL, confidence REAL NOT NULL, observed_at REAL NOT NULL, task_id TEXT, agent_id TEXT, trace_id TEXT, capsule_digest TEXT, metadata_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS facts(id TEXT PRIMARY KEY, subject TEXT NOT NULL, subject_norm TEXT NOT NULL, predicate TEXT NOT NULL, predicate_norm TEXT NOT NULL, object_json TEXT NOT NULL, object_norm TEXT NOT NULL, valid_from REAL NOT NULL, valid_to REAL, status TEXT NOT NULL, confidence REAL NOT NULL, version INTEGER NOT NULL, supersedes_id TEXT, conflict_set_id TEXT, task_id TEXT, agent_id TEXT, trace_id TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL, metadata_json TEXT NOT NULL, embedding_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS fact_evidence(fact_id TEXT NOT NULL, evidence_id TEXT NOT NULL, PRIMARY KEY(fact_id,evidence_id));
        CREATE TABLE IF NOT EXISTS fact_conflicts(conflict_set_id TEXT NOT NULL, fact_id TEXT NOT NULL, reason TEXT NOT NULL, created_at REAL NOT NULL, PRIMARY KEY(conflict_set_id,fact_id));
        CREATE TABLE IF NOT EXISTS fact_versions(fact_id TEXT NOT NULL, version INTEGER NOT NULL, snapshot_json TEXT NOT NULL, changed_at REAL NOT NULL, PRIMARY KEY(fact_id,version));
        CREATE INDEX IF NOT EXISTS idx_fact_sp ON facts(subject_norm,predicate_norm);
        CREATE INDEX IF NOT EXISTS idx_fact_validity ON facts(valid_from,valid_to,status);
        CREATE INDEX IF NOT EXISTS idx_fact_scope ON facts(task_id,agent_id,trace_id);
        CREATE INDEX IF NOT EXISTS idx_evidence_scope ON temporal_evidence(task_id,agent_id,trace_id,observed_at);
        """)
        if c.execute("SELECT COUNT(*) FROM temporal_schema").fetchone()[0] == 0: c.execute("INSERT INTO temporal_schema(version) VALUES (?)", (self.SCHEMA_VERSION,))
        c.commit()

    def upsert_fact(self, fact: Fact, evidence: Optional[Sequence[TemporalEvidence]] = None, *, replace: bool = False) -> Fact:
        now = self.clock(); fact.updated_at = now
        existing = self._active_for_sp(fact.subject, fact.predicate)
        same = next((x for x in existing if _norm_obj(x.object) == _norm_obj(fact.object) and _overlap_interval(x.valid_from, x.valid_to, fact.valid_from, fact.valid_to)), None)
        if same:
            same.confidence = 1.0 - (1.0 - same.confidence) * (1.0 - fact.confidence)
            same.updated_at = now; same.version += 1; self._persist_fact(same)
            self._link_evidence(same, evidence or []); return same
        overlaps = [x for x in existing if _overlap_interval(x.valid_from, x.valid_to, fact.valid_from, fact.valid_to)]
        if overlaps and replace:
            previous = max(overlaps, key=lambda x: x.version)
            previous.status = FactStatus.SUPERSEDED; previous.valid_to = min(fact.valid_from, previous.valid_to or fact.valid_from); previous.updated_at = now; self._persist_fact(previous)
            fact.supersedes_id = previous.id; fact.version = previous.version + 1
        elif overlaps:
            conflict_id = next((x.conflict_set_id for x in overlaps if x.conflict_set_id), f"conflict-{uuid.uuid4().hex}")
            fact.status = FactStatus.DISPUTED; fact.conflict_set_id = conflict_id
            for other in overlaps:
                other.status = FactStatus.DISPUTED; other.conflict_set_id = conflict_id; other.updated_at = now; self._persist_fact(other); self._persist_conflict(conflict_id, other.id, "overlapping contradictory object")
            self._persist_conflict(conflict_id, fact.id, "overlapping contradictory object")
        self._persist_fact(fact); self._link_evidence(fact, evidence or []); return fact

    def invalidate_fact(self, fact_id: str, *, at: Optional[float] = None, reason: str = "") -> Fact:
        fact = self._get_fact(fact_id)
        if fact is None: raise KeyError(fact_id)
        fact.status = FactStatus.INVALIDATED; fact.valid_to = at or self.clock(); fact.updated_at = self.clock(); fact.version += 1; fact.metadata = {**fact.metadata, "invalidation_reason": reason}
        self._persist_fact(fact); return fact

    def detect_conflicts(self, *, subject: Optional[str] = None, predicate: Optional[str] = None) -> List[List[Fact]]:
        sql = "SELECT DISTINCT conflict_set_id FROM facts WHERE conflict_set_id <> ''"; params: List[Any] = []
        if subject: sql += " AND subject_norm=?"; params.append(_norm(subject))
        if predicate: sql += " AND predicate_norm=?"; params.append(_norm(predicate))
        ids = [r[0] for r in self.base._get_conn().execute(sql, params)]
        return [[self._row_to_fact(r) for r in self.base._get_conn().execute("SELECT * FROM facts WHERE conflict_set_id=?", (cid,))] for cid in ids]

    def get_fact_history(self, subject_or_id: str, predicate: Optional[str] = None) -> List[Fact]:
        c = self.base._get_conn()
        if predicate is None and subject_or_id.startswith("fact-"):
            root = self._get_fact(subject_or_id)
            if not root: return []
            subject_or_id, predicate = root.subject, root.predicate
        rows = c.execute("SELECT * FROM facts WHERE subject_norm=? AND predicate_norm=? ORDER BY version,created_at", (_norm(subject_or_id), _norm(predicate or ""))).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def retrieve_facts(self, query: str, *, as_of: Optional[float] = None, task_id: str = "", agent_id: str = "", top_k: int = 5, include_history: bool = False, include_conflicts: bool = False) -> List[FactSearchResult]:
        instant = self.clock() if as_of is None else as_of; rows = self.base._get_conn().execute("SELECT * FROM facts").fetchall(); qemb = self.base.embedding_model.embed(query); qt = _tokenize(query); results = []
        for row in rows:
            fact = self._row_to_fact(row)
            valid = fact.valid_from <= instant and (fact.valid_to is None or instant < fact.valid_to)
            if not include_history and not valid: continue
            # Explicit as-of queries may surface a version that is superseded now
            # but was the valid assertion at the requested instant.
            if as_of is None and not include_history and fact.status in {FactStatus.SUPERSEDED, FactStatus.INVALIDATED}: continue
            if fact.status == FactStatus.DISPUTED and not include_conflicts: continue
            text = f"{fact.subject} {fact.predicate} {fact.object}"; dense = _cosine(qemb, json.loads(row["embedding_json"])); sparse = _sparse_score(qt, _tokenize(text))
            temporal = 1.0 if valid else .15; evidences = self._evidence_for(fact.id); source = max([self.source_reliability.get(e.source_type, .5) for e in evidences] or [.5]); support = min(1.0, math.log2(len(evidences)+1)/2); scope = 1.0 if (task_id and fact.task_id == task_id) or (agent_id and fact.agent_id == agent_id) else (.55 if task_id or agent_id else .7); recency = 1.0/(1.0+max(0.0, instant-fact.updated_at)/86400.0); conflict = 1.0 if fact.status == FactStatus.DISPUTED else 0.0
            b = {"semantic": dense,"sparse":sparse,"temporal":temporal,"source":source,"evidence":support,"scope":scope,"recency":recency,"conflict_penalty":conflict}
            w=self.weights; score=w.semantic*dense+w.sparse*sparse+w.temporal*temporal+w.source*source+w.evidence*support+w.scope*scope+w.recency*recency-w.conflict_penalty*conflict
            results.append(FactSearchResult(fact, round(score,6), {k:round(v,6) for k,v in b.items()}, evidences))
        return sorted(results, key=lambda x:(x.score,x.fact.confidence,x.fact.updated_at), reverse=True)[:top_k]

    def extract_from_capsule(self, capsule: MessageCapsule, *, persist: bool = True) -> List[Fact]:
        if not (capsule.claim or capsule.constraint_delta or capsule.evidence): return []
        evidence = [TemporalEvidence(content=e.content or capsule.claim, source=e.source or e.uri or capsule.sender, source_type=e.metadata.get("source_type", capsule.provenance.source_type), confidence=e.confidence, observed_at=e.observed_at or capsule.created_at, task_id=capsule.provenance.task_id, agent_id=capsule.provenance.agent_id or capsule.sender, trace_id=capsule.provenance.trace_id, capsule_digest=capsule.digest, metadata=e.metadata) for e in capsule.evidence]
        if not evidence and capsule.claim: evidence=[TemporalEvidence(content=capsule.claim, source=capsule.sender, source_type=capsule.provenance.source_type, confidence=max(.1,1.0-capsule.uncertainty), observed_at=capsule.created_at, task_id=capsule.provenance.task_id, agent_id=capsule.provenance.agent_id or capsule.sender, trace_id=capsule.provenance.trace_id, capsule_digest=capsule.digest)]
        facts=[]
        if capsule.claim: facts.append(Fact(subject=capsule.subtask or capsule.goal or capsule.sender, predicate="claims", object=capsule.claim, confidence=max(.1,1-capsule.uncertainty), task_id=capsule.provenance.task_id, agent_id=capsule.sender, trace_id=capsule.provenance.trace_id))
        for key,value in capsule.constraint_delta.items(): facts.append(Fact(subject=capsule.goal or capsule.provenance.task_id or "task", predicate=f"constraint:{key}", object=value, confidence=max(.1,1-capsule.uncertainty), task_id=capsule.provenance.task_id, agent_id=capsule.sender, trace_id=capsule.provenance.trace_id))
        return [self.upsert_fact(f,evidence,replace=f.predicate.startswith("constraint:")) if persist else f for f in facts]

    def _active_for_sp(self, subject: str, predicate: str) -> List[Fact]:
        rows=self.base._get_conn().execute("SELECT * FROM facts WHERE subject_norm=? AND predicate_norm=? AND status IN ('active','disputed')",(_norm(subject),_norm(predicate))).fetchall(); return [self._row_to_fact(r) for r in rows]
    def _persist_fact(self, fact: Fact) -> None:
        c=self.base._get_conn(); text=f"{fact.subject} {fact.predicate} {fact.object}"; emb=self.base.embedding_model.embed(text)
        c.execute("INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(fact.id,fact.subject,_norm(fact.subject),fact.predicate,_norm(fact.predicate),json.dumps(fact.object,ensure_ascii=False,default=str),_norm_obj(fact.object),fact.valid_from,fact.valid_to,fact.status.value,fact.confidence,fact.version,fact.supersedes_id,fact.conflict_set_id,fact.task_id,fact.agent_id,fact.trace_id,fact.created_at,fact.updated_at,json.dumps(fact.metadata,ensure_ascii=False,default=str),json.dumps(emb)))
        c.execute("INSERT OR REPLACE INTO fact_versions VALUES (?,?,?,?)",(fact.id,fact.version,json.dumps(fact.to_dict(),ensure_ascii=False,default=str),fact.updated_at)); c.commit()
    def _link_evidence(self, fact: Fact, evidence: Sequence[TemporalEvidence]) -> None:
        c=self.base._get_conn()
        for e in evidence:
            c.execute("INSERT OR REPLACE INTO temporal_evidence VALUES (?,?,?,?,?,?,?,?,?,?,?)",(e.id,e.content,e.source,e.source_type,e.confidence,e.observed_at,e.task_id,e.agent_id,e.trace_id,e.capsule_digest,json.dumps(e.metadata,ensure_ascii=False,default=str))); c.execute("INSERT OR IGNORE INTO fact_evidence VALUES (?,?)",(fact.id,e.id))
            if e.id not in fact.evidence_ids: fact.evidence_ids.append(e.id)
        c.commit()
    def _persist_conflict(self,cid:str,fid:str,reason:str)->None: self.base._get_conn().execute("INSERT OR REPLACE INTO fact_conflicts VALUES (?,?,?,?)",(cid,fid,reason,self.clock())); self.base._get_conn().commit()
    def _get_fact(self,fid:str)->Optional[Fact]:
        r=self.base._get_conn().execute("SELECT * FROM facts WHERE id=?",(fid,)).fetchone(); return self._row_to_fact(r) if r else None
    def _evidence_for(self,fid:str)->List[TemporalEvidence]:
        rows=self.base._get_conn().execute("SELECT e.* FROM temporal_evidence e JOIN fact_evidence fe ON e.id=fe.evidence_id WHERE fe.fact_id=?",(fid,)).fetchall(); return [TemporalEvidence(id=r["id"],content=r["content"],source=r["source"],source_type=r["source_type"],confidence=r["confidence"],observed_at=r["observed_at"],task_id=r["task_id"] or "",agent_id=r["agent_id"] or "",trace_id=r["trace_id"] or "",capsule_digest=r["capsule_digest"] or "",metadata=json.loads(r["metadata_json"])) for r in rows]
    def _row_to_fact(self,r)->Fact:
        f=Fact(id=r["id"],subject=r["subject"],predicate=r["predicate"],object=json.loads(r["object_json"]),valid_from=r["valid_from"],valid_to=r["valid_to"],status=FactStatus(r["status"]),confidence=r["confidence"],version=r["version"],supersedes_id=r["supersedes_id"] or "",conflict_set_id=r["conflict_set_id"] or "",task_id=r["task_id"] or "",agent_id=r["agent_id"] or "",trace_id=r["trace_id"] or "",created_at=r["created_at"],updated_at=r["updated_at"],metadata=json.loads(r["metadata_json"])); f.evidence_ids=[x[0] for x in self.base._get_conn().execute("SELECT evidence_id FROM fact_evidence WHERE fact_id=?",(f.id,))]; return f


def _norm(v: Any)->str: return re.sub(r"\s+"," ",str(v).strip().lower())
def _norm_obj(v: Any)->str: return _norm(json.dumps(v,ensure_ascii=False,sort_keys=True,default=str))
def _overlap_interval(a0:float,a1:Optional[float],b0:float,b1:Optional[float])->bool: return a0 < (b1 if b1 is not None else float("inf")) and b0 < (a1 if a1 is not None else float("inf"))
