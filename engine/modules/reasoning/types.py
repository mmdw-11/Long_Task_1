from __future__ import annotations

import enum
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


class ConstraintKind(str, enum.Enum):
    PRECEDENCE="precedence"; MUTEX="mutex"; REQUIRED_ROLE="required_role"
    RESOURCE_CAPACITY="resource_capacity"; DEADLINE="deadline"
    HARD_POLICY="hard_policy"; REQUIRED_EVIDENCE="required_evidence"


class ValidationStatus(str, enum.Enum):
    SAT="sat"; UNSAT="unsat"; INVALID="invalid"


@dataclass
class ConstraintIR:
    kind: ConstraintKind
    id: str = field(default_factory=lambda:f"constraint-{uuid.uuid4().hex}")
    source: str = ""
    target: str = ""
    value: Any = None
    resource: str = ""
    hard: bool = True
    description: str = ""
    metadata: Dict[str,Any] = field(default_factory=dict)
    def to_dict(self)->Dict[str,Any]:
        d=asdict(self); d["kind"]=self.kind.value; return d
    @classmethod
    def from_dict(cls,d:Dict[str,Any])->"ConstraintIR":
        p={k:v for k,v in d.items() if k in cls.__dataclass_fields__}; p["kind"]=ConstraintKind(p.get("kind","hard_policy")); return cls(**p)


@dataclass
class SubtaskIR:
    id: str
    description: str
    depends_on: List[str]=field(default_factory=list)
    duration: int=1
    role: str=""
    resource_requirements: Dict[str,int]=field(default_factory=dict)
    preconditions: List[str]=field(default_factory=list)
    effects: List[str]=field(default_factory=list)
    acceptance_criteria: List[str]=field(default_factory=list)
    fallbacks: List[str]=field(default_factory=list)
    idempotency: str="idempotent"
    metadata: Dict[str,Any]=field(default_factory=dict)
    def to_dict(self)->Dict[str,Any]: return asdict(self)
    @classmethod
    def from_dict(cls,d:Dict[str,Any])->"SubtaskIR": return cls(**{k:v for k,v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PlanIR:
    goal: str
    subtasks: List[SubtaskIR]=field(default_factory=list)
    hard_constraints: List[ConstraintIR]=field(default_factory=list)
    soft_constraints: List[ConstraintIR]=field(default_factory=list)
    resources: Dict[str,int]=field(default_factory=dict)
    acceptance_criteria: List[str]=field(default_factory=list)
    risks: List[str]=field(default_factory=list)
    revision: int=0
    provenance: Dict[str,Any]=field(default_factory=dict)
    id: str=field(default_factory=lambda:f"plan-{uuid.uuid4().hex}")
    schema_version: str="1.0"
    created_at: float=field(default_factory=time.time)
    def to_dict(self)->Dict[str,Any]:
        d=asdict(self); d["subtasks"]=[x.to_dict() for x in self.subtasks]; d["hard_constraints"]=[x.to_dict() for x in self.hard_constraints]; d["soft_constraints"]=[x.to_dict() for x in self.soft_constraints]; return d
    @classmethod
    def from_dict(cls,d:Dict[str,Any])->"PlanIR":
        p={k:v for k,v in d.items() if k in cls.__dataclass_fields__}; p["subtasks"]=[SubtaskIR.from_dict(x) for x in p.get("subtasks",[])]; p["hard_constraints"]=[ConstraintIR.from_dict(x) for x in p.get("hard_constraints",[])]; p["soft_constraints"]=[ConstraintIR.from_dict(x) for x in p.get("soft_constraints",[])]; return cls(**p)


@dataclass
class ConstraintViolation:
    code: str; message: str; constraint_ids: List[str]=field(default_factory=list)
    subtask_ids: List[str]=field(default_factory=list); evidence: Dict[str,Any]=field(default_factory=dict)
    severity: str="error"
    def to_dict(self)->Dict[str,Any]: return asdict(self)


@dataclass
class Counterexample:
    violation_code: str; summary: str; witness: Dict[str,Any]=field(default_factory=dict)
    constraint_ids: List[str]=field(default_factory=list); minimal: bool=True
    def to_dict(self)->Dict[str,Any]: return asdict(self)


@dataclass
class ValidationResult:
    status: ValidationStatus; violations: List[ConstraintViolation]=field(default_factory=list)
    counterexamples: List[Counterexample]=field(default_factory=list); schedule: Dict[str,Dict[str,int]]=field(default_factory=dict)
    backend: str="deterministic"; elapsed_ms: float=0.0; metadata: Dict[str,Any]=field(default_factory=dict)
    @property
    def passed(self)->bool: return self.status==ValidationStatus.SAT
    def to_dict(self)->Dict[str,Any]: return {"status":self.status.value,"passed":self.passed,"violations":[x.to_dict() for x in self.violations],"counterexamples":[x.to_dict() for x in self.counterexamples],"schedule":self.schedule,"backend":self.backend,"elapsed_ms":self.elapsed_ms,"metadata":self.metadata}


@dataclass
class PlanRevision:
    before_revision: int; after_revision: int; counterexamples: List[Counterexample]
    changed_subtasks: List[str]=field(default_factory=list); timestamp: float=field(default_factory=time.time)
    def to_dict(self)->Dict[str,Any]: return {"before_revision":self.before_revision,"after_revision":self.after_revision,"counterexamples":[x.to_dict() for x in self.counterexamples],"changed_subtasks":self.changed_subtasks,"timestamp":self.timestamp}
