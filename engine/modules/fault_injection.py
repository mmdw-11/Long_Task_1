from __future__ import annotations
import asyncio, random
from dataclasses import dataclass,field
from typing import Any,Dict,List,Optional


class InjectedFault(RuntimeError):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.failure_kind = kind

@dataclass
class FaultSpec:
    kind:str; node:str="*"; step:Optional[int]=None; attempt:Optional[int]=None
    probability:float=1.0; once:bool=True; metadata:Dict[str,Any]=field(default_factory=dict)

class FaultInjector:
    """Deterministic chaos harness used by tests and controlled experiments."""
    def __init__(self,specs:List[FaultSpec],*,seed:int=42)->None:self.specs=specs;self.rng=random.Random(seed);self.fired=set()
    def match(self,*,node:str,step:int,attempt:int=1)->Optional[FaultSpec]:
        for i,s in enumerate(self.specs):
            if s.once and i in self.fired:continue
            if s.node not in {"*",node} or (s.step is not None and s.step!=step) or (s.attempt is not None and s.attempt!=attempt):continue
            if self.rng.random()<=s.probability:self.fired.add(i);return s
        return None
    async def inject(self,*,node:str,step:int,attempt:int=1,state:Optional[Dict[str,Any]]=None)->None:
        spec=self.match(node=node,step=step,attempt=attempt)
        if spec is None:return
        if spec.kind=="timeout":raise TimeoutError(spec.metadata.get("message","injected timeout"))
        mapping={
            "exception":"agent", "tool_error":"tool", "resource_unavailable":"resource_unavailable",
            "rate_limit":"resource_unavailable", "agent_unavailable":"role_unavailable",
            "capsule_drop":"message_missing", "evidence_removal":"evidence_missing",
            "constraint_conflict":"plan_conflict", "partial_result":"validation_failed",
            "schema_invalid":"validation_failed",
        }
        if spec.kind=="capsule_drop" and state is not None:state.pop("__capsule_context__",None)
        if spec.kind=="evidence_removal" and state is not None:state["__memory_context_items__"]=[]
        if spec.kind=="constraint_conflict" and state is not None:state["__injected_constraint_conflict__"]=True
        if spec.kind in mapping:
            raise InjectedFault(mapping[spec.kind], spec.metadata.get("message",f"injected {spec.kind}"))
