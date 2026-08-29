from __future__ import annotations

import copy, json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from ..communication import MessageCapsule, Provenance
from .types import Counterexample, PlanIR, PlanRevision, SubtaskIR, ValidationResult
from .validator import SymbolicPlanValidator


class PlanGenerator(ABC):
    @abstractmethod
    def generate(self,goal:str,context:Optional[Dict[str,Any]]=None)->PlanIR: ...
class PlanRepairer(ABC):
    @abstractmethod
    def revise(self,plan:PlanIR,counterexamples:List[Counterexample])->PlanIR: ...

class DeterministicPlanGenerator(PlanGenerator):
    def generate(self,goal,context=None):
        raw=(context or {}).get("subtasks") or ["analyze","execute","verify"]
        tasks=[SubtaskIR(id=f"step-{i+1}",description=str(x),depends_on=[] if i==0 else [f"step-{i}"]) for i,x in enumerate(raw)]
        return PlanIR(goal=goal,subtasks=tasks,resources=dict((context or {}).get("resources") or {}),provenance={"generator":"deterministic"})

class DeterministicPlanRepairer(PlanRepairer):
    """Repairs structural counterexamples; useful as an offline control."""
    def revise(self,plan,counterexamples):
        out=copy.deepcopy(plan); out.revision+=1; codes={x.violation_code for x in counterexamples}
        if "dependency_cycle" in codes:
            known=set()
            for task in out.subtasks: task.depends_on=[x for x in task.depends_on if x in known]; known.add(task.id)
        if "missing_dependency" in codes:
            ids={x.id for x in out.subtasks}
            for task in out.subtasks: task.depends_on=[x for x in task.depends_on if x in ids]
        return out

class LLMPlanGenerator(PlanGenerator):
    """Strict JSON adapter around an existing callable model/executor."""
    def __init__(self,invoke:Callable[[str],str])->None:self.invoke=invoke
    def generate(self,goal,context=None):
        prompt="Return only PlanIR JSON with goal, subtasks, hard_constraints, soft_constraints, resources, acceptance_criteria and risks.\nGoal: "+goal+"\nContext: "+json.dumps(context or {},ensure_ascii=False,default=str)
        raw=self.invoke(prompt).strip().removeprefix("```json").removesuffix("```").strip(); parsed=json.loads(raw)
        if not isinstance(parsed,dict):raise ValueError("LLM plan must be a JSON object")
        parsed.setdefault("goal",goal); return PlanIR.from_dict(parsed)

@dataclass
class ReasoningOutcome:
    plan: PlanIR; validation: ValidationResult; revisions: List[PlanRevision]=field(default_factory=list)
    feedback_capsules: List[MessageCapsule]=field(default_factory=list)
    @property
    def verified(self):return self.validation.passed
    def to_dict(self):return {"plan":self.plan.to_dict(),"validation":self.validation.to_dict(),"verified":self.verified,"revisions":[x.to_dict() for x in self.revisions],"feedback_capsules":[x.to_dict() for x in self.feedback_capsules]}

class NeuroSymbolicReasoner:
    def __init__(self,generator:PlanGenerator,validator:SymbolicPlanValidator,repairer:PlanRepairer,*,max_revisions:int=2)->None:self.generator=generator;self.validator=validator;self.repairer=repairer;self.max_revisions=max(0,max_revisions)
    def reason(self,goal:str,context:Optional[Dict[str,Any]]=None)->ReasoningOutcome:
        plan=self.generator.generate(goal,context); revisions=[]; capsules=[]
        for _ in range(self.max_revisions+1):
            validation=self.validator.validate(plan)
            if validation.passed:return ReasoningOutcome(plan,validation,revisions,capsules)
            capsule=MessageCapsule(sender="symbolic_validator",recipients=["planner"],goal=goal,subtask="repair_plan",claim="plan rejected by symbolic validation",evidence=[],constraint_delta={"counterexamples":[x.to_dict() for x in validation.counterexamples]},next_action="revise PlanIR and resubmit",uncertainty=0.0,provenance=Provenance(source_type="symbolic_solver"),metadata={"conflict":True,"validation_backend":validation.backend})
            capsules.append(capsule)
            if len(revisions)>=self.max_revisions:break
            before=plan; plan=self.repairer.revise(plan,validation.counterexamples)
            if plan.revision<=before.revision:plan.revision=before.revision+1
            revisions.append(PlanRevision(before.revision,plan.revision,validation.counterexamples,[x.id for x in plan.subtasks]))
        return ReasoningOutcome(plan,validation,revisions,capsules)
