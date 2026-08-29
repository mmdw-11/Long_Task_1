from __future__ import annotations

import time
from typing import Dict, List, Optional, Set, Tuple
from .types import ConstraintKind, ConstraintViolation, Counterexample, PlanIR, ValidationResult, ValidationStatus


class SymbolicPlanValidator:
    """Deterministic verifier with optional Z3 scheduling/unsat-core backend."""
    def __init__(self,backend:str="auto",*,available_roles:Optional[Set[str]]=None,system_hard_constraints:Optional[List[str]]=None)->None:
        self.backend=backend; self.available_roles=set(available_roles or []); self.system_hard_constraints=list(system_hard_constraints or [])

    def validate(self,plan:PlanIR)->ValidationResult:
        started=time.perf_counter(); violations=self._structural(plan)
        if violations: return self._result(ValidationStatus.INVALID,violations,"deterministic",started)
        use_z3=self.backend in {"auto","z3"} and self._z3_available()
        if use_z3:
            result=self._validate_z3(plan); result.elapsed_ms=(time.perf_counter()-started)*1000; return result
        violations=self._constraints(plan)
        return self._result(ValidationStatus.UNSAT if violations else ValidationStatus.SAT,violations,"deterministic",started,schedule=self._earliest_schedule(plan) if not violations else {})

    def _structural(self,plan:PlanIR)->List[ConstraintViolation]:
        out=[]; ids=[x.id for x in plan.subtasks]; known=set(ids)
        if not plan.goal.strip(): out.append(ConstraintViolation("missing_goal","plan goal is empty"))
        if len(ids)!=len(known): out.append(ConstraintViolation("duplicate_subtask","subtask ids must be unique"))
        for task in plan.subtasks:
            missing=[x for x in task.depends_on if x not in known]
            if missing: out.append(ConstraintViolation("missing_dependency",f"{task.id} depends on unknown tasks",subtask_ids=[task.id,*missing],evidence={"missing":missing}))
            if task.id in task.depends_on: out.append(ConstraintViolation("self_dependency",f"{task.id} depends on itself",subtask_ids=[task.id]))
            if task.duration<0: out.append(ConstraintViolation("negative_duration",f"{task.id} has negative duration",subtask_ids=[task.id]))
        cycle=self._cycle(plan)
        if cycle: out.append(ConstraintViolation("dependency_cycle","dependency graph contains a cycle",subtask_ids=cycle,evidence={"cycle":cycle}))
        return out

    def _constraints(self,plan:PlanIR)->List[ConstraintViolation]:
        out=[]; tasks={x.id:x for x in plan.subtasks}; schedule=self._earliest_schedule(plan)
        for task in plan.subtasks:
            if self.available_roles and task.role and task.role not in self.available_roles: out.append(ConstraintViolation("role_unavailable",f"role {task.role} required by {task.id} is unavailable",subtask_ids=[task.id],evidence={"role":task.role}))
            for resource,demand in task.resource_requirements.items():
                if demand>plan.resources.get(resource,0): out.append(ConstraintViolation("resource_capacity",f"{task.id} requires {demand} {resource}, capacity is {plan.resources.get(resource,0)}",subtask_ids=[task.id],evidence={"resource":resource,"demand":demand,"capacity":plan.resources.get(resource,0)}))
        for c in plan.hard_constraints:
            if c.kind==ConstraintKind.PRECEDENCE and c.source in schedule and c.target in schedule and schedule[c.source]["end"]>schedule[c.target]["start"]: out.append(ConstraintViolation("precedence_conflict",c.description or f"{c.source} must precede {c.target}",[c.id],[c.source,c.target]))
            elif c.kind==ConstraintKind.MUTEX and c.source in schedule and c.target in schedule and self._overlap(schedule[c.source],schedule[c.target]): out.append(ConstraintViolation("mutex_conflict",c.description or f"{c.source} and {c.target} overlap",[c.id],[c.source,c.target]))
            elif c.kind==ConstraintKind.REQUIRED_ROLE and c.source in tasks and str(c.value)!=tasks[c.source].role: out.append(ConstraintViolation("required_role",c.description or f"{c.source} must use role {c.value}",[c.id],[c.source]))
            elif c.kind==ConstraintKind.DEADLINE and c.source in schedule and schedule[c.source]["end"]>int(c.value): out.append(ConstraintViolation("deadline_missed",c.description or f"{c.source} ends after deadline {c.value}",[c.id],[c.source],{"end":schedule[c.source]["end"],"deadline":c.value}))
            elif c.kind==ConstraintKind.HARD_POLICY and str(c.value) in {"false","False","0"}: out.append(ConstraintViolation("hard_policy",c.description or "hard policy explicitly violated",[c.id]))
            elif c.kind==ConstraintKind.REQUIRED_EVIDENCE and c.source in tasks and str(c.value) not in tasks[c.source].acceptance_criteria: out.append(ConstraintViolation("required_evidence",c.description or f"{c.source} lacks evidence criterion {c.value}",[c.id],[c.source]))
        return out

    def _validate_z3(self,plan:PlanIR)->ValidationResult:
        import z3  # type: ignore
        solver=z3.Solver(); solver.set(unsat_core=True); starts={}; ends={}; named={}
        horizon=max(1,sum(max(0,t.duration) for t in plan.subtasks)+max([int(c.value) for c in plan.hard_constraints if c.kind==ConstraintKind.DEADLINE and str(c.value).isdigit()] or [0]))
        for t in plan.subtasks:
            starts[t.id]=z3.Int(f"start_{t.id}"); ends[t.id]=z3.Int(f"end_{t.id}"); solver.add(starts[t.id]>=0,ends[t.id]==starts[t.id]+t.duration,ends[t.id]<=horizon)
        def track(expr,cid):
            lit=z3.Bool(f"constraint_{cid.replace('-','_')}"); named[str(lit)]=cid; solver.assert_and_track(expr,lit)
        for t in plan.subtasks:
            for dep in t.depends_on: track(ends[dep]<=starts[t.id],f"dep:{dep}:{t.id}")
        for c in plan.hard_constraints:
            if c.kind==ConstraintKind.PRECEDENCE and c.source in ends and c.target in starts: track(ends[c.source]<=starts[c.target],c.id)
            elif c.kind==ConstraintKind.MUTEX and c.source in ends and c.target in starts: track(z3.Or(ends[c.source]<=starts[c.target],ends[c.target]<=starts[c.source]),c.id)
            elif c.kind==ConstraintKind.DEADLINE and c.source in ends: track(ends[c.source]<=int(c.value),c.id)
        status=solver.check()
        if status==z3.sat:
            model=solver.model(); schedule={k:{"start":model.eval(starts[k]).as_long(),"end":model.eval(ends[k]).as_long()} for k in starts}; return ValidationResult(ValidationStatus.SAT,schedule=schedule,backend="z3")
        core=[named.get(str(x),str(x)) for x in solver.unsat_core()]; v=ConstraintViolation("unsat_core","symbolic constraints are unsatisfiable",core,evidence={"unsat_core":core}); return ValidationResult(ValidationStatus.UNSAT,[v],[Counterexample("unsat_core","remove or revise one constraint in the unsat core",{"unsat_core":core},core)],backend="z3")

    def _result(self,status,violations,backend,started,schedule=None):
        examples=[Counterexample(v.code,v.message,v.evidence,v.constraint_ids,True) for v in violations]; return ValidationResult(status,violations,examples,schedule or {},backend,(time.perf_counter()-started)*1000)
    def _cycle(self,plan):
        graph={t.id:list(t.depends_on) for t in plan.subtasks}; visiting=set(); done=set(); path=[]
        def dfs(n):
            if n in visiting: return path[path.index(n):]+[n]
            if n in done:return []
            visiting.add(n); path.append(n)
            for x in graph.get(n,[]):
                found=dfs(x)
                if found:return found
            path.pop(); visiting.remove(n); done.add(n); return []
        for n in graph:
            found=dfs(n)
            if found:return found
        return []
    def _earliest_schedule(self,plan):
        tasks={t.id:t for t in plan.subtasks}; memo={}
        def end(tid):
            if tid in memo:return memo[tid]["end"]
            start=max([end(d) for d in tasks[tid].depends_on] or [0]); memo[tid]={"start":start,"end":start+tasks[tid].duration}; return memo[tid]["end"]
        for tid in tasks:end(tid)
        return memo
    @staticmethod
    def _overlap(a,b):return a["start"]<b["end"] and b["start"]<a["end"]
    @staticmethod
    def _z3_available():
        try: import z3; return True
        except ImportError:return False
