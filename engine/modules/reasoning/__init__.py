"""Typed neuro-symbolic planning and counterexample-guided refinement."""
PLAN_IR_KEY = "__plan_ir__"
PLAN_VALIDATION_KEY = "__plan_validation__"
PLAN_REVISIONS_KEY = "__plan_revisions__"
REASONING_CONTEXT_KEY = "__reasoning_context__"
COMPLETED_SUBTASKS_KEY = "__completed_subtasks__"
from .types import (
    ConstraintIR, ConstraintKind, ConstraintViolation, Counterexample,
    PlanIR, PlanRevision, SubtaskIR, ValidationResult, ValidationStatus,
)
from .validator import SymbolicPlanValidator
from .reasoner import (
    DeterministicPlanGenerator, DeterministicPlanRepairer, LLMPlanGenerator,
    NeuroSymbolicReasoner, PlanGenerator, PlanRepairer, ReasoningOutcome,
)

__all__ = [
    "ConstraintIR", "ConstraintKind", "ConstraintViolation", "Counterexample",
    "DeterministicPlanGenerator", "DeterministicPlanRepairer", "LLMPlanGenerator",
    "NeuroSymbolicReasoner", "PlanGenerator", "PlanIR", "PlanRepairer",
    "PlanRevision", "ReasoningOutcome", "SubtaskIR", "SymbolicPlanValidator",
    "ValidationResult", "ValidationStatus",
    "PLAN_IR_KEY", "PLAN_VALIDATION_KEY", "PLAN_REVISIONS_KEY",
    "REASONING_CONTEXT_KEY",
    "COMPLETED_SUBTASKS_KEY",
]
