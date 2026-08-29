"""恢复策略模块：依据失败轨迹决定修复路径。

节点执行失败时，``RecoveryStrategy`` 基于失败轨迹（:class:`FailureTrace`）判断
是否激活修复路径，以及采取何种动作：重试、改道（reroute 到修复节点）、补偿或
中止。可用于替换框架默认「失败即抛出」的行为。

本文件仅提供接口与空实现桩（``NoOpRecoveryStrategy`` 返回 ABORT，维持现状）。
"""

from __future__ import annotations

import enum
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class RecoveryAction(str, enum.Enum):
    """恢复动作类型。"""

    RETRY = "retry"           # 重试当前失败节点
    REROUTE = "reroute"       # 改道到修复/兜底节点
    COMPENSATE = "compensate" # 执行补偿操作
    ABORT = "abort"           # 中止（维持抛错）
    FALLBACK_MODEL="fallback_model"; FALLBACK_TOOL="fallback_tool"
    FALLBACK_NODE="fallback_node"; RESUME_CHECKPOINT="resume_checkpoint"
    REPLAN="replan"; REDECOMPOSE="redecompose"; SIBLING_TAKEOVER="sibling_takeover"
    MIGRATE_RESOURCE="migrate_resource"; DEGRADE="degrade"
    HUMAN_REVIEW="human_review"


@dataclass
class RepairPlan:
    """修复计划：恢复动作 + 目标节点 + 理由。"""

    action: RecoveryAction
    targets: List[str] = field(default_factory=list)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def should_abort(self) -> bool:
        return self.action == RecoveryAction.ABORT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "targets": list(self.targets),
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass
class RecoveryPolicy:
    max_attempts: int=3; max_episode_actions: int=5; base_backoff_seconds: float=0.0
    jitter_seconds: float=0.0; resource_degradation_order: List[str]=field(default_factory=lambda:["cloud","edge","device"])
    human_review_for_non_idempotent: bool=True


@dataclass
class RecoveryAttempt:
    action: RecoveryAction; node: str; attempt: int; reason: str
    target: str=""; success: Optional[bool]=None; started_at: float=field(default_factory=time.time)
    finished_at: Optional[float]=None; metadata: Dict[str,Any]=field(default_factory=dict)
    def to_dict(self):return {"action":self.action.value,"node":self.node,"attempt":self.attempt,"reason":self.reason,"target":self.target,"success":self.success,"started_at":self.started_at,"finished_at":self.finished_at,"metadata":self.metadata}


@dataclass
class RecoveryEpisode:
    id: str; failure: Dict[str,Any]; attempts: List[RecoveryAttempt]=field(default_factory=list)
    status: str="active"; started_at: float=field(default_factory=time.time); finished_at: Optional[float]=None
    def to_dict(self):return {"id":self.id,"failure":self.failure,"attempts":[x.to_dict() for x in self.attempts],"status":self.status,"started_at":self.started_at,"finished_at":self.finished_at}


class RecoveryStrategy(ABC):
    """恢复策略接口。"""

    @abstractmethod
    def plan(
        self,
        failure_trace: Any,
        *,
        node: str,
        state: Dict[str, Any],
    ) -> RepairPlan:
        """依据失败轨迹给出修复计划。

        :param failure_trace: 失败轨迹（:class:`FailureTrace`）。
        :param node: 当前失败节点名。
        :param state: 当前状态快照。
        """
        raise NotImplementedError


class NoOpRecoveryStrategy(RecoveryStrategy):
    """空实现桩：恒返回 ABORT（等价于框架现有的失败即抛错行为）。"""

    def plan(
        self,
        failure_trace: Any,
        *,
        node: str,
        state: Dict[str, Any],
    ) -> RepairPlan:
        return RepairPlan(action=RecoveryAction.ABORT, reason="noop")


class PolicyRecoveryStrategy(RecoveryStrategy):
    """Explainable, bounded, side-effect-aware graph recovery policy."""
    def __init__(self,policy:Optional[RecoveryPolicy]=None,*,seed:int=42)->None:
        self.policy=policy or RecoveryPolicy(); self.rng=random.Random(seed)
    def plan(self,failure_trace:Any,*,node:str,state:Dict[str,Any])->RepairPlan:
        from ..failure import FailureKind
        context=dict(state.get("__failure_context__") or {}); kind=str(context.get("kind") or self._infer_kind(failure_trace,node)); attempts=len(failure_trace.by_node(node)) if hasattr(failure_trace,"by_node") else 1
        metadata=dict(context.get("metadata") or {}); side=str(context.get("side_effect_class") or metadata.get("idempotency") or "idempotent")
        if side=="non_idempotent" and self.policy.human_review_for_non_idempotent and not metadata.get("compensation_node"):
            return RepairPlan(RecoveryAction.HUMAN_REVIEW,reason="non-idempotent side effect cannot be replayed safely",metadata={"pause":True,"side_effect_class":side})
        if side=="compensatable" and metadata.get("compensation_node"):
            return RepairPlan(RecoveryAction.COMPENSATE,[str(metadata["compensation_node"])],"compensate before replay",{"then":"resume_checkpoint"})
        episodes=list(state.get("__recovery_trace__") or [])
        active_for_node=sum(1 for item in episodes if item.get("failure",{}).get("node")==node)
        if active_for_node>=self.policy.max_episode_actions:
            return RepairPlan(RecoveryAction.ABORT,reason="recovery episode action budget exhausted")
        if attempts<self.policy.max_attempts and kind in {FailureKind.TIMEOUT.value,FailureKind.AGENT.value} and side in {"idempotent","read_only"}:
            delay=self.policy.base_backoff_seconds*(2**max(0,attempts-1))+self.rng.random()*self.policy.jitter_seconds
            return RepairPlan(RecoveryAction.RETRY,[node],"bounded retry for transient failure",{"attempt":attempts+1,"backoff_seconds":delay})
        if kind==FailureKind.TOOL.value and metadata.get("fallback_tools"):
            return RepairPlan(RecoveryAction.FALLBACK_TOOL,[],"primary tool failed",{"fallback_tool":metadata["fallback_tools"][0]})
        if kind==FailureKind.AGENT.value and metadata.get("fallback_models"):
            return RepairPlan(RecoveryAction.FALLBACK_MODEL,[node],"primary model failed",{"fallback_model":metadata["fallback_models"][0]})
        if kind==FailureKind.RESOURCE_UNAVAILABLE.value:
            return RepairPlan(RecoveryAction.MIGRATE_RESOURCE,[node],"resource tier unavailable",{"degradation_order":self.policy.resource_degradation_order})
        if kind in {FailureKind.MESSAGE_MISSING.value,FailureKind.EVIDENCE_MISSING.value}:
            targets=list(metadata.get("evidence_repair_nodes") or metadata.get("fallback_nodes") or [])
            return RepairPlan(RecoveryAction.FALLBACK_NODE,targets,"restore missing message/evidence",{"wakeup_level":"recovery"}) if targets else RepairPlan(RecoveryAction.HUMAN_REVIEW,reason="required evidence unavailable",metadata={"pause":True})
        if kind in {FailureKind.PLAN_CONFLICT.value,FailureKind.STEP_FAILED.value,FailureKind.VALIDATION_FAILED.value}:
            targets=list(metadata.get("replan_nodes") or metadata.get("fallback_nodes") or [])
            return RepairPlan(RecoveryAction.REPLAN,targets,"verified plan must be revised from checkpoint",{"checkpoint_id":"latest_verified"})
        if kind==FailureKind.ROLE_UNAVAILABLE.value and metadata.get("standby_nodes"):
            return RepairPlan(RecoveryAction.SIBLING_TAKEOVER,[str(metadata["standby_nodes"][0])],"critical role unavailable")
        if metadata.get("fallback_nodes"):
            return RepairPlan(RecoveryAction.FALLBACK_NODE,[str(metadata["fallback_nodes"][0])],"configured fallback node")
        if metadata.get("degrade_contract"):
            return RepairPlan(RecoveryAction.DEGRADE,[node],"continue under explicit degradation contract",{"degrade_contract":metadata["degrade_contract"]})
        return RepairPlan(RecoveryAction.ABORT,reason="recovery budget exhausted or no safe action")
    @staticmethod
    def _infer_kind(trace,node):
        from ..failure import FailureKind
        recent=trace.by_node(node)[-1] if hasattr(trace,"by_node") and trace.by_node(node) else None
        return getattr(recent,"kind",FailureKind.AGENT.value)
