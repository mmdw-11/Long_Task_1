"""恢复策略模块：依据失败轨迹决定修复路径。

节点执行失败时，``RecoveryStrategy`` 基于失败轨迹（:class:`FailureTrace`）判断
是否激活修复路径，以及采取何种动作：重试、改道（reroute 到修复节点）、补偿或
中止。可用于替换框架默认「失败即抛出」的行为。

本文件仅提供接口与空实现桩（``NoOpRecoveryStrategy`` 返回 ABORT，维持现状）。
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class RecoveryAction(str, enum.Enum):
    """恢复动作类型。"""

    RETRY = "retry"           # 重试当前失败节点
    REROUTE = "reroute"       # 改道到修复/兜底节点
    COMPENSATE = "compensate" # 执行补偿操作
    ABORT = "abort"           # 中止（维持抛错）


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
