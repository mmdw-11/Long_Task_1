"""可信工作区策略。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ._types import ResourceTier, SENSITIVITY_SCORE, SensitivityLevel


@dataclass
class TrustedWorkspacePolicy:
    """可信工作区策略。

    ``trusted_tiers`` 表示被视为可信工作区的推理位置。敏感等级达到
    ``sensitive_threshold`` 后，调度器优先限制在可信层级内；如果必须选择非可信
    层级，则需要人工审计批准。
    """

    trusted_tiers: List[ResourceTier] = field(
        default_factory=lambda: [ResourceTier.DEVICE, ResourceTier.EDGE]
    )
    sensitive_threshold: SensitivityLevel = SensitivityLevel.CONFIDENTIAL
    allow_untrusted_with_review: bool = True

    def is_sensitive(self, level: SensitivityLevel) -> bool:
        return SENSITIVITY_SCORE[level] >= SENSITIVITY_SCORE[self.sensitive_threshold]

    def is_trusted(self, tier: ResourceTier) -> bool:
        return tier in self.trusted_tiers
