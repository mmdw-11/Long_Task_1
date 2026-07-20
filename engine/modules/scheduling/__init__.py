"""端-边-云异构资源自适应调度模块。

模块拆分：
- ``_types``: 枚举、画像、资源画像、分配结果等数据结构。
- ``gate``: 任务门控与画像构建接口，默认实现为本地启发式规则。
- ``policy``: 可信工作区与人工审计策略。
- ``base``: 调度器抽象接口与 NoOp 实现。
- ``scheduler``: 默认自适应调度器。
"""

from ._types import (
    ModelSplitStep,
    RealtimeRequirement,
    ResourceAllocation,
    ResourceProfile,
    ResourceRequest,
    ResourceStatus,
    ResourceTier,
    SchedulingDecision,
    SchedulingTrace,
    SensitivityLevel,
    TaskComplexity,
    TaskProfile,
)
from .base import NoOpResourceScheduler, ResourceScheduler
from .gate import HeuristicTaskGate, OpenAITaskGate, TaskGate
from .learning import (
    BinaryTextRouterModel,
    BgeM3Encoder,
    benchmark_router,
    CascadeRouteDecision,
    evaluate_gate,
    FallbackCascadeTeacher,
    LearnedTaskGate,
    PseudoCascadeTeacher,
    RealCascadeTeacher,
    RouteDataset,
    RouteExample,
    RouteTeacher,
    build_route_dataset,
    build_route_dataset_from_requests,
    default_training_texts,
    render_experiment_report,
    save_experiment_report,
    train_router,
)
from .advanced_training import (
    AdvancedTrainingResult,
    TransformerTextRouter,
    benchmark_transformer_router,
    render_extended_experiment_report,
    save_extended_experiment_report,
    train_transformer_router,
)
from .monitor import (
    MutableResourceMonitor,
    NoOpResourceMonitor,
    ResourceMonitor,
    StaticResourceMonitor,
    SystemResourceMonitor,
)
from .policy import TrustedWorkspacePolicy
from .production import load_production_gate, resolve_router_path
from .scheduler import AdaptiveResourceScheduler

__all__ = [
    "AdaptiveResourceScheduler",
    "AdvancedTrainingResult",
    "HeuristicTaskGate",
    "BinaryTextRouterModel",
    "BgeM3Encoder",
    "benchmark_router",
    "benchmark_transformer_router",
    "CascadeRouteDecision",
    "FallbackCascadeTeacher",
    "LearnedTaskGate",
    "ModelSplitStep",
    "MutableResourceMonitor",
    "NoOpResourceScheduler",
    "NoOpResourceMonitor",
    "OpenAITaskGate",
    "RealtimeRequirement",
    "ResourceAllocation",
    "ResourceProfile",
    "ResourceRequest",
    "ResourceScheduler",
    "ResourceStatus",
    "ResourceTier",
    "ResourceMonitor",
    "SchedulingDecision",
    "SchedulingTrace",
    "SensitivityLevel",
    "TaskComplexity",
    "TaskGate",
    "TaskProfile",
    "TransformerTextRouter",
    "RouteDataset",
    "RouteExample",
    "RouteTeacher",
    "FallbackCascadeTeacher",
    "PseudoCascadeTeacher",
    "RealCascadeTeacher",
    "build_route_dataset",
    "build_route_dataset_from_requests",
    "default_training_texts",
    "evaluate_gate",
    "render_experiment_report",
    "render_extended_experiment_report",
    "resolve_router_path",
    "save_experiment_report",
    "save_extended_experiment_report",
    "load_production_gate",
    "train_router",
    "train_transformer_router",
    "StaticResourceMonitor",
    "SystemResourceMonitor",
    "TrustedWorkspacePolicy",
]
