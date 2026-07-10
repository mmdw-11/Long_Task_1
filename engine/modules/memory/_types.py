"""记忆模块类型定义。

枚举、数据类与 wakeup_profile 工厂函数。
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class MemoryScope(str, enum.Enum):
    """记忆作用域（层级从窄到宽）。"""

    WORKING = "working"    # 工作级：当前节点执行内
    TASK = "task"          # 任务级：一次图执行（run）内
    PROJECT = "project"    # 项目级：同一编排定义跨 run
    GLOBAL = "global"      # 全局级：跨项目

    @staticmethod
    def hierarchy() -> List["MemoryScope"]:
        """返回从窄到宽的层级顺序。"""
        return [
            MemoryScope.WORKING,
            MemoryScope.TASK,
            MemoryScope.PROJECT,
            MemoryScope.GLOBAL,
        ]


class MemoryMedium(str, enum.Enum):
    """Storage media used inside each memory scope."""

    HOT = "hot"                # In-process dict / LRU.
    WARM = "warm"              # Redis, optionally shared and TTL-based.
    COLD = "cold"              # Markdown / JSONL audit archive.
    STRUCTURED = "structured"  # SQLite / PostgreSQL row store.
    SEMANTIC = "semantic"      # Vector index, currently SQLite embedding rows.
    GRAPH = "graph"            # Zep / Graphiti style graph memory.


@dataclass
class MediaRoute:
    """Resolved media destinations for a memory item."""

    media: List[MemoryMedium]
    ttl_seconds: Optional[int] = None

    def contains(self, medium: MemoryMedium) -> bool:
        return medium in self.media


class RetrievalMode(str, enum.Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    HYBRID_TEMPORAL = "hybrid_temporal"


class ArchiveExpandPolicy(str, enum.Enum):
    NONE = "none"
    ON_DEMAND = "on_demand"
    PRE_EXPAND_TOP = "pre_expand_top"
    EXPAND_ALL = "expand_all"


class WakeupLevel(int, enum.Enum):
    SILENT = 0
    STANDARD = 1
    DEEP = 2
    RECOVERY = 3


class MemoryUpdateAction(str, enum.Enum):
    """LLM 判断后的记忆更新动作。

    仿照 mem0 的设计，由 LLM 决定新记忆与已有记忆之间的关系。
    """

    ADD = "add"        # 新记忆是全新信息，直接添加
    UPDATE = "update"  # 新记忆更新/补充了旧记忆，替换旧记忆
    DELETE = "delete"  # 新记忆与旧记忆重复或已过时，丢弃新记忆
    NOOP = "noop"      # 不做任何操作


@dataclass(frozen=True)
class WakeupProfile:
    """Context wakeup profile controlling retrieval and archive expansion."""

    level: WakeupLevel
    name: str
    scopes: List[MemoryScope]
    top_k: int
    retrieval_mode: RetrievalMode
    expand_policy: ArchiveExpandPolicy
    temporal_weight: float = 0.0
    pre_expand_limit: int = 0


@dataclass
class WakeupResult:
    profile: WakeupProfile
    items: List["MemoryItem"]
    context_text: str


def wakeup_profile(level: int | WakeupLevel | str = WakeupLevel.STANDARD) -> WakeupProfile:
    """Return a built-in wakeup profile."""
    if isinstance(level, str):
        normalized = level.lower()
        aliases = {
            "0": WakeupLevel.SILENT,
            "silent": WakeupLevel.SILENT,
            "light": WakeupLevel.SILENT,
            "1": WakeupLevel.STANDARD,
            "standard": WakeupLevel.STANDARD,
            "default": WakeupLevel.STANDARD,
            "2": WakeupLevel.DEEP,
            "deep": WakeupLevel.DEEP,
            "3": WakeupLevel.RECOVERY,
            "recovery": WakeupLevel.RECOVERY,
        }
        level = aliases.get(normalized, WakeupLevel.STANDARD)
    level = WakeupLevel(level)
    profiles = {
        WakeupLevel.SILENT: WakeupProfile(
            level=WakeupLevel.SILENT,
            name="silent_light",
            scopes=[MemoryScope.WORKING, MemoryScope.TASK],
            top_k=3,
            retrieval_mode=RetrievalMode.DENSE,
            expand_policy=ArchiveExpandPolicy.NONE,
        ),
        WakeupLevel.STANDARD: WakeupProfile(
            level=WakeupLevel.STANDARD,
            name="standard_business",
            scopes=MemoryScope.hierarchy(),
            top_k=5,
            retrieval_mode=RetrievalMode.HYBRID,
            expand_policy=ArchiveExpandPolicy.ON_DEMAND,
        ),
        WakeupLevel.DEEP: WakeupProfile(
            level=WakeupLevel.DEEP,
            name="deep_task",
            scopes=MemoryScope.hierarchy(),
            top_k=8,
            retrieval_mode=RetrievalMode.HYBRID,
            expand_policy=ArchiveExpandPolicy.PRE_EXPAND_TOP,
            pre_expand_limit=2,
        ),
        WakeupLevel.RECOVERY: WakeupProfile(
            level=WakeupLevel.RECOVERY,
            name="recovery_traceback",
            scopes=MemoryScope.hierarchy(),
            top_k=10,
            retrieval_mode=RetrievalMode.HYBRID_TEMPORAL,
            expand_policy=ArchiveExpandPolicy.EXPAND_ALL,
            temporal_weight=0.15,
        ),
    }
    return profiles[level]


@dataclass
class MemoryContext:
    """记忆上下文：标识当前操作所属的各级作用域 id。

    未设置的层级视为 ``None``，表示该层级不参与精确匹配。
    级联检索时，从 ``narrowest`` 指定的层级开始逐级向上。

    :param working_id:  当前节点执行标识（如 ``"step_3:researcher"``）。
    :param task_id:     当前图执行标识（如一次 run 的 uuid）。
    :param project_id:  当前编排/项目标识（如 orchestrator 名称或 id）。
    :param global_id:   全局命名空间（通常为固定值，如 ``"default"``）。
    """

    working_id: Optional[str] = None
    task_id: Optional[str] = None
    project_id: Optional[str] = None
    global_id: Optional[str] = "default"

    def id_for(self, scope: MemoryScope) -> Optional[str]:
        """返回指定层级对应的 id。"""
        return {
            MemoryScope.WORKING: self.working_id,
            MemoryScope.TASK: self.task_id,
            MemoryScope.PROJECT: self.project_id,
            MemoryScope.GLOBAL: self.global_id,
        }.get(scope)


@dataclass
class MemoryItem:
    """一条记忆。

    :param content:  记忆内容（任意可序列化结构）。
    :param scope:    该条记忆写入的作用域层级。
    :param scope_id: 该条记忆绑定的作用域实例 id（由 MemoryContext 提供）。
    :param tags:     标签列表，用于过滤与分类检索。
    :param metadata: 附加元信息。
    :param ts:       写入时间戳。
    """

    content: Any
    scope: MemoryScope = MemoryScope.TASK
    scope_id: Optional[str] = None
    id: str = field(default_factory=lambda: f"mem-{uuid.uuid4().hex}")
    summary: str = ""
    raw_ref: Optional[str] = None
    modality: str = "text"
    embedding: Optional[List[float]] = None
    importance: float = 0.0
    token_count: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "summary": self.summary,
            "raw_ref": self.raw_ref,
            "scope": self.scope.value,
            "scope_id": self.scope_id,
            "modality": self.modality,
            "embedding": self.embedding,
            "importance": self.importance,
            "token_count": self.token_count,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "ts": self.ts,
        }
