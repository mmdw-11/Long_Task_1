"""记忆模块：四级层级记忆的读写与检索。

``MemoryStore`` 抽象了一套四级层级记忆存储接口：

- **WORKING**（工作级）: 当前节点单次执行内的临时上下文，执行结束即可丢弃。
- **TASK**（任务级）: 一次图执行（run）内的跨节点共享记忆，run 结束后可归档或丢弃。
- **PROJECT**（项目级）: 同一编排定义（workflow）跨多次 run 的持久化记忆。
- **GLOBAL**（全局级）: 跨项目的全局知识，长期持久化。

层级从窄到宽排列：WORKING → TASK → PROJECT → GLOBAL。
检索时支持**级联查找**：从指定层级开始，逐级向上扩展搜索范围，直至收集到
足够结果或穷尽所有层级。
"""

# Types & enums
from ._types import (
    ArchiveExpandPolicy,
    MediaRoute,
    MemoryContext,
    MemoryItem,
    MemoryMedium,
    MemoryScope,
    MemoryUpdateAction,
    RetrievalMode,
    WakeupLevel,
    WakeupProfile,
    WakeupResult,
    wakeup_profile,
)

# Abstract base
from .base import MemoryStore, NoOpMemoryStore

# Embedding models
from .embedding import BGEM3EmbeddingModel, EmbeddingModel, HashingEmbeddingModel

# LLM judge
from .judge import MemoryLLMJudge, OpenAIMemoryJudge

# Store implementation
from .store import HybridTieredMemoryStore

__all__ = [
    # Types
    "ArchiveExpandPolicy",
    "MediaRoute",
    "MemoryContext",
    "MemoryItem",
    "MemoryMedium",
    "MemoryScope",
    "MemoryUpdateAction",
    "RetrievalMode",
    "WakeupLevel",
    "WakeupProfile",
    "WakeupResult",
    "wakeup_profile",
    # Base
    "MemoryStore",
    "NoOpMemoryStore",
    # Embedding
    "BGEM3EmbeddingModel",
    "EmbeddingModel",
    "HashingEmbeddingModel",
    # Judge
    "MemoryLLMJudge",
    "OpenAIMemoryJudge",
    # Store
    "HybridTieredMemoryStore",
]
