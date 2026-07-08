"""记忆模块：四级层级记忆的读写与检索。

``MemoryStore`` 抽象了一套四级层级记忆存储接口：

- **WORKING**（工作级）: 当前节点单次执行内的临时上下文，执行结束即可丢弃。
- **TASK**（任务级）: 一次图执行（run）内的跨节点共享记忆，run 结束后可归档或丢弃。
- **PROJECT**（项目级）: 同一编排定义（workflow）跨多次 run 的持久化记忆。
- **GLOBAL**（全局级）: 跨项目的全局知识，长期持久化。

层级从窄到宽排列：WORKING → TASK → PROJECT → GLOBAL。
检索时支持**级联查找**：从指定层级开始，逐级向上扩展搜索范围，直至收集到
足够结果或穷尽所有层级。

本文件仅提供接口与空实现桩（``NoOpMemoryStore`` 写入丢弃、读取返回空）。
"""

from __future__ import annotations

import enum
import time
from abc import ABC, abstractmethod
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
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "scope": self.scope.value,
            "scope_id": self.scope_id,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "ts": self.ts,
        }


class MemoryStore(ABC):
    """四级层级记忆存储接口。

    实现者应按 ``MemoryScope`` 的四个层级分别维护存储（可以是不同的后端），
    并支持级联检索。
    """

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #
    @abstractmethod
    def write(self, item: MemoryItem, *, context: Optional[MemoryContext] = None) -> None:
        """写入一条记忆。

        :param item:    待写入的记忆条目。
        :param context: 可选的上下文；若提供，可用 ``context.id_for(item.scope)``
            自动填充 ``item.scope_id``。
        """
        raise NotImplementedError

    def append(
        self,
        content: Any,
        scope: MemoryScope = MemoryScope.TASK,
        *,
        context: Optional[MemoryContext] = None,
        tags: Optional[List[str]] = None,
        **metadata: Any,
    ) -> MemoryItem:
        """便捷写入：从内容构造 ``MemoryItem`` 并写入。"""
        scope_id = context.id_for(scope) if context else None
        item = MemoryItem(
            content=content,
            scope=scope,
            scope_id=scope_id,
            tags=list(tags or []),
            metadata=dict(metadata),
        )
        self.write(item, context=context)
        return item

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    @abstractmethod
    def read(
        self,
        query: str,
        *,
        scope: Optional[MemoryScope] = None,
        context: Optional[MemoryContext] = None,
        top_k: int = 5,
    ) -> List[MemoryItem]:
        """检索相关记忆。

        :param query:   检索关键词 / 查询条件。
        :param scope:   限定作用域层级；为 None 时等价于 ``MemoryScope.TASK``。
        :param context: 可选上下文，用于按 scope_id 精确过滤。
        :param top_k:   最多返回条数。
        :return: 匹配的记忆列表（按相关性 / 时间排序）。
        """
        raise NotImplementedError

    def cascade_read(
        self,
        query: str,
        *,
        narrowest: MemoryScope = MemoryScope.WORKING,
        context: Optional[MemoryContext] = None,
        top_k: int = 5,
    ) -> List[MemoryItem]:
        """级联检索：从 ``narrowest`` 层级开始逐级向上查找，直至收集到
        足够结果或穷尽 GLOBAL。

        默认实现依次调用 :meth:`read` 并合并结果；子类可覆盖以优化性能。

        :param narrowest: 起始（最窄）层级。
        :param context:   可选上下文。
        :param top_k:     总共最多返回条数。
        """
        hierarchy = MemoryScope.hierarchy()
        start_idx = hierarchy.index(narrowest)
        collected: List[MemoryItem] = []
        for scope in hierarchy[start_idx:]:
            remaining = top_k - len(collected)
            if remaining <= 0:
                break
            items = self.read(query, scope=scope, context=context, top_k=remaining)
            collected.extend(items)
        return collected

    # ------------------------------------------------------------------ #
    # 清空
    # ------------------------------------------------------------------ #
    @abstractmethod
    def clear(
        self,
        scope: Optional[MemoryScope] = None,
        *,
        context: Optional[MemoryContext] = None,
    ) -> None:
        """清空记忆。

        :param scope:   限定层级；None 表示清空所有层级。
        :param context: 可选上下文；若提供且 scope 不为 None，则只清空该
            scope_id 对应的记忆（不影响同层级其他实例）。
        """
        raise NotImplementedError


class NoOpMemoryStore(MemoryStore):
    """空实现桩：写入丢弃、读取返回空列表。"""

    def write(self, item: MemoryItem, *, context: Optional[MemoryContext] = None) -> None:
        return None

    def read(
        self,
        query: str,
        *,
        scope: Optional[MemoryScope] = None,
        context: Optional[MemoryContext] = None,
        top_k: int = 5,
    ) -> List[MemoryItem]:
        return []

    def clear(
        self,
        scope: Optional[MemoryScope] = None,
        *,
        context: Optional[MemoryContext] = None,
    ) -> None:
        return None
