"""记忆存储抽象接口与空实现桩。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from ._types import MemoryContext, MemoryItem, MemoryScope


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

    def delete(self, memory_id: str) -> None:
        """删除单条记忆。

        :param memory_id: 要删除的记忆 ID。
        """
        raise NotImplementedError

    def close(self) -> None:
        """释放后端持有的资源（如数据库连接）。

        默认实现为空操作；持有持久连接的后端应覆盖此方法。
        """


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

    def delete(self, memory_id: str) -> None:
        return None
