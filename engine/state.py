"""图状态与归并器（reducer）。

仿照 LangGraph 的 ``State`` 语义：

- 图在执行过程中维护一个**共享状态**（一个 dict）。
- 每个节点接收当前状态，返回一个"部分状态更新"（dict）。
- 框架按 key 将更新**归并**进共享状态；每个 key 可以配置一个 reducer 决定
  如何合并（默认覆盖；也可以追加，如消息列表）。

这样既支持简单的"值覆盖"，也支持"消息累加"这类常见的多 agent 协作模式。
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional

# reducer 签名：(旧值, 新值) -> 合并后的值
Reducer = Callable[[Any, Any], Any]


def replace_reducer(old: Any, new: Any) -> Any:
    """默认归并策略：直接用新值覆盖旧值。"""
    return new


def append_reducer(old: Any, new: Any) -> Any:
    """列表追加归并：把新值追加到旧列表中。

    - 若 ``old`` 为 None，视为空列表。
    - 若 ``new`` 是 list，则逐项追加；否则作为单个元素追加。
    """
    result = list(old) if old else []
    if isinstance(new, list):
        result.extend(new)
    else:
        result.append(new)
    return result


# 语义化别名：常用于聚合多 agent 产生的消息。
add_messages = append_reducer


def default_schema() -> Dict[str, Reducer]:
    """返回框架约定字段的默认归并策略。

    - ``messages``     : 追加（多 agent 协作时累积对话）。
    - ``__failures__`` : 追加（累积失败轨迹，供路由/恢复/流控模块读取上下文）。
    """
    from .failure import FAILURES_KEY

    return {"messages": append_reducer, FAILURES_KEY: append_reducer}


class GraphState:
    """图的共享状态容器。

    :param schema:
        可选的字段 -> reducer 映射。未在 schema 中声明的字段默认使用
        ``replace_reducer``（覆盖）。
    :param initial:
        初始状态字典。
    """

    def __init__(
        self,
        schema: Optional[Dict[str, Reducer]] = None,
        initial: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._reducers: Dict[str, Reducer] = dict(schema or {})
        self._values: Dict[str, Any] = copy.deepcopy(initial) if initial else {}

    # ------------------------------------------------------------------ #
    # 基本读取
    # ------------------------------------------------------------------ #
    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __contains__(self, key: str) -> bool:
        return key in self._values

    def snapshot(self) -> Dict[str, Any]:
        """返回当前状态的深拷贝，供节点安全读取。"""
        return copy.deepcopy(self._values)

    # ------------------------------------------------------------------ #
    # 归并更新
    # ------------------------------------------------------------------ #
    def set_reducer(self, key: str, reducer: Reducer) -> None:
        self._reducers[key] = reducer

    def update(self, delta: Optional[Dict[str, Any]]) -> None:
        """把节点返回的部分更新按 key 归并进状态。"""
        if not delta:
            return
        if not isinstance(delta, dict):
            raise TypeError(
                f"节点返回值必须是 dict 或 None，实际得到：{type(delta)!r}"
            )
        for key, new_value in delta.items():
            reducer = self._reducers.get(key, replace_reducer)
            old_value = self._values.get(key)
            self._values[key] = reducer(old_value, new_value)

    def to_dict(self) -> Dict[str, Any]:
        return self.snapshot()

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"GraphState({self._values!r})"
