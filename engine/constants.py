"""图编排中的哨兵节点。

仿照 LangGraph 的 ``START`` 与 ``END`` 虚拟节点：
- 从 ``START`` 出发的边定义了图的入口。
- 指向 ``END`` 的边表示一条执行路径结束。
"""

from __future__ import annotations


class _Sentinel(str):
    """一个既是字符串又可作为唯一哨兵使用的对象。

    继承自 ``str`` 使其可以直接作为节点名参与边的存取与序列化，
    同时保持单例语义（``START is START``）。
    """

    _instances: dict[str, "_Sentinel"] = {}

    def __new__(cls, value: str) -> "_Sentinel":
        if value not in cls._instances:
            obj = super().__new__(cls, value)
            cls._instances[value] = obj
        return cls._instances[value]

    def __repr__(self) -> str:  # pragma: no cover - 仅用于调试展示
        return f"<{self}>"


# 图的虚拟起点：从它出发的边即为图的入口节点。
START = _Sentinel("__start__")

# 图的虚拟终点：指向它的边表示一条路径的结束。
END = _Sentinel("__end__")
