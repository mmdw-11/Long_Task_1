"""server 子包：基于 FastAPI 的 REST 服务，供前端可视化编排。"""

from .app import create_app

__all__ = ["create_app"]
