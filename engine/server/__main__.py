"""便于 `python -m engine.server` 直接启动服务。"""

from __future__ import annotations


def main() -> None:
    import uvicorn

    uvicorn.run(
        "engine.server.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
