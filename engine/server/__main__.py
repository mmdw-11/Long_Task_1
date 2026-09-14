"""便于 `python -m engine.server` 直接启动服务。"""

from __future__ import annotations


def main() -> None:
    import uvicorn

    uvicorn.run(
        "engine.server.app:app",
        # Bind all container interfaces.  On a local non-Docker run this is
        # also useful when testing from another device on the same network.
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
