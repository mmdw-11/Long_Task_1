"""Fail fast when required offline Docker model assets are absent."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ENCODER = ROOT / "runs" / "router_learning" / "final_bge_m3_contrastive_smoke" / "encoder"
ROUTER = ROOT / "runs" / "router_learning" / "final_bge_m3_contrastive_smoke" / "router"

REQUIRED = (
    ENCODER / "config.json",
    ENCODER / "modules.json",
    ENCODER / "tokenizer.json",
    ENCODER / "model.safetensors",
    ROUTER / "classifier.pkl",
    ROUTER / "summary.json",
)


def _size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    missing = [path.relative_to(ROOT) for path in REQUIRED if not path.is_file()]
    if missing:
        print("Docker 离线模型资产不完整：")
        for path in missing:
            print(f"- {path}")
        print("请恢复 final_bge_m3_contrastive_smoke 的 encoder 与 router 目录后再构建镜像。")
        return 1
    print("Docker 离线模型资产检查通过。")
    print(f"BGE-M3 encoder: {_size(ENCODER) / 1024**3:.2f} GiB")
    print(f"Route classifier: {_size(ROUTER) / 1024**2:.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
