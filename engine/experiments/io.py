"""实验文件读写工具。

公开数据集常见格式是 JSON、JSONL，也可能已经被你预处理成列表或字典。本文件只负责
可靠读取，不包含具体数据集语义。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


def read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """读取 JSON/JSONL 文件，统一返回对象列表。"""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"dataset not found: {data_path}")
    if data_path.suffix.lower() == ".jsonl":
        return list(read_jsonl(data_path))
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "examples", "samples", "questions", "items", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    raise ValueError(f"unsupported JSON dataset payload in {data_path}")


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    """逐行读取 JSONL。空行会被跳过。"""
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, dict):
                yield data


def write_json(path: str | Path, payload: Any) -> Path:
    """写 JSON 文件，自动创建父目录。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> Path:
    """写 JSONL 文件，自动创建父目录。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return out
