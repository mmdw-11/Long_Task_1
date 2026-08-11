"""实验 CLI 的断点续跑工具。

同一条命令再次运行时，读取输出目录中的 rows.jsonl，跳过已完成样本，只执行剩余样本。
如果需要完整重跑，在 CLI 中传入 --fresh。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, TypeVar

from engine.experiments.io import read_jsonl
from engine.experiments.reports import ExperimentReport
from engine.experiments.types import ExperimentRow


T = TypeVar("T")


def load_existing_rows(output_dir: str | Path, *, fresh: bool = False) -> List[ExperimentRow]:
    """读取已有明细；fresh=True 时忽略旧结果。"""
    if fresh:
        return []
    rows_path = Path(output_dir) / "rows.jsonl"
    if not rows_path.exists():
        return []
    rows: List[ExperimentRow] = []
    for item in read_jsonl(rows_path):
        row = ExperimentRow.from_dict(item)
        if row.id:
            rows.append(row)
    return rows


def filter_remaining(examples: Sequence[T], existing_rows: Iterable[ExperimentRow]) -> List[T]:
    """根据样本 id 跳过已完成样本。"""
    done = {row.id for row in existing_rows}
    return [example for example in examples if getattr(example, "id", "") not in done]


def merge_reports(name: str, existing_rows: List[ExperimentRow], new_report: ExperimentReport) -> ExperimentReport:
    """把历史结果和本次新增结果合并成一个报告。"""
    return ExperimentReport(
        name=name,
        rows=[*existing_rows, *new_report.rows],
        metadata={**new_report.metadata, "resumed": bool(existing_rows), "previous_rows": len(existing_rows)},
    )
