"""实验报告汇总与落盘。

所有模块统一输出 summary.json、rows.jsonl、report.md，后续写论文表格时可以直接读取
summary，也可以人工查看 Markdown。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from .io import write_json, write_jsonl
from .types import ExperimentRow


@dataclass
class ExperimentReport:
    """一次实验的完整结果。"""

    name: str
    rows: List[ExperimentRow]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        total = len(self.rows)
        passed = sum(1 for row in self.rows if row.passed)
        scores = [float(row.score) for row in self.rows]
        summary = {
            "name": self.name,
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total, 6) if total else 0.0,
            "avg_score": round(mean(scores), 6) if scores else 0.0,
            "metadata": dict(self.metadata),
        }
        summary.update(_average_numeric_metrics(self.rows))
        return summary

    def to_markdown(self) -> str:
        data = self.summary()
        lines = [
            f"# {self.name}",
            "",
            "## 汇总",
            "",
            f"- 样本数：{data['total']}",
            f"- 通过数：{data['passed']}",
            f"- 通过率：{data['pass_rate']:.4f}",
            f"- 平均分：{data['avg_score']:.4f}",
        ]
        for key, value in sorted(data.items()):
            if key.startswith("avg_") and key != "avg_score":
                lines.append(f"- {key}：{value}")
        lines.extend(["", "## 前 20 条明细", "", "| id | passed | score | expected | prediction |", "|---|---:|---:|---|---|"])
        for row in self.rows[:20]:
            lines.append(
                "| {id} | {passed} | {score:.4f} | {expected} | {prediction} |".format(
                    id=_cell(row.id),
                    passed="是" if row.passed else "否",
                    score=row.score,
                    expected=_cell(row.expected[:80]),
                    prediction=_cell(row.prediction[:80]),
                )
            )
        return "\n".join(lines) + "\n"


def save_report(report: ExperimentReport, output_dir: str | Path) -> Dict[str, str]:
    """保存统一实验产物。"""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = write_json(root / "summary.json", report.summary())
    rows_path = write_jsonl(root / "rows.jsonl", [row.to_dict() for row in report.rows])
    markdown_path = root / "report.md"
    markdown_path.write_text(report.to_markdown(), encoding="utf-8")
    return {
        "summary": str(summary_path),
        "rows": str(rows_path),
        "report": str(markdown_path),
    }


def _average_numeric_metrics(rows: List[ExperimentRow]) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.metrics.items():
            if isinstance(value, (int, float)):
                buckets.setdefault(key, []).append(float(value))
    return {
        f"avg_{key}": round(mean(values), 6)
        for key, values in buckets.items()
        if values
    }


def _cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
