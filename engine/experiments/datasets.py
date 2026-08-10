"""实验数据集准备与标准化。

本文件支持四类数据：
1. LongMemEval：长期记忆主实验。
2. LoCoMo：长期对话记忆补充实验。
3. SkillEvolBench：技能演化主实验的小子集。
4. 本项目自建 300 条 workflow 编排数据集。

公开数据集版本可能字段不同，所以标准化逻辑采用“候选字段优先级”的方式，尽量兼容
JSON/JSONL 中常见字段名。无法识别的样本会被跳过，而不是让整批实验中断。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .io import read_json_or_jsonl, write_jsonl
from .types import MemoryExample, SkillExample, WorkflowExample


def load_memory_dataset(
    path: str | Path,
    *,
    source: str,
    limit: int = 0,
) -> List[MemoryExample]:
    """加载 LongMemEval/LoCoMo 这类长期记忆数据。"""
    rows = read_json_or_jsonl(path)
    examples: List[MemoryExample] = []
    for index, row in enumerate(rows):
        example = _memory_example_from_row(row, source=source, index=index)
        if example is not None:
            examples.append(example)
        if limit and len(examples) >= limit:
            break
    if not examples:
        raise ValueError(f"no usable memory examples loaded from {path}")
    return examples


def load_skill_dataset(
    path: str | Path,
    *,
    source: str = "skillevolbench",
    limit: int = 0,
) -> List[SkillExample]:
    """加载 SkillEvolBench 或等价的技能演化数据。"""
    rows = read_json_or_jsonl(path)
    examples: List[SkillExample] = []
    for index, row in enumerate(rows):
        example = _skill_example_from_row(row, source=source, index=index)
        if example is not None:
            examples.append(example)
        if limit and len(examples) >= limit:
            break
    if not examples:
        raise ValueError(f"no usable skill examples loaded from {path}")
    return examples


def save_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> Path:
    """对外暴露一个统一的 JSONL 保存函数。"""
    return write_jsonl(path, rows)


def build_workflow_dataset(
    *,
    size: int = 300,
    seed: int = 42,
    output: Optional[str | Path] = None,
) -> List[WorkflowExample]:
    """构造 300 条小型 workflow 编排数据。

    数据覆盖顺序、扇出扇入、条件分支、父子 Agent、带循环上限五类结构。它不依赖真实
    LLM，因此适合作为编排引擎的第一批可复现实验。
    """
    if size <= 0:
        raise ValueError("size must be positive")
    rng = random.Random(seed)
    patterns = ["sequential", "fanout", "conditional", "parent_child", "loop"]
    topics = [
        "整理会议纪要",
        "制定旅行计划",
        "分析代码缺陷",
        "规划课程学习",
        "生成产品日报",
        "处理客户邮件",
        "检查预算风险",
        "安排日历会议",
        "准备项目复盘",
        "汇总调研材料",
    ]
    examples: List[WorkflowExample] = []
    for index in range(size):
        pattern = patterns[index % len(patterns)]
        topic = rng.choice(topics)
        branch = rng.choice(["L", "R"]) if pattern == "conditional" else None
        expected = _expected_agents_for_pattern(pattern, branch)
        examples.append(
            WorkflowExample(
                id=f"workflow-{index:03d}",
                pattern=pattern,
                prompt=f"请用 {pattern} 编排完成任务：{topic}，并保留可检查的执行轨迹。",
                expected_agents=expected,
                branch=branch,
                metadata={"topic": topic, "seed": seed},
            )
        )
    if output:
        write_jsonl(output, [_workflow_to_dict(item) for item in examples])
    return examples


def load_workflow_dataset(path: str | Path, *, limit: int = 0) -> List[WorkflowExample]:
    """读取已经生成的 workflow JSONL。"""
    rows = read_json_or_jsonl(path)
    examples: List[WorkflowExample] = []
    for row in rows:
        try:
            examples.append(
                WorkflowExample(
                    id=str(row["id"]),
                    pattern=str(row["pattern"]),
                    prompt=str(row["prompt"]),
                    expected_agents=[str(item) for item in row.get("expected_agents") or []],
                    branch=row.get("branch"),
                    metadata=dict(row.get("metadata") or {}),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
        if limit and len(examples) >= limit:
            break
    if not examples:
        raise ValueError(f"no usable workflow examples loaded from {path}")
    return examples


def _memory_example_from_row(row: Dict[str, Any], *, source: str, index: int) -> Optional[MemoryExample]:
    question = _first_text(row, ["question", "query", "input", "prompt", "q"])
    answer = _first_text(row, ["answer", "gold_answer", "target", "output", "a", "expected_answer"])
    memories = _memory_texts(row)
    if not question or not answer or not memories:
        return None
    return MemoryExample(
        id=str(row.get("id") or row.get("question_id") or f"{source}-{index:05d}"),
        question=question,
        answer=answer,
        memories=memories,
        source=source,
        metadata={k: v for k, v in row.items() if k not in {"question", "answer", "haystack_sessions"}},
    )


def _skill_example_from_row(row: Dict[str, Any], *, source: str, index: int) -> Optional[SkillExample]:
    task = _first_text(row, ["task", "instruction", "prompt", "query", "input", "problem"])
    trajectory = _first_text(row, ["trajectory", "trace", "experience", "demonstration", "history", "solution"])
    if not trajectory:
        trajectory = "\n".join(_string_list(row.get("trajectories") or row.get("examples")))
    expected_steps = _string_list(
        row.get("expected_steps")
        or row.get("steps")
        or row.get("key_steps")
        or row.get("rubric")
        or row.get("answer")
    )
    if not task or not trajectory:
        return None
    if not expected_steps:
        expected_steps = _fallback_expected_steps(trajectory)
    return SkillExample(
        id=str(row.get("id") or row.get("task_id") or f"{source}-{index:05d}"),
        task=task,
        trajectory=trajectory,
        expected_steps=expected_steps,
        task_type=str(row.get("task_type") or row.get("domain") or "general"),
        source=source,
        metadata=dict(row.get("metadata") or {}),
    )


def _memory_texts(row: Dict[str, Any]) -> List[str]:
    for key in ("memories", "memory", "contexts", "context", "evidence", "documents", "haystack"):
        value = row.get(key)
        items = _string_list(value)
        if items:
            return items
    sessions = row.get("haystack_sessions") or row.get("sessions") or row.get("conversation")
    items = _flatten_conversation(sessions)
    return items


def _flatten_conversation(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        parts: List[str] = []
        for item in value.values():
            parts.extend(_flatten_conversation(item))
        return parts
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role") or item.get("speaker") or item.get("name") or ""
                content = _first_text(item, ["content", "text", "message", "utterance", "value"])
                if content:
                    parts.append(f"{role}: {content}" if role else content)
                else:
                    parts.extend(_flatten_conversation(item))
            else:
                parts.extend(_flatten_conversation(item))
        return parts
    return []


def _first_text(row: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
    return ""


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items()]
    if isinstance(value, list):
        items: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                items.append(item.strip())
            elif isinstance(item, dict):
                text = _first_text(item, ["content", "text", "message", "step", "answer", "action"])
                if text:
                    items.append(text)
        return items
    return []


def _fallback_expected_steps(trajectory: str) -> List[str]:
    lines = [line.strip(" -\t") for line in trajectory.splitlines() if line.strip()]
    return lines[:3] if lines else [trajectory[:80]]


def _expected_agents_for_pattern(pattern: str, branch: Optional[str]) -> List[str]:
    if pattern == "sequential":
        return ["planner", "executor", "reviewer"]
    if pattern == "fanout":
        return ["planner", "researcher", "writer", "reviewer"]
    if pattern == "conditional":
        return ["router", "left_worker" if branch == "L" else "right_worker"]
    if pattern == "parent_child":
        return ["parent", "child_a", "child_b"]
    if pattern == "loop":
        return ["worker", "worker", "worker"]
    return []


def _workflow_to_dict(item: WorkflowExample) -> Dict[str, Any]:
    return {
        "id": item.id,
        "pattern": item.pattern,
        "prompt": item.prompt,
        "expected_agents": list(item.expected_agents),
        "branch": item.branch,
        "metadata": dict(item.metadata),
    }
