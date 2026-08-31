"""Deterministic chunking policies, isolated from persistence and parsing."""
from __future__ import annotations
import re

def chunk_text(text: str, strategy: str, size: int, overlap: int) -> list[str]:
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text: return []
    if strategy == "paragraph": units = [x.strip() for x in re.split(r"\n\s*\n", text) if x.strip()]
    elif strategy == "heading": units = [x.strip() for x in re.split(r"(?=^#{1,6}\s)|(?=^[^\n]{1,80}\n[=-]{3,}$)", text, flags=re.M) if x.strip()]
    elif strategy == "page": units = [x.strip() for x in text.split("\f") if x.strip()]
    elif strategy == "regex": units = [x.strip() for x in re.split(r"\n[-*•]\s+", text) if x.strip()]
    else: units = [x.strip() for x in re.split(r"(?<=[。！？.!?])\s+|\n+", text) if x.strip()]
    output: list[str] = []; current = ""
    for unit in units:
        if len(current) + len(unit) + 1 <= size:
            current = (current + "\n" + unit).strip(); continue
        if current: output.append(current)
        while len(unit) > size:
            output.append(unit[:size]); unit = unit[max(1, size - overlap):]
        current = unit
    if current: output.append(current)
    return [item for item in output if len(item) >= 20] or [text[:size]]
