"""Knowledge-base domain records and serialization boundaries."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class KnowledgeBase:
    id: str
    name: str
    description: str = ""
    owner_user_id: str = ""
    workspace_id: str = "local"
    type: str = "document"
    edition: str = "standard"
    status: str = "ready"
    embedding_model: str = "hashing"
    retrieval_mode: str = "hybrid"
    chunk_strategy: str = "smart"
    chunk_size: int = 600
    chunk_overlap: int = 80
    similarity_threshold: float = 0.15
    top_k: int = 5
    rerank_enabled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    document_count: int = 0
    chunk_count: int = 0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass
class KnowledgeDocument:
    id: str
    knowledge_base_id: str
    filename: str
    source_type: str = "upload"
    source_uri: str = ""
    mime_type: str = ""
    file_size: int = 0
    checksum: str = ""
    parse_status: str = "pending"
    index_status: str = "pending"
    error_message: str = ""
    labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass
class KnowledgeChunk:
    id: str
    knowledge_base_id: str
    document_id: str
    content: str
    title: str = ""
    page_number: int | None = None
    chunk_index: int = 0
    token_count: int = 0
    embedding: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self, include_embedding: bool = False) -> dict[str, Any]:
        value = asdict(self)
        if not include_embedding: value.pop("embedding", None)
        return value
