"""Enterprise knowledge-base domain package.

Public imports stay stable while implementation is organized by responsibility.
"""
from .chunking import chunk_text
from .models import KnowledgeBase, KnowledgeChunk, KnowledgeDocument
from .store import KnowledgeStore

__all__ = ["KnowledgeBase", "KnowledgeChunk", "KnowledgeDocument", "KnowledgeStore", "chunk_text"]
