"""Embedding 模型层：可插拔的向量化提供者。"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Protocol

from ._utils import _normalize, _tokenize


class EmbeddingModel(Protocol):
    """Small protocol for pluggable local or remote embedding providers."""

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError


class HashingEmbeddingModel:
    """Deterministic local embedding based on feature hashing.

    This is intentionally dependency-free. It gives usable semantic-ish keyword
    matching for tests and local development, and can be replaced by an OpenAI,
    sentence-transformers, Mem0, or Zep embedding adapter later.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        return _normalize(vector)


class BGEM3EmbeddingModel:
    """基于 BAAI/bge-m3 的稠密向量 embedding 模型。

    bge-m3 支持多语言、多粒度检索，产出 1024 维归一化向量。
    首次使用会从 HuggingFace 下载模型（约 2GB），之后会使用本地缓存。

    :param model_name: HuggingFace 模型名称，默认 ``BAAI/bge-m3``。
    :param device: 推理设备，``"cpu"`` / ``"cuda"`` / ``"mps"``；默认自动选择。
    :param batch_size: 批处理大小。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
        except ImportError:
            raise ImportError(
                "使用 BGEM3EmbeddingModel 需要安装 FlagEmbedding：\n"
                "  pip install FlagEmbedding\n"
                "或安装本项目的 memory 可选依赖组：\n"
                "  pip install -e '.[memory]'"
            )
        self.model_name = model_name
        self._model = BGEM3FlagModel(
            model_name,
            use_fp16=True,
            device=device,
            batch_size=batch_size,
        )

    def embed(self, text: str) -> List[float]:
        """生成文本的 1024 维稠密向量（已归一化）。"""
        output = self._model.encode(
            [text],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = output["dense_embeds"][0].tolist()
        return dense

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成稠密向量。"""
        if not texts:
            return []
        output = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return [vec.tolist() for vec in output["dense_embeds"]]
