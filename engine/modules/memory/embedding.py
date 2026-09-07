"""Embedding 模型层：可插拔的向量化提供者。"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Protocol

from ..bge_local import resolve_bge_m3_cache_dir, resolve_bge_m3_model_path
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
        cache_dir: Optional[str] = None,
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
        self.model_name = resolve_bge_m3_model_path(model_name)
        self.cache_dir = resolve_bge_m3_cache_dir(cache_dir)
        try:
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=True,
                devices=device,
                batch_size=batch_size,
                cache_dir=self.cache_dir,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load BGE-M3 model '{self.model_name}'. "
                f"Ensure the model is available and the cache directory is writable: {self.cache_dir}"
            ) from exc

    def embed(self, text: str) -> List[float]:
        """生成文本的 1024 维稠密向量（已归一化）。"""
        output = self._model.encode(
            [text],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = _dense_output(output)[0].tolist()
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
        return [vec.tolist() for vec in _dense_output(output)]


def _dense_output(output):
    """Support both legacy and current FlagEmbedding BGE-M3 result keys."""
    dense = output.get("dense_embeds")
    if dense is None:
        dense = output.get("dense_vecs")
    if dense is None:
        raise KeyError(f"BGE-M3 response has no dense vectors; keys={sorted(output)}")
    return dense
