"""Embedding readiness helpers for DualPathRetriever."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.common.logger import get_logger

from .types import RetrievalResult

logger = get_logger("A_Memorix.DualPathRetriever")


class DualPathEmbeddingMixin:
    """Embedding readiness and sparse-mode decision helpers."""
    def set_runtime_sparse_only(self, enabled: bool) -> None:
        """由运行时控制强制 sparse-only（不改用户配置文件）。"""
        self._runtime_sparse_only = bool(enabled)

    def _is_sparse_only_runtime(self) -> bool:
        mode = str(getattr(self.config.sparse, "mode", "auto") or "auto").strip().lower()
        return bool(self._runtime_sparse_only or mode == "fallback_only")

    def _is_valid_embedding(self, emb: Optional[np.ndarray]) -> bool:
        if emb is None:
            return False
        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim == 0 or arr.size == 0:
            return False
        return bool(np.all(np.isfinite(arr)))

    def _get_embedding_dim(self, emb: Optional[np.ndarray]) -> Optional[int]:
        if emb is None:
            return None
        arr = np.asarray(emb)
        if arr.ndim == 1:
            return int(arr.shape[0]) if arr.size > 0 else None
        if arr.ndim == 2:
            if arr.shape[0] == 0:
                return None
            return int(arr.shape[1])
        return None

    def _is_embedding_dimension_compatible(self, emb: Optional[np.ndarray]) -> bool:
        got_dim = self._get_embedding_dim(emb)
        expected_dim = int(getattr(self.vector_store, "dimension", 0) or 0)
        if got_dim is None or expected_dim <= 0:
            return False
        return got_dim == expected_dim

    def _is_embedding_ready_for_vector_search(
        self,
        emb: Optional[np.ndarray],
        *,
        stage: str,
    ) -> bool:
        if not self._is_valid_embedding(emb):
            return False
        if self._is_embedding_dimension_compatible(emb):
            return True

        expected_dim = int(getattr(self.vector_store, "dimension", 0) or 0)
        got_dim = self._get_embedding_dim(emb)
        logger.warning(
            "metric.embedding_dim_mismatch_fallback_count=1 "
            f"stage={stage} expected_dim={expected_dim} got_dim={got_dim}"
        )
        return False

    def _should_use_sparse(
        self,
        embedding_ok: bool,
        vector_results: Optional[List[RetrievalResult]] = None,
    ) -> bool:
        if not self.config.sparse.enabled or self.sparse_index is None:
            return False

        mode = self.config.sparse.mode
        if mode == "hybrid":
            return True
        if mode == "fallback_only":
            return True
        # auto
        if not embedding_ok:
            return True
        if not vector_results:
            return True
        best = max((float(r.score) for r in vector_results), default=0.0)
        return best < 0.45

    def _should_use_sparse_relations(
        self,
        embedding_ok: bool,
        relation_results: Optional[List[RetrievalResult]] = None,
        force_enable: bool = False,
    ) -> bool:
        if force_enable and self.config.sparse.enabled and self.sparse_index is not None:
            return True
        if not self.config.sparse.enable_relation_sparse_fallback:
            return False
        return self._should_use_sparse(embedding_ok, relation_results)

