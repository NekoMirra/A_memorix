"""Vector store candidate collection for DualPathRetriever."""

from __future__ import annotations

from typing import List, Optional, Tuple

import asyncio

import numpy as np

from src.common.logger import get_logger

from .types import RetrievalResult, TemporalQueryOptions

logger = get_logger("A_Memorix.DualPathRetriever")


class DualPathVectorMixin:
    """Vector/FAISS candidate collection helpers."""
    def _mixed_candidate_budget(
        self,
        para_top_k: int,
        rel_top_k: int,
        temporal: Optional[TemporalQueryOptions],
    ) -> int:
        multiplier = max(1, temporal.candidate_multiplier) if temporal else 1
        base = max(para_top_k + rel_top_k, max(para_top_k, rel_top_k) * 2)
        return max(base * 6 * multiplier, 48)

    def _merge_backfilled_results(
        self,
        *,
        primary_results: List[RetrievalResult],
        backfill_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        merged: Dict[str, RetrievalResult] = {}
        for item in primary_results:
            merged[item.hash_value] = item
        for item in backfill_results:
            existing = merged.get(item.hash_value)
            if existing is None or float(item.score) > float(existing.score):
                merged[item.hash_value] = item

        results = list(merged.values())
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def _collect_mixed_candidates(
        self,
        query_emb: np.ndarray,
        temporal: Optional[TemporalQueryOptions] = None,
        relation_top_k: Optional[int] = None,
    ) -> Tuple[List[RetrievalResult], List[RetrievalResult]]:
        para_top_k = self.config.top_k_paragraphs
        rel_top_k = relation_top_k if relation_top_k is not None else self.config.top_k_relations
        candidate_k = self._mixed_candidate_budget(para_top_k, rel_top_k, temporal)
        candidate_k = self._cap_temporal_scan_k(candidate_k, temporal)
        ids, scores = self.vector_store.search(query_emb, k=candidate_k)

        para_candidates: List[RetrievalResult] = []
        rel_candidates: List[RetrievalResult] = []
        seen_para = set()
        seen_rel = set()

        for hash_value, score in zip(ids, scores):
            paragraph = self.metadata_store.get_paragraph(hash_value)
            if paragraph is not None and hash_value not in seen_para:
                seen_para.add(hash_value)
                para_candidates.append(
                    RetrievalResult(
                        hash_value=hash_value,
                        content=paragraph["content"],
                        score=float(score),
                        result_type="paragraph",
                        source="paragraph_search",
                        metadata={
                            "word_count": paragraph.get("word_count", 0),
                            "time_meta": self._build_time_meta_from_paragraph(
                                paragraph,
                                temporal=temporal,
                            ),
                        },
                    )
                )
                continue

            relation = self.metadata_store.get_relation(hash_value, include_inactive=False)
            if relation is None or hash_value in seen_rel:
                continue

            relation_time_meta = None
            if temporal:
                relation_time_meta = self._best_supporting_time_meta(hash_value, temporal)
                if relation_time_meta is None:
                    continue

            seen_rel.add(hash_value)
            rel_candidates.append(
                RetrievalResult(
                    hash_value=hash_value,
                    content=f"{relation['subject']} {relation['predicate']} {relation['object']}",
                    score=float(score),
                    result_type="relation",
                    source="relation_search",
                    metadata={
                        "subject": relation["subject"],
                        "predicate": relation["predicate"],
                        "object": relation["object"],
                        "confidence": relation.get("confidence", 1.0),
                        "time_meta": relation_time_meta,
                    },
                )
            )

        para_results = self._apply_temporal_filter_to_paragraphs(para_candidates, temporal)
        rel_results = self._apply_temporal_filter_to_relations(rel_candidates, temporal)

        # 双重方案里，向量主干优先解决“召回不够”，因此主检索走共享候选池，
        # 但再补一层按类型回填，避免 paragraph / relation 任一侧被饿死。
        para_backfill = self._search_paragraphs(query_emb, para_top_k, temporal)
        rel_backfill = self._search_relations(query_emb, rel_top_k, temporal)
        para_results = self._merge_backfilled_results(
            primary_results=para_results,
            backfill_results=para_backfill,
            top_k=para_top_k,
        )
        rel_results = self._merge_backfilled_results(
            primary_results=rel_results,
            backfill_results=rel_backfill,
            top_k=rel_top_k,
        )
        return para_results, rel_results

    def _search_paragraphs(
        self,
        query_emb: np.ndarray,
        top_k: int,
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> List[RetrievalResult]:
        """
        搜索段落

        Args:
            query_emb: 查询嵌入
            top_k: 返回数量

        Returns:
            段落结果列表
        """
        multiplier = max(1, temporal.candidate_multiplier) if temporal else 1
        candidate_k = self._cap_temporal_scan_k(top_k * multiplier, temporal)
        para_ids, para_scores = self.vector_store.search(query_emb, k=candidate_k)

        results = []
        for hash_value, score in zip(para_ids, para_scores):
            paragraph = self.metadata_store.get_paragraph(hash_value)
            if paragraph is None:
                continue

            time_meta = self._build_time_meta_from_paragraph(
                paragraph,
                temporal=temporal,
            )
            results.append(RetrievalResult(
                hash_value=hash_value,
                content=paragraph["content"],
                score=float(score),
                result_type="paragraph",
                source="paragraph_search",
                metadata={
                    "word_count": paragraph.get("word_count", 0),
                    "time_meta": time_meta,
                },
            ))

        return self._apply_temporal_filter_to_paragraphs(results, temporal)

    def _search_relations(
        self,
        query_emb: np.ndarray,
        top_k: int,
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> List[RetrievalResult]:
        """
        搜索关系

        Args:
            query_emb: 查询嵌入
            top_k: 返回数量

        Returns:
            关系结果列表
        """
        multiplier = max(1, temporal.candidate_multiplier) if temporal else 1
        candidate_k = self._cap_temporal_scan_k(top_k * multiplier, temporal)
        rel_ids, rel_scores = self.vector_store.search(query_emb, k=candidate_k)

        results = []
        for hash_value, score in zip(rel_ids, rel_scores):
            relation = self.metadata_store.get_relation(hash_value, include_inactive=False)
            if relation is None:
                continue

            relation_time_meta = None
            if temporal:
                relation_time_meta = self._best_supporting_time_meta(hash_value, temporal)
                if relation_time_meta is None:
                    continue

            content = f"{relation['subject']} {relation['predicate']} {relation['object']}"

            results.append(RetrievalResult(
                hash_value=hash_value,
                content=content,
                score=float(score),
                result_type="relation",
                source="relation_search",
                metadata={
                    "subject": relation["subject"],
                    "predicate": relation["predicate"],
                    "object": relation["object"],
                    "confidence": relation.get("confidence", 1.0),
                    "time_meta": relation_time_meta,
                },
            ))

        return self._apply_temporal_filter_to_relations(results, temporal)

    async def _parallel_retrieve(
        self,
        query_emb: np.ndarray,
        temporal: Optional[TemporalQueryOptions] = None,
        relation_top_k: Optional[int] = None,
    ) -> Tuple[List[RetrievalResult], List[RetrievalResult]]:
        """
        并行检索段落和关系（异步方法）

        Args:
            query_emb: 查询嵌入

        Returns:
            (段落结果, 关系结果)
        """
        try:
            return await asyncio.to_thread(
                self._collect_mixed_candidates,
                query_emb,
                temporal,
                relation_top_k,
            )
        except Exception as e:
            logger.error(f"并行检索失败: {e}")
            return [], []

    def _sequential_retrieve(
        self,
        query_emb: np.ndarray,
        temporal: Optional[TemporalQueryOptions] = None,
        relation_top_k: Optional[int] = None,
    ) -> Tuple[List[RetrievalResult], List[RetrievalResult]]:
        """
        顺序检索段落和关系

        Args:
            query_emb: 查询嵌入

        Returns:
            (段落结果, 关系结果)
        """
        return self._collect_mixed_candidates(
            query_emb,
            temporal,
            relation_top_k,
        )

