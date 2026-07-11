"""Sparse BM25 / graph recall helpers for DualPathRetriever."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.common.logger import get_logger

from .types import RetrievalResult, TemporalQueryOptions

logger = get_logger("A_Memorix.DualPathRetriever")


class DualPathSparseMixin:
    """Sparse BM25 and graph relation recall helpers."""
    def _search_paragraphs_sparse(
        self,
        query: str,
        top_k: int,
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> List[RetrievalResult]:
        """BM25 段落召回。"""
        if not self.sparse_index or not self.config.sparse.enabled:
            return []

        candidate_k = max(top_k, self.config.sparse.candidate_k)
        candidate_k = self._cap_temporal_scan_k(candidate_k, temporal)
        sparse_rows = self.sparse_index.search(query=query, k=candidate_k)
        sparse_rows = self._filter_sparse_paragraph_rows(sparse_rows)
        results: List[RetrievalResult] = []
        for row in sparse_rows:
            hash_value = row["hash"]
            paragraph = self.metadata_store.get_paragraph(hash_value)
            if paragraph is None:
                continue
            time_meta = self._build_time_meta_from_paragraph(paragraph, temporal=temporal)
            results.append(
                RetrievalResult(
                    hash_value=hash_value,
                    content=paragraph["content"],
                    score=float(row.get("score", 0.0)),
                    result_type="paragraph",
                    source="sparse_bm25",
                    metadata={
                        "word_count": paragraph.get("word_count", 0),
                        "time_meta": time_meta,
                        "bm25_score": float(row.get("bm25_score", 0.0)),
                    },
                )
            )
        results = self._apply_temporal_filter_to_paragraphs(results, temporal)
        if self.config.fusion.normalize_score and self.config.fusion.normalize_method == "minmax":
            self._normalize_scores_minmax(results)
        return results

    def _filter_sparse_paragraph_rows(
        self,
        rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        过滤 paragraph sparse tail。

        目标不是压缩强 lexical hit，而是避免只命中一个弱 token 的尾部结果
        在 weighted RRF 中拿到过高的 rank credit。
        """
        if len(rows) <= 2:
            return rows

        top_score = max(0.0, float(rows[0].get("score", 0.0) or 0.0))
        if top_score <= 0.0:
            return rows[:2]

        relative_floor = top_score * 0.2
        filtered_rows: List[Dict[str, Any]] = []
        removed_count = 0
        for index, row in enumerate(rows):
            if index < 2:
                filtered_rows.append(row)
                continue

            raw_score = float(row.get("score", 0.0) or 0.0)
            matched_token_count = int(row.get("matched_token_count", 0) or 0)
            matched_token_ratio = float(row.get("matched_token_ratio", 0.0) or 0.0)

            if (
                raw_score >= relative_floor
                or matched_token_count >= 3
                or (matched_token_count >= 2 and matched_token_ratio >= 0.12)
            ):
                filtered_rows.append(row)
                continue

            removed_count += 1

        if removed_count > 0:
            logger.debug(
                "sparse_paragraph_tail_pruned=1 "
                f"removed_count={removed_count} "
                f"kept_count={len(filtered_rows)}"
            )
        return filtered_rows

    def _search_relations_sparse(
        self,
        query: str,
        top_k: int,
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> List[RetrievalResult]:
        """关系 BM25 召回。"""
        if not self.sparse_index or not self.config.sparse.enabled:
            return []
        if not self.config.sparse.enable_relation_sparse_fallback:
            return []

        candidate_k = max(top_k, self.config.sparse.relation_candidate_k)
        candidate_k = self._cap_temporal_scan_k(candidate_k, temporal)
        rows = self.sparse_index.search_relations(query=query, k=candidate_k)
        results: List[RetrievalResult] = []
        for row in rows:
            hash_value = row["hash"]
            relation = self.metadata_store.get_relation(hash_value, include_inactive=False)
            if relation is None:
                continue

            relation_time_meta = None
            if temporal:
                relation_time_meta = self._best_supporting_time_meta(hash_value, temporal)
                if relation_time_meta is None:
                    continue

            content = f"{relation['subject']} {relation['predicate']} {relation['object']}"
            results.append(
                RetrievalResult(
                    hash_value=hash_value,
                    content=content,
                    score=float(row.get("score", 0.0)),
                    result_type="relation",
                    source="sparse_relation_bm25",
                    metadata={
                        "subject": relation["subject"],
                        "predicate": relation["predicate"],
                        "object": relation["object"],
                        "confidence": relation.get("confidence", 1.0),
                        "time_meta": relation_time_meta,
                        "bm25_score": float(row.get("bm25_score", 0.0)),
                    },
                )
            )

        if self.config.fusion.normalize_score and self.config.fusion.normalize_method == "minmax":
            self._normalize_scores_minmax(results)
        return self._apply_temporal_filter_to_relations(results, temporal)

    def _extract_graph_seed_entities(self, query: str, limit: int = 2) -> List[str]:
        entities = self._extract_entities(query)
        if not entities:
            return []
        ranked = sorted(
            entities.items(),
            key=lambda x: (-float(x[1]), -len(str(x[0])), str(x[0]).lower()),
        )
        return [str(name) for name, _ in ranked[: max(0, int(limit))]]

    def _search_relations_graph(
        self,
        query: str,
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> List[RetrievalResult]:
        service = getattr(self, "_graph_relation_recall", None)
        if service is None or not bool(getattr(self.config.graph_recall, "enabled", True)):
            return []

        seed_entities = self._extract_graph_seed_entities(query, limit=2)
        if not seed_entities:
            return []

        payloads = service.recall(seed_entities=seed_entities)
        results: List[RetrievalResult] = []
        for payload in payloads:
            meta = payload.to_payload()
            results.append(
                RetrievalResult(
                    hash_value=str(meta["hash"]),
                    content=str(meta["content"]),
                    score=0.0,
                    result_type="relation",
                    source="graph_relation_recall",
                    metadata={
                        "subject": meta["subject"],
                        "predicate": meta["predicate"],
                        "object": meta["object"],
                        "confidence": float(meta["confidence"]),
                        "graph_seed_entities": list(meta["graph_seed_entities"]),
                        "graph_hops": int(meta["graph_hops"]),
                        "graph_candidate_type": str(meta["graph_candidate_type"]),
                        "supporting_paragraph_count": int(meta["supporting_paragraph_count"]),
                    },
                )
            )
        return self._apply_temporal_filter_to_relations(results, temporal)

