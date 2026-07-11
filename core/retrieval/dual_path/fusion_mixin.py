"""Fusion and relation merge helpers for DualPathRetriever."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.common.logger import get_logger

from .types import RetrievalResult

logger = get_logger("A_Memorix.DualPathRetriever")


class DualPathFusionMixin:
    """Score fusion, relation merge, and pair rerank."""
    def _normalize_scores_minmax(self, results: List[RetrievalResult]) -> None:
        if not results:
            return
        vals = [float(r.score) for r in results]
        lo = min(vals)
        hi = max(vals)
        if hi - lo < 1e-12:
            for r in results:
                r.score = 1.0
            return
        for r in results:
            r.score = (float(r.score) - lo) / (hi - lo)

    def _build_minmax_score_map(self, results: List[RetrievalResult]) -> Dict[str, float]:
        if not results:
            return {}
        vals = [float(r.score) for r in results]
        lo = min(vals)
        hi = max(vals)
        if hi - lo < 1e-12:
            return {r.hash_value: 1.0 for r in results}
        return {
            r.hash_value: (float(r.score) - lo) / (hi - lo)
            for r in results
        }

    def _clone_retrieval_result(item: RetrievalResult) -> RetrievalResult:
        return RetrievalResult(
            hash_value=item.hash_value,
            content=item.content,
            score=float(item.score),
            result_type=item.result_type,
            source=item.source,
            metadata=dict(item.metadata or {}),
        )

    def _fuse_ranked_lists_weighted_rrf(
        self,
        vector_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """按 weighted RRF 融合两路段落召回。"""
        if not vector_results:
            out = sparse_results[:]
            if self.config.fusion.normalize_score:
                self._normalize_scores_minmax(out)
            return out
        if not sparse_results:
            out = vector_results[:]
            if self.config.fusion.normalize_score:
                self._normalize_scores_minmax(out)
            return out

        k = self.config.fusion.rrf_k
        w_vec = self.config.fusion.vector_weight
        w_sparse = self.config.fusion.bm25_weight
        merged: Dict[str, RetrievalResult] = {}
        score_map: Dict[str, float] = {}

        for rank, item in enumerate(vector_results, start=1):
            h = item.hash_value
            if h not in merged:
                merged[h] = item
                merged[h].source = "fusion_rrf"
            score_map[h] = score_map.get(h, 0.0) + w_vec * (1.0 / (k + rank))

        for rank, item in enumerate(sparse_results, start=1):
            h = item.hash_value
            if h not in merged:
                merged[h] = item
                merged[h].source = "fusion_rrf"
            score_map[h] = score_map.get(h, 0.0) + w_sparse * (1.0 / (k + rank))

        out = list(merged.values())
        for item in out:
            item.score = float(score_map.get(item.hash_value, 0.0))

        out.sort(key=lambda x: x.score, reverse=True)
        if self.config.fusion.normalize_score and self.config.fusion.normalize_method == "minmax":
            self._normalize_scores_minmax(out)
        return out

    def _merge_relation_results(
        self,
        vector_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """合并关系候选，按 hash 去重并保留更高分。"""
        merged: Dict[str, RetrievalResult] = {}
        for item in vector_results:
            merged[item.hash_value] = item
        for item in sparse_results:
            old = merged.get(item.hash_value)
            if old is None or float(item.score) > float(old.score):
                merged[item.hash_value] = item
            elif old is not None and old.source != item.source:
                old.source = "relation_fusion"
        out = list(merged.values())
        out.sort(key=lambda x: x.score, reverse=True)
        return out

    def _merge_relation_results_graph_enhanced(
        self,
        vector_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Graph-aware relation fusion with semantic + graph + evidence scoring."""
        vector_norm = self._build_minmax_score_map(vector_results)
        sparse_norm = self._build_minmax_score_map(sparse_results)
        graph_score_map = {
            "direct_pair": 1.0,
            "one_hop_seed": 0.75,
            "two_hop_pair": 0.55,
        }

        merged: Dict[str, RetrievalResult] = {}
        source_sets: Dict[str, set[str]] = {}
        support_cache: Dict[str, int] = {}

        for group in (vector_results, sparse_results, graph_results):
            for item in group:
                existing = merged.get(item.hash_value)
                if existing is None:
                    existing = self._clone_retrieval_result(item)
                    merged[item.hash_value] = existing
                else:
                    for key, value in dict(item.metadata or {}).items():
                        if key not in existing.metadata or existing.metadata.get(key) in (None, "", []):
                            existing.metadata[key] = value
                source_sets.setdefault(item.hash_value, set()).add(str(item.source or "").strip() or "relation_search")

        out = list(merged.values())
        for item in out:
            meta = item.metadata if isinstance(item.metadata, dict) else {}
            semantic_norm = max(
                float(vector_norm.get(item.hash_value, 0.0)),
                float(sparse_norm.get(item.hash_value, 0.0)),
            )
            graph_candidate_type = str(meta.get("graph_candidate_type", "") or "")
            graph_score = float(graph_score_map.get(graph_candidate_type, 0.0))

            if item.hash_value not in support_cache:
                cached = meta.get("supporting_paragraph_count")
                if cached is None:
                    support_cache[item.hash_value] = len(
                        self.metadata_store.get_paragraphs_by_relation(item.hash_value)
                    )
                else:
                    support_cache[item.hash_value] = max(0, int(cached))
            supporting_paragraph_count = support_cache[item.hash_value]
            evidence_score = min(1.0, supporting_paragraph_count / 3.0)

            meta["supporting_paragraph_count"] = supporting_paragraph_count
            meta["graph_seed_entities"] = list(meta.get("graph_seed_entities") or [])
            if "graph_hops" in meta:
                meta["graph_hops"] = int(meta.get("graph_hops") or 0)
            item.score = 0.60 * semantic_norm + 0.30 * graph_score + 0.10 * evidence_score

            sources = source_sets.get(item.hash_value, set())
            if len(sources) > 1:
                item.source = "relation_fusion"
            elif sources:
                item.source = next(iter(sources))

        out.sort(key=lambda x: x.score, reverse=True)
        return out

    def _fuse_results(
        self,
        para_results: List[RetrievalResult],
        rel_results: List[RetrievalResult],
        query_emb: Optional[np.ndarray] = None,
        alpha_override: Optional[float] = None,
        preserve_top_relations: int = 0,
    ) -> List[RetrievalResult]:
        """
        融合段落和关系结果

        融合策略：
        1. 计算加权分数
        2. 去重（基于段落和关系的关联）
        3. 排序

        Args:
            para_results: 段落结果
            rel_results: 关系结果
            query_emb: 查询嵌入（兼容参数，当前未使用）

        Returns:
            融合后的结果列表
        """
        del query_emb  # 参数保留用于兼容
        alpha = float(alpha_override) if alpha_override is not None else self.config.alpha

        # 为段落结果计算加权分数
        for result in para_results:
            result.score = result.score * alpha
            result.source = "fusion"

        # 为关系结果计算加权分数
        for result in rel_results:
            result.score = result.score * (1 - alpha)
            result.source = "fusion"

        preserve_top_relations = max(0, int(preserve_top_relations))
        preserved_relation_hashes = set()
        if preserve_top_relations > 0 and rel_results:
            rel_ranked = sorted(rel_results, key=lambda x: x.score, reverse=True)
            preserved_relation_hashes = {
                item.hash_value for item in rel_ranked[:preserve_top_relations]
            }

        # 合并结果
        all_results = para_results + rel_results
        all_results.sort(key=lambda x: x.score, reverse=True)

        # 去重：如果段落有关联的关系，只保留分数更高的
        seen_paragraphs = set()
        seen_items = set()
        deduplicated_results = []

        for result in all_results:
            if result.hash_value in seen_items:
                continue
            if result.result_type == "paragraph":
                hash_val = result.hash_value
                if hash_val not in seen_paragraphs:
                    seen_paragraphs.add(hash_val)
                    seen_items.add(hash_val)
                    deduplicated_results.append(result)
            else:  # relation
                if result.hash_value in preserved_relation_hashes:
                    seen_items.add(result.hash_value)
                    deduplicated_results.append(result)
                    continue
                # 检查关系关联的段落是否已存在
                relation = self.metadata_store.get_relation(result.hash_value, include_inactive=False)
                if relation:
                    # 获取关联的段落
                    para_rels = self.metadata_store.query("""
                        SELECT paragraph_hash FROM paragraph_relations
                        WHERE relation_hash = ?
                    """, (result.hash_value,))

                    if para_rels:
                        # 检查段落是否已在结果中
                        for para_rel in para_rels:
                            if para_rel["paragraph_hash"] in seen_paragraphs:
                                # 段落已存在，跳过此关系
                                break
                        else:
                            # 所有段落都不存在，添加关系
                            seen_items.add(result.hash_value)
                            deduplicated_results.append(result)
                    else:
                        # 没有关联段落，直接添加
                        seen_items.add(result.hash_value)
                        deduplicated_results.append(result)
                else:
                    seen_items.add(result.hash_value)
                    deduplicated_results.append(result)

        # 按分数排序
        deduplicated_results.sort(key=lambda x: x.score, reverse=True)

        return deduplicated_results

    def _apply_relation_intent_pair_rerank(
        self,
        results: List[RetrievalResult],
        *,
        enabled: bool,
        pair_rerank_enabled: bool,
        pair_limit: int,
    ) -> List[RetrievalResult]:
        """仅在 relation-intent 下对关系项执行同主客体多谓词重排。"""
        if not enabled or not pair_rerank_enabled:
            return results
        return self._rerank_relation_items_by_pair(results, pair_limit=pair_limit)

    def _rerank_relation_items_by_pair(
        self,
        results: List[RetrievalResult],
        pair_limit: int,
    ) -> List[RetrievalResult]:
        """
        同主客体多谓词重排：
        1. 关系项按 (subject, object) 分组
        2. 组内按分数降序 + 原始位置升序
        3. 组间按组最高分降序 + 组最早位置升序
        4. 先拼接每组前 N 条，再拼接每组 overflow 条目
        5. 回填到原关系槽位，段落槽位不变
        """
        if len(results) <= 1:
            return results

        relation_positions: List[int] = []
        relation_items: List[Tuple[int, RetrievalResult]] = []
        for idx, item in enumerate(results):
            if item.result_type == "relation":
                relation_positions.append(idx)
                relation_items.append((idx, item))

        if len(relation_items) <= 1:
            return results

        pair_limit = max(1, int(pair_limit))

        grouped: Dict[Tuple[str, str], List[Tuple[int, RetrievalResult]]] = {}
        for original_idx, item in relation_items:
            metadata = item.metadata if isinstance(item.metadata, dict) else {}
            subject = str(metadata.get("subject", "")).strip().lower()
            obj = str(metadata.get("object", "")).strip().lower()
            if subject and obj:
                key = (subject, obj)
            else:
                key = ("__missing__", item.hash_value)
            grouped.setdefault(key, []).append((original_idx, item))

        for grouped_items in grouped.values():
            grouped_items.sort(key=lambda x: (-float(x[1].score), x[0]))

        ordered_groups = sorted(
            grouped.values(),
            key=lambda grouped_items: (
                -float(grouped_items[0][1].score),
                grouped_items[0][0],
            ),
        )

        prioritized: List[RetrievalResult] = []
        overflow: List[RetrievalResult] = []
        for grouped_items in ordered_groups:
            prioritized.extend([item for _, item in grouped_items[:pair_limit]])
            overflow.extend([item for _, item in grouped_items[pair_limit:]])

        reordered_relations = prioritized + overflow
        if len(reordered_relations) != len(relation_items):
            return results

        logger.debug(
            "relation_rerank_applied=1 "
            f"relation_pair_groups={len(ordered_groups)} "
            f"relation_pair_overflow_count={len(overflow)} "
            f"relation_pair_limit={pair_limit}"
        )

        rebuilt = list(results)
        for slot_idx, relation_item in zip(relation_positions, reordered_relations):
            rebuilt[slot_idx] = relation_item
        return rebuilt

