"""Retrieval orchestration for DualPathRetriever."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.common.logger import get_logger

from ..posterior_graph import apply_posterior_graph_gate
from .types import RetrievalResult, RetrievalStrategy, TemporalQueryOptions

logger = get_logger("A_Memorix.DualPathRetriever")


class DualPathRetrieveMixin:
    """Retrieval orchestration entrypoints."""
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        strategy: Optional[RetrievalStrategy] = None,
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> List[RetrievalResult]:
        """
        执行检索（异步方法）

        Args:
            query: 查询文本
            top_k: 返回结果数量（默认使用配置值）
            strategy: 检索策略（默认使用配置值）
            temporal: 时序查询选项（可选）

        Returns:
            检索结果列表
        """
        top_k = top_k or self.config.top_k_final
        strategy = strategy or self.config.retrieval_strategy
        relation_intent_ctx = self._build_relation_intent_context(query=query, top_k=top_k)

        logger.info(
            "执行检索: "
            f"query='{query[:50]}...', "
            f"strategy={strategy.value}, "
            f"relation_intent={relation_intent_ctx.get('enabled', False)}"
        )

        if temporal and not (query or "").strip():
            return self._retrieve_temporal_only(temporal, top_k)

        # 根据策略执行检索
        if strategy == RetrievalStrategy.PARA_ONLY:
            results = await self._retrieve_paragraphs_only(query, top_k, temporal=temporal)
        elif strategy == RetrievalStrategy.REL_ONLY:
            results = await self._retrieve_relations_only(query, top_k, temporal=temporal)
        else:  # DUAL_PATH
            results = await self._retrieve_dual_path(
                query,
                top_k,
                temporal=temporal,
                relation_intent=relation_intent_ctx,
            )

        logger.info(f"检索完成: 返回 {len(results)} 条结果")

        # 调试模式：打印结果原文
        if self.config.debug:
            logger.info(f"[DEBUG] 检索结果内容原文:")
            for i, res in enumerate(results):
                logger.info(f"  {i+1}. [{res.result_type}] (Score: {res.score:.4f}) {res.content}")

        return results

    def _is_relation_intent_query(self, query: str) -> bool:
        q = str(query or "").strip()
        if not q:
            return False
        if "|" in q or "->" in q:
            return True
        return self._relation_intent_pattern.search(q) is not None

    def _build_relation_intent_context(self, query: str, top_k: int) -> Dict[str, Any]:
        cfg = self.config.relation_intent
        enabled = bool(cfg.enabled) and self._is_relation_intent_query(query)
        base_relation_k = max(1, int(self.config.top_k_relations))
        relation_top_k = max(base_relation_k, int(top_k))
        if enabled:
            relation_top_k = max(
                relation_top_k,
                relation_top_k * int(cfg.relation_candidate_multiplier),
            )
        return {
            "enabled": enabled,
            "alpha_override": float(cfg.alpha_override) if enabled else None,
            "relation_top_k": int(relation_top_k),
            "preserve_top_relations": int(cfg.preserve_top_relations) if enabled else 0,
            "force_relation_sparse": bool(cfg.force_relation_sparse) if enabled else False,
            "pair_predicate_rerank_enabled": bool(cfg.pair_predicate_rerank_enabled) if enabled else False,
            "pair_predicate_limit": int(cfg.pair_predicate_limit) if enabled else 0,
        }

    async def _retrieve_paragraphs_only(
        self,
        query: str,
        top_k: int,
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> List[RetrievalResult]:
        """
        仅检索段落（异步方法）

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        if self._is_sparse_only_runtime():
            sparse_results = self._search_paragraphs_sparse(query, top_k, temporal=temporal)
            return sparse_results[:top_k]

        query_emb = None
        embedding_ok = False
        vector_results: List[RetrievalResult] = []

        try:
            query_emb = await self.embedding_manager.encode(query)
            embedding_ok = self._is_embedding_ready_for_vector_search(
                query_emb,
                stage="paragraph_only",
            )
        except Exception as e:
            logger.warning(f"段落检索 embedding 生成失败，将尝试 sparse 回退: {e}")

        if embedding_ok:
            multiplier = max(1, temporal.candidate_multiplier) if temporal else 1
            candidate_k = self._cap_temporal_scan_k(top_k * 2 * multiplier, temporal)
            para_ids, para_scores = self.vector_store.search(
                query_emb,  # type: ignore[arg-type]
                k=candidate_k,
            )

            for hash_value, score in zip(para_ids, para_scores):
                paragraph = self.metadata_store.get_paragraph(hash_value)
                if paragraph is None:
                    continue
                time_meta = self._build_time_meta_from_paragraph(paragraph, temporal=temporal)
                vector_results.append(
                    RetrievalResult(
                        hash_value=hash_value,
                        content=paragraph["content"],
                        score=float(score),
                        result_type="paragraph",
                        source="paragraph_search",
                        metadata={
                            "word_count": paragraph.get("word_count", 0),
                            "time_meta": time_meta,
                        },
                    )
                )
            vector_results = self._apply_temporal_filter_to_paragraphs(vector_results, temporal)

        sparse_results: List[RetrievalResult] = []
        if self._should_use_sparse(embedding_ok, vector_results):
            sparse_results = self._search_paragraphs_sparse(query, top_k, temporal=temporal)

        if self.config.fusion.method == "weighted_rrf" and (vector_results and sparse_results):
            results = self._fuse_ranked_lists_weighted_rrf(vector_results, sparse_results)
        elif vector_results and sparse_results:
            results = vector_results + sparse_results
            results.sort(key=lambda x: x.score, reverse=True)
        else:
            results = vector_results if vector_results else sparse_results

        return results[:top_k]

    async def _retrieve_relations_only(
        self,
        query: str,
        top_k: int,
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> List[RetrievalResult]:
        """
        仅检索关系 (通过实体枢纽 Entity-Pivot)
        
        策略:
        1. 检索向量库中的 Top-K 实体 (Entity)
        2. 通过图结构/元数据扩展出与实体关联的关系 (Relation)
        3. 以实体相似度作为基础分返回关系

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        if self._is_sparse_only_runtime():
            sparse_results = self._search_relations_sparse(query=query, top_k=top_k, temporal=temporal)
            graph_results = self._search_relations_graph(query=query, temporal=temporal)
            if graph_results:
                merged = self._merge_relation_results_graph_enhanced(
                    [],
                    sparse_results,
                    graph_results,
                )
                return merged[:top_k]
            return sparse_results[:top_k]

        query_emb = None
        embedding_ok = False
        vector_results: List[RetrievalResult] = []
        try:
            query_emb = await self.embedding_manager.encode(query)
            embedding_ok = self._is_embedding_ready_for_vector_search(
                query_emb,
                stage="relation_only",
            )
        except Exception as e:
            logger.warning(f"关系检索 embedding 生成失败，将尝试 sparse 回退: {e}")

        if embedding_ok:
            # 1. 检索向量 (混合了段落和实体，所以扩大检索范围以召回足够多实体)
            multiplier = max(1, temporal.candidate_multiplier) if temporal else 1
            candidate_k = self._cap_temporal_scan_k(top_k * 3 * multiplier, temporal)
            ids, scores = self.vector_store.search(
                query_emb,  # type: ignore[arg-type]
                k=candidate_k,
            )

            seen_relations = set()
            for hash_value, score in zip(ids, scores):
                entity = self.metadata_store.get_entity(hash_value)
                if not entity:
                    continue
                entity_name = entity["name"]

                related_rels = []
                related_rels.extend(self.metadata_store.get_relations(subject=entity_name, include_inactive=False))
                related_rels.extend(self.metadata_store.get_relations(object=entity_name, include_inactive=False))

                for rel in related_rels:
                    if rel["hash"] in seen_relations:
                        continue
                    seen_relations.add(rel["hash"])

                    relation_time_meta = None
                    if temporal:
                        relation_time_meta = self._best_supporting_time_meta(rel["hash"], temporal)
                        if relation_time_meta is None:
                            continue

                    content = f"{rel['subject']} {rel['predicate']} {rel['object']}"
                    vector_results.append(
                        RetrievalResult(
                            hash_value=rel["hash"],
                            content=content,
                            score=float(score),
                            result_type="relation",
                            source="relation_search (via entity)",
                            metadata={
                                "subject": rel["subject"],
                                "predicate": rel["predicate"],
                                "object": rel["object"],
                                "confidence": rel.get("confidence", 1.0),
                                "pivot_entity": entity_name,
                                "time_meta": relation_time_meta,
                            },
                        )
                    )

            vector_results = self._apply_temporal_filter_to_relations(vector_results, temporal)

        sparse_results: List[RetrievalResult] = []
        if self._should_use_sparse_relations(embedding_ok, vector_results):
            sparse_results = self._search_relations_sparse(query=query, top_k=top_k, temporal=temporal)

        graph_results = self._search_relations_graph(query=query, temporal=temporal)
        if graph_results:
            results = self._merge_relation_results_graph_enhanced(
                vector_results,
                sparse_results,
                graph_results,
            )
        elif vector_results and sparse_results:
            results = self._merge_relation_results(vector_results, sparse_results)
        else:
            results = vector_results if vector_results else sparse_results

        return results[:top_k]

    async def _retrieve_dual_path(
        self,
        query: str,
        top_k: int,
        temporal: Optional[TemporalQueryOptions] = None,
        relation_intent: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        双路检索（段落+关系）（异步方法）

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            融合后的检索结果列表
        """
        query_emb = None
        embedding_ok = False
        relation_intent = relation_intent or {}
        relation_top_k = max(
            1,
            int(relation_intent.get("relation_top_k", self.config.top_k_relations)),
        )
        force_relation_sparse = bool(relation_intent.get("force_relation_sparse", False))
        preserve_top_relations = max(
            0,
            int(relation_intent.get("preserve_top_relations", 0)),
        )
        pair_predicate_rerank_enabled = bool(
            relation_intent.get("pair_predicate_rerank_enabled", False)
        )
        pair_predicate_limit = max(
            1,
            int(
                relation_intent.get(
                    "pair_predicate_limit",
                    self.config.relation_intent.pair_predicate_limit,
                )
            ),
        )
        alpha_override = relation_intent.get("alpha_override")

        if self._is_sparse_only_runtime():
            para_results = self._search_paragraphs_sparse(
                query=query,
                top_k=max(top_k * 2, self.config.sparse.candidate_k),
                temporal=temporal,
            )
            sparse_rel_results = self._search_relations_sparse(
                query=query,
                top_k=max(
                    top_k,
                    self.config.sparse.relation_candidate_k,
                    relation_top_k,
                ),
                temporal=temporal,
            )
            graph_rel_results: List[RetrievalResult] = []
            if bool(relation_intent.get("enabled", False)):
                graph_rel_results = self._search_relations_graph(query=query, temporal=temporal)
            if graph_rel_results:
                rel_results = self._merge_relation_results_graph_enhanced(
                    [],
                    sparse_rel_results,
                    graph_rel_results,
                )
            else:
                rel_results = sparse_rel_results

            fused_results = self._fuse_results(
                para_results,
                rel_results,
                None,
                alpha_override=alpha_override,
                preserve_top_relations=preserve_top_relations,
            )
            if self.config.enable_ppr:
                fused_results = await self._rerank_with_ppr(
                    fused_results,
                    query,
                )
            if temporal:
                fused_results = self._sort_results_with_temporal(fused_results, temporal)
            fused_results = apply_posterior_graph_gate(
                self,
                query=query,
                base_results=fused_results,
                top_k=top_k,
                temporal=temporal,
                relation_intent=relation_intent,
            )
            fused_results = self._apply_relation_intent_pair_rerank(
                fused_results,
                enabled=bool(relation_intent.get("enabled", False)),
                pair_rerank_enabled=pair_predicate_rerank_enabled,
                pair_limit=pair_predicate_limit,
            )
            return fused_results[:top_k]

        try:
            query_emb = await self.embedding_manager.encode(query)
            embedding_ok = self._is_embedding_ready_for_vector_search(
                query_emb,
                stage="dual_path",
            )
        except Exception as e:
            logger.warning(f"双路检索 embedding 生成失败，将尝试 sparse 回退: {e}")

        para_results: List[RetrievalResult] = []
        rel_results: List[RetrievalResult] = []
        if embedding_ok:
            # 并行检索（使用 asyncio）
            if self.config.enable_parallel:
                para_results, rel_results = await self._parallel_retrieve(
                    query_emb,
                    temporal=temporal,
                    relation_top_k=relation_top_k,
                )  # type: ignore[arg-type]
            else:
                para_results, rel_results = self._sequential_retrieve(
                    query_emb,
                    temporal=temporal,
                    relation_top_k=relation_top_k,
                )  # type: ignore[arg-type]
        else:
            logger.warning("embedding 不可用，跳过向量段落/关系召回")

        sparse_para_results: List[RetrievalResult] = []
        if self._should_use_sparse(embedding_ok, para_results):
            sparse_para_results = self._search_paragraphs_sparse(
                query=query,
                top_k=max(top_k * 2, self.config.sparse.candidate_k),
                temporal=temporal,
            )
        sparse_rel_results: List[RetrievalResult] = []
        if self._should_use_sparse_relations(
            embedding_ok,
            rel_results,
            force_enable=force_relation_sparse,
        ):
            sparse_rel_results = self._search_relations_sparse(
                query=query,
                top_k=max(
                    top_k,
                    self.config.sparse.relation_candidate_k,
                    relation_top_k,
                ),
                temporal=temporal,
            )

        graph_rel_results: List[RetrievalResult] = []
        if bool(relation_intent.get("enabled", False)):
            graph_rel_results = self._search_relations_graph(query=query, temporal=temporal)

        if self.config.fusion.method == "weighted_rrf" and para_results and sparse_para_results:
            para_results = self._fuse_ranked_lists_weighted_rrf(para_results, sparse_para_results)
        elif para_results and sparse_para_results:
            para_results = para_results + sparse_para_results
            para_results.sort(key=lambda x: x.score, reverse=True)
        elif sparse_para_results and (not para_results or not embedding_ok):
            para_results = sparse_para_results

        if graph_rel_results:
            rel_results = self._merge_relation_results_graph_enhanced(
                rel_results,
                sparse_rel_results,
                graph_rel_results,
            )
        elif rel_results and sparse_rel_results:
            rel_results = self._merge_relation_results(rel_results, sparse_rel_results)
        elif sparse_rel_results and (not rel_results or not embedding_ok):
            rel_results = sparse_rel_results

        # 融合结果
        fused_results = self._fuse_results(
            para_results,
            rel_results,
            query_emb,
            alpha_override=alpha_override,
            preserve_top_relations=preserve_top_relations,
        )

        # PageRank重排序
        if self.config.enable_ppr:
            fused_results = await self._rerank_with_ppr(
                fused_results,
                query,
            )

        if temporal:
            fused_results = self._sort_results_with_temporal(fused_results, temporal)

        fused_results = apply_posterior_graph_gate(
            self,
            query=query,
            base_results=fused_results,
            top_k=top_k,
            temporal=temporal,
            relation_intent=relation_intent,
        )

        fused_results = self._apply_relation_intent_pair_rerank(
            fused_results,
            enabled=bool(relation_intent.get("enabled", False)),
            pair_rerank_enabled=pair_predicate_rerank_enabled,
            pair_limit=pair_predicate_limit,
        )

        return fused_results[:top_k]

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取检索统计信息

        Returns:
            统计信息字典
        """
        vector_size = getattr(self.vector_store, "size", None)
        if vector_size is None:
            vector_size = getattr(self.vector_store, "num_vectors", 0)

        return {
            "config": {
                "top_k_paragraphs": self.config.top_k_paragraphs,
                "top_k_relations": self.config.top_k_relations,
                "top_k_final": self.config.top_k_final,
                "alpha": self.config.alpha,
                "enable_ppr": self.config.enable_ppr,
                "enable_parallel": self.config.enable_parallel,
                "strategy": self.config.retrieval_strategy.value,
                "sparse_mode": self.config.sparse.mode,
                "fusion_method": self.config.fusion.method,
                "relation_intent_enabled": self.config.relation_intent.enabled,
                "relation_intent_alpha_override": self.config.relation_intent.alpha_override,
                "relation_intent_candidate_multiplier": self.config.relation_intent.relation_candidate_multiplier,
                "relation_intent_preserve_top_relations": self.config.relation_intent.preserve_top_relations,
                "relation_intent_force_sparse": self.config.relation_intent.force_relation_sparse,
                "relation_intent_pair_rerank_enabled": self.config.relation_intent.pair_predicate_rerank_enabled,
                "relation_intent_pair_predicate_limit": self.config.relation_intent.pair_predicate_limit,
                "graph_recall_enabled": self.config.graph_recall.enabled,
                "graph_recall_candidate_k": self.config.graph_recall.candidate_k,
                "graph_recall_allow_two_hop_pair": self.config.graph_recall.allow_two_hop_pair,
                "graph_recall_max_paths": self.config.graph_recall.max_paths,
            },
            "vector_store": {
                "size": int(vector_size),
            },
            "graph_store": {
                "num_nodes": self.graph_store.num_nodes,
                "num_edges": self.graph_store.num_edges,
            },
            "metadata_store": self.metadata_store.get_statistics(),
            "sparse": self.sparse_index.stats() if self.sparse_index else None,
        }

    def __repr__(self) -> str:
        return (
            f"DualPathRetriever("
            f"strategy={self.config.retrieval_strategy.value}, "
            f"para_k={self.config.top_k_paragraphs}, "
            f"rel_k={self.config.top_k_relations})"
        )

