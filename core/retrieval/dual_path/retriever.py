"""DualPathRetriever composed from focused mixins."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger

from ...embedding import EmbeddingAPIAdapter
from ...storage import GraphStore, MetadataStore, VectorStore
from ...utils.matcher import AhoCorasick
from ..graph_relation_recall import GraphRelationRecallService
from ..pagerank import PageRankConfig, PersonalizedPageRank
from ..sparse_bm25 import SparseBM25Index
from .embedding_mixin import DualPathEmbeddingMixin
from .fusion_mixin import DualPathFusionMixin
from .rerank_mixin import DualPathRerankMixin
from .retrieve_mixin import DualPathRetrieveMixin
from .sparse_mixin import DualPathSparseMixin
from .temporal_mixin import DualPathTemporalMixin
from .types import DualPathRetrieverConfig
from .vector_mixin import DualPathVectorMixin

logger = get_logger("A_Memorix.DualPathRetriever")


class DualPathRetriever(
    DualPathRetrieveMixin,
    DualPathVectorMixin,
    DualPathSparseMixin,
    DualPathFusionMixin,
    DualPathRerankMixin,
    DualPathTemporalMixin,
    DualPathEmbeddingMixin,
):
    """
    双路检索器

    功能：
    - 并行检索段落和关系
    - 结果融合与排序
    - PageRank重排序
    - 实体识别与加权

    参数：
        vector_store: 向量存储
        graph_store: 图存储
        metadata_store: 元数据存储
        embedding_manager: 嵌入管理器
        config: 检索配置
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        metadata_store: MetadataStore,
        embedding_manager: EmbeddingAPIAdapter,
        sparse_index: Optional[SparseBM25Index] = None,
        config: Optional[DualPathRetrieverConfig] = None,
    ):
        """
        初始化双路检索器

        Args:
            vector_store: 向量存储
            graph_store: 图存储
            metadata_store: 元数据存储
            embedding_manager: 嵌入管理器
            config: 检索配置
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.metadata_store = metadata_store
        self.embedding_manager = embedding_manager
        self.config = config or DualPathRetrieverConfig()
        self.sparse_index = sparse_index

        # PageRank计算器
        ppr_config = PageRankConfig(alpha=self.config.ppr_alpha)
        self._ppr = PersonalizedPageRank(
            graph_store=graph_store,
            config=ppr_config,
        )
        self._ppr_semaphore = asyncio.Semaphore(self.config.ppr_concurrency_limit)
        self._graph_relation_recall = GraphRelationRecallService(
            graph_store=graph_store,
            metadata_store=metadata_store,
            config=self.config.graph_recall,
        )

        logger.debug(
            f"DualPathRetriever 初始化: "
            f"strategy={self.config.retrieval_strategy.value}, "
            f"top_k_para={self.config.top_k_paragraphs}, "
            f"top_k_rel={self.config.top_k_relations}"
        )

        # 缓存 Aho-Corasick 匹配器
        self._ac_matcher: Optional[AhoCorasick] = None
        self._ac_nodes_count = 0
        self._relation_intent_pattern = re.compile(
            r"(什么关系|有哪些关系|和.+关系|关联|关系网|subject|predicate|object|"
            r"relation|related|between.+and)",
            re.IGNORECASE,
        )
        self._runtime_sparse_only = False
