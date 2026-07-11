"""Dual-path retrieval types and configs."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from ..graph_relation_recall import GraphRelationRecallConfig
from ..posterior_graph import PosteriorGraphConfig
from ..sparse_bm25 import SparseBM25Config


class RetrievalStrategy(Enum):
    """检索策略"""

    PARA_ONLY = "paragraph_only"  # 仅段落检索
    REL_ONLY = "relation_only"   # 仅关系检索
    DUAL_PATH = "dual_path"      # 双路检索（推荐）


@dataclass
class RetrievalResult:
    """
    检索结果

    属性：
        hash_value: 哈希值
        content: 内容（段落或关系）
        score: 相似度分数
        result_type: 结果类型（paragraph/relation）
        source: 来源（paragraph_search/relation_search/fusion）
        metadata: 额外元数据
    """

    hash_value: str
    content: str
    score: float
    result_type: str  # "paragraph" or "relation"
    source: str  # "paragraph_search", "relation_search", "fusion"
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "hash": self.hash_value,
            "content": self.content,
            "score": self.score,
            "type": self.result_type,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class DualPathRetrieverConfig:
    """
    双路检索器配置

    属性：
        top_k_paragraphs: 段落检索数量
        top_k_relations: 关系检索数量
        top_k_final: 最终返回数量
        alpha: 段落和关系的融合权重（0-1）
            - 0: 仅使用关系分数
            - 1: 仅使用段落分数
            - 0.5: 平均融合
        enable_ppr: 是否启用PageRank重排序
        ppr_alpha: PageRank的alpha参数
        ppr_concurrency_limit: PPR计算的最大并发数
        enable_parallel: 是否并行检索
        retrieval_strategy: 检索策略
        debug: 是否启用调试模式（打印搜索结果原文）
    """
 
    top_k_paragraphs: int = 20
    top_k_relations: int = 10
    top_k_final: int = 10
    alpha: float = 0.5  # 融合权重
    enable_ppr: bool = True
    ppr_alpha: float = 0.85
    ppr_timeout_seconds: float = 1.5
    ppr_concurrency_limit: int = 4
    enable_parallel: bool = True
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.DUAL_PATH
    debug: bool = False
    sparse: SparseBM25Config = field(default_factory=SparseBM25Config)
    fusion: "FusionConfig" = field(default_factory=lambda: FusionConfig())
    relation_intent: "RelationIntentConfig" = field(default_factory=lambda: RelationIntentConfig())
    graph_recall: GraphRelationRecallConfig = field(default_factory=GraphRelationRecallConfig)
    posterior_graph: PosteriorGraphConfig = field(default_factory=PosteriorGraphConfig)

    def __post_init__(self):
        """验证配置"""
        if isinstance(self.sparse, dict):
            self.sparse = SparseBM25Config(**self.sparse)
        if isinstance(self.fusion, dict):
            self.fusion = FusionConfig(**self.fusion)
        if isinstance(self.relation_intent, dict):
            self.relation_intent = RelationIntentConfig(**self.relation_intent)
        if isinstance(self.graph_recall, dict):
            self.graph_recall = GraphRelationRecallConfig(**self.graph_recall)
        if isinstance(self.posterior_graph, dict):
            self.posterior_graph = PosteriorGraphConfig(**self.posterior_graph)

        if not 0 <= self.alpha <= 1:
            raise ValueError(f"alpha必须在[0, 1]之间: {self.alpha}")

        if self.top_k_paragraphs <= 0:
            raise ValueError(f"top_k_paragraphs必须大于0: {self.top_k_paragraphs}")

        if self.top_k_relations <= 0:
            raise ValueError(f"top_k_relations必须大于0: {self.top_k_relations}")

        if self.top_k_final <= 0:
            raise ValueError(f"top_k_final必须大于0: {self.top_k_final}")
        if self.ppr_timeout_seconds <= 0:
            raise ValueError(f"ppr_timeout_seconds必须大于0: {self.ppr_timeout_seconds}")


@dataclass
class TemporalQueryOptions:
    """时序查询选项。"""

    time_from: Optional[float] = None
    time_to: Optional[float] = None
    person: Optional[str] = None
    source: Optional[str] = None
    allow_created_fallback: bool = True
    candidate_multiplier: int = 8
    max_scan: int = 1000


@dataclass
class RelationIntentConfig:
    """关系意图增强配置。"""

    enabled: bool = True
    alpha_override: float = 0.35
    relation_candidate_multiplier: int = 4
    preserve_top_relations: int = 3
    force_relation_sparse: bool = True
    pair_predicate_rerank_enabled: bool = True
    pair_predicate_limit: int = 3

    def __post_init__(self):
        self.alpha_override = min(1.0, max(0.0, float(self.alpha_override)))
        self.relation_candidate_multiplier = max(1, int(self.relation_candidate_multiplier))
        self.preserve_top_relations = max(0, int(self.preserve_top_relations))
        self.force_relation_sparse = bool(self.force_relation_sparse)
        self.pair_predicate_rerank_enabled = bool(self.pair_predicate_rerank_enabled)
        self.pair_predicate_limit = max(1, int(self.pair_predicate_limit))


@dataclass
class FusionConfig:
    """融合配置。"""

    method: str = "weighted_rrf"  # weighted_rrf | alpha_legacy
    rrf_k: int = 60
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    normalize_score: bool = True
    normalize_method: str = "minmax"

    def __post_init__(self):
        self.method = str(self.method or "weighted_rrf").strip().lower()
        self.normalize_method = str(self.normalize_method or "minmax").strip().lower()
        self.rrf_k = max(1, int(self.rrf_k))
        self.vector_weight = max(0.0, float(self.vector_weight))
        self.bm25_weight = max(0.0, float(self.bm25_weight))
        s = self.vector_weight + self.bm25_weight
        if s <= 0:
            self.vector_weight = 0.7
            self.bm25_weight = 0.3
        elif abs(s - 1.0) > 1e-8:
            self.vector_weight /= s
            self.bm25_weight /= s
