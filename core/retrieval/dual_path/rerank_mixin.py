"""PPR rerank helpers for DualPathRetriever."""

from __future__ import annotations

from typing import Dict, List

import asyncio

import numpy as np

from src.common.logger import get_logger

from ...utils.matcher import AhoCorasick
from .types import RetrievalResult

logger = get_logger("A_Memorix.DualPathRetriever")


class DualPathRerankMixin:
    """PageRank rerank and entity extraction."""
    async def _rerank_with_ppr(
        self,
        results: List[RetrievalResult],
        query: str,
    ) -> List[RetrievalResult]:
        """
        使用PageRank重排序结果 (异步 + 线程池)

        Args:
            results: 检索结果
            query: 查询文本

        Returns:
            重排序后的结果
        """
        # 从查询中提取实体
        entities = self._extract_entities(query)

        if not entities:
            logger.debug("未识别到实体，跳过PPR重排序")
            return results

        # 计算PPR分数 (放入线程池运行，避免阻塞主循环)
        ppr_timeout_s = max(0.1, float(getattr(self.config, "ppr_timeout_seconds", 1.5) or 1.5))
        try:
            async with self._ppr_semaphore:
                ppr_scores = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._ppr.compute,
                        personalization=entities,
                        normalize=True,
                    ),
                    timeout=ppr_timeout_s,
                )
        except asyncio.TimeoutError:
            logger.warning(
                "metric.ppr_timeout_skip_count=1 "
                f"timeout_s={ppr_timeout_s} "
                f"entities={len(entities)}"
            )
            return results
        except Exception as e:
            logger.warning(f"PPR 重排序失败，回退原排序: {e}")
            return results

        # 调整结果分数
        ppr_scores_by_name = {
            str(name).strip().lower(): float(score)
            for name, score in ppr_scores.items()
        }
        for result in results:
            if result.result_type == "paragraph":
                # 获取段落的实体
                para_entities = self.metadata_store.get_paragraph_entities(
                    result.hash_value
                )

                # 计算实体的平均PPR分数
                if para_entities:
                    entity_scores = []
                    for ent in para_entities:
                        ent_name = str(ent.get("name", "")).strip().lower()
                        if ent_name in ppr_scores_by_name:
                            entity_scores.append(ppr_scores_by_name[ent_name])

                    if entity_scores:
                        # 只使用命中的高价值图实体做正向增益，避免把原本高分的正确段落
                        # 因为“实体多但非全部命中”而反向压低。
                        focus_scores = sorted(entity_scores, reverse=True)[:2]
                        ppr_signal = float(np.mean(focus_scores))
                        boost_weight = 0.12 if len(focus_scores) >= 2 else 0.06
                        boost = ppr_signal * boost_weight

                        metadata = result.metadata if isinstance(result.metadata, dict) else {}
                        metadata["ppr_signal"] = round(ppr_signal, 4)
                        metadata["ppr_focus_entity_count"] = len(focus_scores)
                        metadata["ppr_boost"] = round(boost, 4)
                        result.metadata = metadata

                        result.score = float(result.score) + float(boost)

        # 重新排序
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def _extract_entities(self, text: str) -> Dict[str, float]:
        """
        从文本中提取实体（简化版本）

        Args:
            text: 输入文本

        Returns:
            实体字典 {实体名: 权重}
        """
        # 获取所有实体
        all_entities = self.graph_store.get_nodes()
        if not all_entities:
            return {}

        # 检查是否需要更新 Aho-Corasick 匹配器
        if self._ac_matcher is None or self._ac_nodes_count != len(all_entities):
            self._ac_matcher = AhoCorasick()
            for entity in all_entities:
                self._ac_matcher.add_pattern(entity.lower())
            self._ac_matcher.build()
            self._ac_nodes_count = len(all_entities)

        # 执行匹配
        text_lower = text.lower()
        stats = self._ac_matcher.find_all(text_lower)

        # 映射回原始名称并使用出现次数作为权重
        node_map = {node.lower(): node for node in all_entities}
        entities = {node_map[low_name]: float(count) for low_name, count in stats.items()}

        return entities

