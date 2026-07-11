"""Temporal filtering helpers for DualPathRetriever."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ...utils.time_parser import format_timestamp
from .types import RetrievalResult, TemporalQueryOptions


class DualPathTemporalMixin:
    """Temporal query filtering and sorting."""
    def _cap_temporal_scan_k(
        self,
        candidate_k: int,
        temporal: Optional[TemporalQueryOptions],
    ) -> int:
        """对 temporal 模式候选召回数应用 max_scan 上限。"""
        k = max(1, int(candidate_k))
        if temporal and temporal.max_scan and temporal.max_scan > 0:
            k = min(k, int(temporal.max_scan))
        return max(1, k)

    def _retrieve_temporal_only(
        self,
        temporal: TemporalQueryOptions,
        top_k: int,
    ) -> List[RetrievalResult]:
        """无语义 query 时，直接走时序索引查询。"""
        limit = self._cap_temporal_scan_k(
            top_k * max(1, temporal.candidate_multiplier),
            temporal,
        )
        paragraphs = self.metadata_store.query_paragraphs_temporal(
            start_ts=temporal.time_from,
            end_ts=temporal.time_to,
            person=temporal.person,
            source=temporal.source,
            limit=limit,
            allow_created_fallback=temporal.allow_created_fallback,
        )
        results: List[RetrievalResult] = []
        for para in paragraphs:
            time_meta = self._build_time_meta_from_paragraph(para, temporal=temporal)
            results.append(
                RetrievalResult(
                    hash_value=para["hash"],
                    content=para["content"],
                    score=1.0,
                    result_type="paragraph",
                    source="temporal_scan",
                    metadata={
                        "word_count": para.get("word_count", 0),
                        "time_meta": time_meta,
                    },
                )
            )

        results = self._sort_results_with_temporal(results, temporal)
        return results[:top_k]

    def _extract_effective_time(
        self,
        paragraph: Dict[str, Any],
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """提取段落有效时间区间与命中依据。"""
        event_time = paragraph.get("event_time")
        event_start = paragraph.get("event_time_start")
        event_end = paragraph.get("event_time_end")

        if event_start is not None or event_end is not None:
            effective_start = event_start if event_start is not None else (
                event_time if event_time is not None else event_end
            )
            effective_end = event_end if event_end is not None else (
                event_time if event_time is not None else event_start
            )
            return effective_start, effective_end, "event_time_range"

        if event_time is not None:
            return event_time, event_time, "event_time"

        allow_fallback = True
        if temporal is not None:
            allow_fallback = temporal.allow_created_fallback

        created_at = paragraph.get("created_at")
        if allow_fallback and created_at is not None:
            return created_at, created_at, "created_at_fallback"

        return None, None, None

    def _build_time_meta_from_paragraph(
        self,
        paragraph: Dict[str, Any],
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> Dict[str, Any]:
        """构建统一 time_meta 结构。"""
        effective_start, effective_end, match_basis = self._extract_effective_time(
            paragraph,
            temporal=temporal,
        )
        return {
            "event_time": paragraph.get("event_time"),
            "event_time_start": paragraph.get("event_time_start"),
            "event_time_end": paragraph.get("event_time_end"),
            "ingest_time": paragraph.get("created_at"),
            "time_granularity": paragraph.get("time_granularity"),
            "time_confidence": paragraph.get("time_confidence", 1.0),
            "effective_start": effective_start,
            "effective_end": effective_end,
            "effective_start_text": format_timestamp(effective_start),
            "effective_end_text": format_timestamp(effective_end),
            "match_basis": match_basis or "none",
        }

    def _matches_person_filter(self, paragraph_hash: str, person: Optional[str]) -> bool:
        if not person:
            return True
        target = person.strip().lower()
        if not target:
            return True
        para_entities = self.metadata_store.get_paragraph_entities(paragraph_hash)
        for ent in para_entities:
            name = str(ent.get("name", "")).strip().lower()
            if target in name:
                return True
        return False

    def _is_temporal_match(
        self,
        paragraph: Dict[str, Any],
        temporal: TemporalQueryOptions,
    ) -> bool:
        """判断段落是否命中时序筛选。"""
        if temporal.source and paragraph.get("source") != temporal.source:
            return False

        if not self._matches_person_filter(paragraph.get("hash", ""), temporal.person):
            return False

        effective_start, effective_end, _ = self._extract_effective_time(paragraph, temporal=temporal)
        if effective_start is None or effective_end is None:
            return False

        if temporal.time_from is not None and temporal.time_to is not None:
            return effective_end >= temporal.time_from and effective_start <= temporal.time_to
        if temporal.time_from is not None:
            return effective_end >= temporal.time_from
        if temporal.time_to is not None:
            return effective_start <= temporal.time_to
        return True

    def _apply_temporal_filter_to_paragraphs(
        self,
        results: List[RetrievalResult],
        temporal: Optional[TemporalQueryOptions],
    ) -> List[RetrievalResult]:
        if not temporal:
            return results

        filtered: List[RetrievalResult] = []
        for result in results:
            paragraph = self.metadata_store.get_paragraph(result.hash_value)
            if not paragraph:
                continue
            if not self._is_temporal_match(paragraph, temporal):
                continue
            result.metadata["time_meta"] = self._build_time_meta_from_paragraph(paragraph, temporal=temporal)
            filtered.append(result)

        return self._sort_results_with_temporal(filtered, temporal)

    def _best_supporting_time_meta(
        self,
        relation_hash: str,
        temporal: TemporalQueryOptions,
    ) -> Optional[Dict[str, Any]]:
        """获取关系在时序窗口内最优支撑段落的 time_meta。"""
        supports = self.metadata_store.get_paragraphs_by_relation(relation_hash)
        if not supports:
            return None

        best_meta: Optional[Dict[str, Any]] = None
        best_time = float("-inf")
        for para in supports:
            if not self._is_temporal_match(para, temporal):
                continue
            meta = self._build_time_meta_from_paragraph(para, temporal=temporal)
            eff = meta.get("effective_end")
            score = float(eff) if eff is not None else float("-inf")
            if score >= best_time:
                best_time = score
                best_meta = meta

        return best_meta

    def _apply_temporal_filter_to_relations(
        self,
        results: List[RetrievalResult],
        temporal: Optional[TemporalQueryOptions],
    ) -> List[RetrievalResult]:
        if not temporal:
            return results

        filtered: List[RetrievalResult] = []
        for result in results:
            meta = result.metadata.get("time_meta")
            if meta is None:
                meta = self._best_supporting_time_meta(result.hash_value, temporal)
                if meta is None:
                    continue
                result.metadata["time_meta"] = meta
            filtered.append(result)

        return self._sort_results_with_temporal(filtered, temporal)

    def _sort_results_with_temporal(
        self,
        results: List[RetrievalResult],
        temporal: TemporalQueryOptions,
    ) -> List[RetrievalResult]:
        """语义优先，时间次排序（新到旧）。"""
        del temporal  # temporal 保留给未来扩展，目前只使用结果内 time_meta

        def _temporal_key(item: RetrievalResult) -> float:
            time_meta = item.metadata.get("time_meta", {})
            effective = time_meta.get("effective_end")
            if effective is None:
                effective = time_meta.get("effective_start")
            if effective is None:
                return float("-inf")
            return float(effective)

        results.sort(key=lambda x: (x.score, _temporal_key(x)), reverse=True)
        return results

