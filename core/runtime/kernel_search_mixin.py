from __future__ import annotations

from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Sequence
from src.common.logger import get_logger
from ..retrieval import RetrievalResult, SparseBM25Config, SparseBM25Index
from ..utils.search_execution_service import SearchExecutionRequest, SearchExecutionService
from ..utils.time_parser import format_timestamp, parse_query_datetime_to_timestamp
from .kernel_types import KernelSearchRequest, NormalizedSearchTimeWindow

logger = get_logger("A_Memorix.SDKMemoryKernel")


class KernelSearchMixin:
    async def search_memory(self, request: KernelSearchRequest) -> Dict[str, Any]:
        if self._is_chat_filtered(
            respect_filter=request.respect_filter,
            stream_id=request.chat_id,
            group_id=request.group_id,
            user_id=request.user_id,
        ):
            return {"summary": "", "hits": [], "filtered": True}

        await self.initialize()
        assert self.retriever is not None
        assert self.episode_retriever is not None
        assert self.aggregate_query_service is not None

        mode = str(request.mode or "search").strip().lower() or "search"
        query = str(request.query or "").strip()
        limit = max(1, int(request.limit or 5))
        supported_modes = {"search", "time", "hybrid", "episode", "aggregate"}
        if mode not in supported_modes:
            return {
                "summary": "",
                "hits": [],
                "error": (
                    f"不支持的检索模式: {mode}（仅支持 search/time/hybrid/episode/aggregate，"
                    "semantic 已移除）"
                ),
            }
        try:
            time_window = self._normalize_search_time_window(request.time_start, request.time_end)
        except ValueError as exc:
            return {"summary": "", "hits": [], "error": str(exc)}

        if mode == "episode":
            rows = await self.episode_retriever.query(
                query=query,
                top_k=limit,
                time_from=time_window.numeric_start,
                time_to=time_window.numeric_end,
                person=request.person_id or None,
                source=self._chat_source(request.chat_id),
            )
            hits = self._filter_episode_hits([self._episode_hit(row) for row in rows])
            return {"summary": self._summary(hits), "hits": hits}

        if mode == "aggregate":
            payload = await self.aggregate_query_service.execute(
                query=query,
                top_k=limit,
                mix=True,
                mix_top_k=limit,
                time_from=time_window.query_start,
                time_to=time_window.query_end,
                search_runner=lambda: self._aggregate_search(query, limit, request),
                time_runner=lambda: self._aggregate_time(query, limit, request, time_window),
                episode_runner=lambda: self._aggregate_episode(query, limit, request, time_window),
            )
            hits = [dict(item) for item in payload.get("mixed_results", []) if isinstance(item, dict)]
            for item in hits:
                item.setdefault("metadata", {})
            filtered = self._filter_hits(hits, request.person_id)
            filtered = self._filter_user_visible_hits(filtered)
            return {"summary": self._summary(filtered), "hits": filtered}

        query_type = mode
        runtime_config = self._build_runtime_config()
        result = await SearchExecutionService.execute(
            retriever=self.retriever,
            threshold_filter=self.threshold_filter,
            plugin_config=runtime_config,
            request=SearchExecutionRequest(
                caller="sdk_memory_kernel",
                stream_id=str(request.chat_id or "") or None,
                group_id=str(request.group_id or "") or None,
                user_id=str(request.user_id or "") or None,
                query_type=query_type,
                query=query,
                top_k=limit,
                time_from=time_window.query_start,
                time_to=time_window.query_end,
                person=str(request.person_id or "") or None,
                source=self._chat_source(request.chat_id),
                use_threshold=True,
                enable_ppr=bool(self._cfg("retrieval.enable_ppr", True)),
            ),
            enforce_chat_filter=bool(request.respect_filter),
            reinforce_access=True,
        )
        if not result.success:
            return {"summary": "", "hits": [], "error": result.error}
        if result.chat_filtered:
            return {"summary": "", "hits": [], "filtered": True}

        hits = [self._retrieval_result_hit(item) for item in result.results]
        filtered = self._filter_hits(hits, request.person_id)
        filtered = self._filter_user_visible_hits(filtered)
        return {"summary": self._summary(filtered), "hits": filtered}

    async def _aggregate_search(self, query: str, limit: int, request: KernelSearchRequest) -> Dict[str, Any]:
        result = await SearchExecutionService.execute(
            retriever=self.retriever,
            threshold_filter=self.threshold_filter,
            plugin_config=self._build_runtime_config(),
            request=SearchExecutionRequest(
                caller="sdk_memory_kernel.aggregate",
                stream_id=str(request.chat_id or "") or None,
                query_type="search",
                query=query,
                top_k=limit,
                person=str(request.person_id or "") or None,
                source=self._chat_source(request.chat_id),
                use_threshold=True,
                enable_ppr=bool(self._cfg("retrieval.enable_ppr", True)),
            ),
            enforce_chat_filter=False,
            reinforce_access=True,
        )
        hits = [self._retrieval_result_hit(item) for item in result.results] if result.success else []
        return {"success": result.success, "results": hits, "count": len(hits), "query_type": "search", "error": result.error}

    async def _aggregate_time(
        self,
        query: str,
        limit: int,
        request: KernelSearchRequest,
        time_window: NormalizedSearchTimeWindow,
    ) -> Dict[str, Any]:
        result = await SearchExecutionService.execute(
            retriever=self.retriever,
            threshold_filter=self.threshold_filter,
            plugin_config=self._build_runtime_config(),
            request=SearchExecutionRequest(
                caller="sdk_memory_kernel.aggregate",
                stream_id=str(request.chat_id or "") or None,
                query_type="time",
                query=query,
                top_k=limit,
                time_from=time_window.query_start,
                time_to=time_window.query_end,
                person=str(request.person_id or "") or None,
                source=self._chat_source(request.chat_id),
                use_threshold=True,
                enable_ppr=bool(self._cfg("retrieval.enable_ppr", True)),
            ),
            enforce_chat_filter=False,
            reinforce_access=True,
        )
        hits = [self._retrieval_result_hit(item) for item in result.results] if result.success else []
        return {"success": result.success, "results": hits, "count": len(hits), "query_type": "time", "error": result.error}

    async def _aggregate_episode(
        self,
        query: str,
        limit: int,
        request: KernelSearchRequest,
        time_window: NormalizedSearchTimeWindow,
    ) -> Dict[str, Any]:
        assert self.episode_retriever
        rows = await self.episode_retriever.query(
            query=query,
            top_k=limit,
            time_from=time_window.numeric_start,
            time_to=time_window.numeric_end,
            person=request.person_id or None,
            source=self._chat_source(request.chat_id),
        )
        hits = self._filter_episode_hits([self._episode_hit(row) for row in rows])
        return {"success": True, "results": hits, "count": len(hits), "query_type": "episode"}

    def _filter_episode_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.metadata_store is None or not self._feedback_cfg_episode_query_block_enabled():
            return hits
        filtered: List[Dict[str, Any]] = []
        for item in hits:
            if str(item.get("type", "") or "").strip() != "episode":
                filtered.append(item)
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            source = str(metadata.get("source", "") or item.get("source", "") or "").strip()
            if source and self.metadata_store.is_episode_source_query_blocked(source):
                continue
            filtered.append(item)
        return filtered

    def _filter_user_visible_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._filter_active_relation_hits(self._filter_episode_hits(hits))

    @staticmethod
    def _graph_search_match_rank(value: str, keyword: str) -> Optional[int]:
        token = str(value or "").strip().lower()
        if not token or not keyword:
            return None
        if token == keyword:
            return 0
        if token.startswith(keyword):
            return 1
        if keyword in token:
            return 2
        return None

    @classmethod
    def _pick_graph_search_match(
        cls,
        fields: Sequence[tuple[str, str]],
        keyword: str,
    ) -> Optional[tuple[str, str, int]]:
        best_match: Optional[tuple[str, str, int]] = None
        for field, raw_value in fields:
            value = str(raw_value or "").strip()
            if not value:
                continue
            rank = cls._graph_search_match_rank(value, keyword)
            if rank is None:
                continue
            if best_match is None or rank < best_match[2]:
                best_match = (field, value, rank)
        return best_match

    def _search_graph(self, *, query: str, limit: int) -> Dict[str, Any]:
        assert self.metadata_store is not None
        token = str(query or "").strip()
        normalized_query = token.lower()
        safe_limit = max(1, int(limit or 50))
        if not token:
            return {
                "success": False,
                "query": token,
                "limit": safe_limit,
                "count": 0,
                "items": [],
                "error": "query 不能为空",
            }

        like_keyword = f"%{normalized_query}%"
        entity_rows = self.metadata_store.query(
            """
            SELECT hash, name, appearance_count, created_at
            FROM entities
            WHERE (is_deleted IS NULL OR is_deleted = 0)
              AND (
                LOWER(COALESCE(name, '')) LIKE ?
                OR LOWER(COALESCE(hash, '')) LIKE ?
              )
            """,
            (like_keyword, like_keyword),
        )

        relation_rows = self.metadata_store.query(
            """
            SELECT hash, subject, predicate, object, confidence, created_at
            FROM relations
            WHERE (is_inactive IS NULL OR is_inactive = 0)
              AND (
                LOWER(COALESCE(subject, '')) LIKE ?
                OR LOWER(COALESCE(object, '')) LIKE ?
                OR LOWER(COALESCE(predicate, '')) LIKE ?
                OR LOWER(COALESCE(hash, '')) LIKE ?
              )
            """,
            (like_keyword, like_keyword, like_keyword, like_keyword),
        )

        entity_items: List[Dict[str, Any]] = []
        seen_entity_keys: set[str] = set()
        for row in entity_rows:
            name = str(row.get("name", "") or "").strip()
            hash_value = str(row.get("hash", "") or "").strip()
            match = self._pick_graph_search_match(
                [("name", name), ("hash", hash_value)],
                normalized_query,
            )
            if match is None:
                continue
            dedupe_key = hash_value or f"name:{name.lower()}"
            if dedupe_key in seen_entity_keys:
                continue
            seen_entity_keys.add(dedupe_key)
            matched_field, matched_value, rank = match
            entity_items.append(
                {
                    "type": "entity",
                    "title": name or hash_value,
                    "matched_field": matched_field,
                    "matched_value": matched_value,
                    "entity_name": name or hash_value,
                    "entity_hash": hash_value,
                    "appearance_count": int(row.get("appearance_count", 0) or 0),
                    "_rank": rank,
                }
            )

        relation_items: List[Dict[str, Any]] = []
        seen_relation_keys: set[str] = set()
        for row in relation_rows:
            subject = str(row.get("subject", "") or "").strip()
            predicate = str(row.get("predicate", "") or "").strip()
            obj = str(row.get("object", "") or "").strip()
            relation_hash = str(row.get("hash", "") or "").strip()
            match = self._pick_graph_search_match(
                [
                    ("subject", subject),
                    ("object", obj),
                    ("predicate", predicate),
                    ("hash", relation_hash),
                ],
                normalized_query,
            )
            if match is None:
                continue
            dedupe_key = relation_hash or f"{subject.lower()}|{predicate.lower()}|{obj.lower()}"
            if dedupe_key in seen_relation_keys:
                continue
            seen_relation_keys.add(dedupe_key)
            matched_field, matched_value, rank = match
            relation_items.append(
                {
                    "type": "relation",
                    "title": self._format_relation_text(subject, predicate, obj),
                    "matched_field": matched_field,
                    "matched_value": matched_value,
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "relation_hash": relation_hash,
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                    "created_at": float(row.get("created_at", 0.0) or 0.0),
                    "_rank": rank,
                }
            )

        items = entity_items + relation_items
        items.sort(
            key=lambda item: (
                int(item["_rank"]) if item.get("_rank") is not None else 99,
                0 if str(item.get("type", "") or "") == "entity" else 1,
                -int(item.get("appearance_count", 0) or 0)
                if str(item.get("type", "") or "") == "entity"
                else -float(item.get("confidence", 0.0) or 0.0),
                0.0 if str(item.get("type", "") or "") == "entity" else -float(item.get("created_at", 0.0) or 0.0),
                str(item.get("entity_name", item.get("subject", "")) or "").lower(),
                str(item.get("predicate", "") or "").lower(),
                str(item.get("object", "") or "").lower(),
                str(item.get("entity_hash", item.get("relation_hash", "")) or "").lower(),
            )
        )

        normalized_items: List[Dict[str, Any]] = []
        for item in items[:safe_limit]:
            normalized = dict(item)
            normalized.pop("_rank", None)
            normalized_items.append(normalized)

        return {
            "success": True,
            "query": token,
            "limit": safe_limit,
            "count": len(normalized_items),
            "items": normalized_items,
        }

    @classmethod
    def _normalize_search_time_bound(cls, value: Any, *, is_end: bool) -> tuple[Optional[float], Optional[str]]:
        if value in {None, ""}:
            return None, None
        if isinstance(value, (int, float)):
            ts = float(value)
            return ts, format_timestamp(ts)

        text = str(value or "").strip()
        if not text:
            return None, None

        numeric = cls._optional_float(text)
        if numeric is not None:
            return numeric, format_timestamp(numeric)

        try:
            ts = parse_query_datetime_to_timestamp(text, is_end=is_end)
        except ValueError as exc:
            raise ValueError(f"时间参数错误: {exc}") from exc
        return ts, text

    @classmethod
    def _normalize_search_time_window(cls, time_start: Any, time_end: Any) -> NormalizedSearchTimeWindow:
        numeric_start, query_start = cls._normalize_search_time_bound(time_start, is_end=False)
        numeric_end, query_end = cls._normalize_search_time_bound(time_end, is_end=True)
        if numeric_start is not None and numeric_end is not None and numeric_start > numeric_end:
            raise ValueError("时间参数错误: time_start 不能晚于 time_end")
        return NormalizedSearchTimeWindow(
            numeric_start=numeric_start,
            numeric_end=numeric_end,
            query_start=query_start,
            query_end=query_end,
        )

    @staticmethod
    def _retrieval_result_hit(item: RetrievalResult) -> Dict[str, Any]:
        payload = item.to_dict()
        return {
            "hash": payload.get("hash", ""),
            "content": payload.get("content", ""),
            "score": payload.get("score", 0.0),
            "type": payload.get("type", ""),
            "source": payload.get("source", ""),
            "metadata": payload.get("metadata", {}) or {},
        }

    @staticmethod
    def _episode_hit(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "episode",
            "episode_id": str(row.get("episode_id", "") or ""),
            "title": str(row.get("title", "") or ""),
            "content": str(row.get("summary", "") or ""),
            "score": float(row.get("lexical_score", 0.0) or 0.0),
            "source": "episode",
            "metadata": {
                "participants": row.get("participants", []) or [],
                "keywords": row.get("keywords", []) or [],
                "source": row.get("source"),
                "event_time_start": row.get("event_time_start"),
                "event_time_end": row.get("event_time_end"),
            },
        }

    @staticmethod
    def _summary(hits: Sequence[Dict[str, Any]]) -> str:
        if not hits:
            return ""
        lines = []
        for index, item in enumerate(hits[:5], start=1):
            content = str(item.get("content", "") or "").strip().replace("\n", " ")
            lines.append(f"{index}. {(content[:120] + '...') if len(content) > 120 else content}")
        return "\n".join(lines)

    @staticmethod
    def _filter_hits(hits: List[Dict[str, Any]], person_id: str) -> List[Dict[str, Any]]:
        if not person_id:
            return hits
        filtered = []
        for item in hits:
            metadata = item.get("metadata", {}) or {}
            if person_id in (metadata.get("person_ids", []) or []):
                filtered.append(item)
                continue
            if person_id and person_id in str(item.get("content", "") or ""):
                filtered.append(item)
        return filtered or hits

    def _filter_active_relation_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.metadata_store is None:
            return hits
        relation_hashes: List[str] = []
        paragraph_relation_cache: Dict[str, List[str]] = {}
        paragraph_hashes: List[str] = []
        seen_relation_hashes: set[str] = set()

        for item in hits:
            item_type = str(item.get("type", "") or "").strip()
            item_hash = str(item.get("hash", "") or "").strip()
            if item_type == "relation" and item_hash and item_hash not in seen_relation_hashes:
                seen_relation_hashes.add(item_hash)
                relation_hashes.append(item_hash)
                continue
            if item_type != "paragraph" or not item_hash:
                continue
            paragraph_hashes.append(item_hash)
            linked_relations = self.metadata_store.get_paragraph_relations(item_hash)
            linked_hashes: List[str] = []
            for relation in linked_relations:
                linked_hash = str(relation.get("hash", "") or "").strip()
                if not linked_hash or linked_hash in seen_relation_hashes:
                    continue
                seen_relation_hashes.add(linked_hash)
                relation_hashes.append(linked_hash)
                linked_hashes.append(linked_hash)
            if linked_hashes:
                paragraph_relation_cache[item_hash] = linked_hashes

        marks_by_paragraph, _ = self._load_paragraph_stale_marks(paragraph_hashes)
        stale_relation_hashes = self._tokens(
            mark.get("relation_hash", "")
            for marks in marks_by_paragraph.values()
            for mark in marks
            if isinstance(mark, dict)
        )
        for relation_hash in stale_relation_hashes:
            if relation_hash in seen_relation_hashes:
                continue
            seen_relation_hashes.add(relation_hash)
            relation_hashes.append(relation_hash)

        if not relation_hashes and not marks_by_paragraph:
            return hits

        status_map = self.metadata_store.get_relation_status_batch(relation_hashes)
        filtered: List[Dict[str, Any]] = []
        for item in hits:
            item_type = str(item.get("type", "") or "").strip()
            if item_type == "paragraph":
                paragraph_hash = str(item.get("hash", "") or "").strip()
                if self._paragraph_hidden_by_stale_marks(
                    paragraph_hash,
                    marks_by_paragraph=marks_by_paragraph,
                    relation_status_map=status_map,
                ):
                    continue
                linked_hashes = paragraph_relation_cache.get(paragraph_hash, [])
                if not linked_hashes:
                    filtered.append(item)
                    continue
                if any(
                    not bool((status_map.get(linked_hash) or {}).get("is_inactive"))
                    for linked_hash in linked_hashes
                ):
                    filtered.append(item)
                continue
            if item_type != "relation":
                filtered.append(item)
                continue
            hash_value = str(item.get("hash", "") or "").strip()
            status = status_map.get(hash_value) if isinstance(status_map, dict) else None
            if status is None:
                continue
            if bool(status.get("is_inactive")):
                continue
            filtered.append(item)
        return filtered

    def _resolve_relation_hashes(self, target: str) -> List[str]:
        assert self.metadata_store
        token = str(target or "").strip()
        if not token:
            return []
        if len(token) == 64 and all(ch in "0123456789abcdef" for ch in token.lower()):
            return [token]
        hashes = self.metadata_store.search_relation_hashes_by_text(token, limit=10)
        if hashes:
            return hashes
        return [
            str(row.get("hash", "") or "")
            for row in self.metadata_store.get_relations(subject=token)[:10]
            if str(row.get("hash", "")).strip()
        ]

    def _resolve_deleted_relation_hashes(self, target: str) -> List[str]:
        assert self.metadata_store
        token = str(target or "").strip()
        if not token:
            return []
        if len(token) == 64 and all(ch in "0123456789abcdef" for ch in token.lower()):
            return [token]
        return self.metadata_store.search_deleted_relation_hashes_by_text(token, limit=10)
