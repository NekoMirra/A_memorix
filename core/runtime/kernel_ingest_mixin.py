from __future__ import annotations

from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Sequence
from src.common.logger import get_logger
from ..utils.hash import compute_hash, normalize_text

logger = get_logger("A_Memorix.SDKMemoryKernel")


class KernelIngestMixin:
    async def summarize_chat_stream(
        self,
        *,
        chat_id: str,
        context_length: Optional[int] = None,
        include_personality: Optional[bool] = None,
        time_end: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        await self.initialize()
        assert self.summary_importer
        success, detail = await self.summary_importer.import_from_stream(
            stream_id=str(chat_id or "").strip(),
            context_length=context_length,
            include_personality=include_personality,
            time_end=time_end,
            metadata=metadata,
        )
        if success:
            await self.rebuild_episodes_for_sources([self._build_source("chat_summary", chat_id, [])])
            self._persist()
        return {"success": bool(success), "detail": detail}

    async def ingest_summary(
        self,
        *,
        external_id: str,
        chat_id: str,
        text: str,
        participants: Optional[Sequence[str]] = None,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        respect_filter: bool = True,
        user_id: str = "",
        group_id: str = "",
    ) -> Dict[str, Any]:
        external_token = str(external_id or "").strip() or compute_hash(f"chat_summary:{chat_id}:{text}")
        if self._is_chat_filtered(
            respect_filter=respect_filter,
            stream_id=chat_id,
            group_id=group_id,
            user_id=user_id,
        ):
            return {
                "success": True,
                "stored_ids": [],
                "skipped_ids": [external_token],
                "detail": "chat_filtered",
            }

        summary_meta = dict(metadata or {})
        summary_meta.setdefault("kind", "chat_summary")
        if not str(text or "").strip() or bool(summary_meta.get("generate_from_chat", False)):
            result = await self.summarize_chat_stream(
                chat_id=chat_id,
                context_length=self._optional_int(summary_meta.get("context_length")),
                include_personality=summary_meta.get("include_personality"),
                time_end=time_end,
                metadata={
                    **summary_meta,
                    "external_id": external_token,
                    "chat_id": str(chat_id or "").strip(),
                    "source_type": "chat_summary",
                },
            )
            result.setdefault("external_id", external_id)
            result.setdefault("chat_id", chat_id)
            return result
        return await self.ingest_text(
            external_id=external_id,
            source_type="chat_summary",
            text=text,
            chat_id=chat_id,
            participants=participants,
            time_start=time_start,
            time_end=time_end,
            tags=tags,
            metadata=summary_meta,
            respect_filter=respect_filter,
            user_id=user_id,
            group_id=group_id,
        )

    async def ingest_text(
        self,
        *,
        external_id: str,
        source_type: str,
        text: str,
        chat_id: str = "",
        person_ids: Optional[Sequence[str]] = None,
        participants: Optional[Sequence[str]] = None,
        timestamp: Optional[float] = None,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entities: Optional[Sequence[str]] = None,
        relations: Optional[Sequence[Dict[str, Any]]] = None,
        respect_filter: bool = True,
        user_id: str = "",
        group_id: str = "",
    ) -> Dict[str, Any]:
        content = normalize_text(text)
        external_token = str(external_id or "").strip() or compute_hash(f"{source_type}:{chat_id}:{content}")
        if self._is_chat_filtered(
            respect_filter=respect_filter,
            stream_id=chat_id,
            group_id=group_id,
            user_id=user_id,
        ):
            return {
                "success": True,
                "stored_ids": [],
                "skipped_ids": [external_token],
                "detail": "chat_filtered",
            }

        await self.initialize()
        assert self.metadata_store is not None
        assert self.vector_store is not None
        assert self.graph_store is not None
        assert self.embedding_manager is not None
        assert self.relation_write_service is not None

        if not content:
            return {"stored_ids": [], "skipped_ids": [external_token], "reason": "empty_text"}

        existing_ref = self.metadata_store.get_external_memory_ref(external_token)
        if existing_ref:
            return {
                "stored_ids": [],
                "skipped_ids": [str(existing_ref.get("paragraph_hash", "") or "")],
                "reason": "exists",
            }

        person_tokens = self._tokens(person_ids)
        participant_tokens = self._tokens(participants)
        entity_tokens = self._merge_tokens(entities, person_tokens, participant_tokens)
        source = self._build_source(source_type, chat_id, person_tokens)
        paragraph_meta = dict(metadata or {})
        paragraph_meta.update(
            {
                "external_id": external_token,
                "source_type": str(source_type or "").strip(),
                "chat_id": str(chat_id or "").strip(),
                "person_ids": person_tokens,
                "participants": participant_tokens,
                "tags": self._tokens(tags),
            }
        )
        warnings: List[str] = []

        paragraph_hash = self.metadata_store.add_paragraph(
            content=content,
            source=source,
            metadata=paragraph_meta,
            knowledge_type=self._resolve_knowledge_type(source_type),
            time_meta=self._time_meta(timestamp, time_start, time_end),
        )
        vector_result = await self._write_paragraph_vector_or_enqueue(
            paragraph_hash=paragraph_hash,
            content=content,
            context="ingest_text",
        )
        warning = str(vector_result.get("warning", "") or "").strip()
        if warning:
            warnings.append(warning)

        for name in entity_tokens:
            self.metadata_store.add_entity(name=name, source_paragraph=paragraph_hash)

        stored_relations: List[str] = []
        for row in [dict(item) for item in (relations or []) if isinstance(item, dict)]:
            subject = str(row.get("subject", "") or "").strip()
            predicate = str(row.get("predicate", "") or "").strip()
            obj = str(row.get("object", "") or "").strip()
            if not (subject and predicate and obj):
                continue
            result = await self.relation_write_service.upsert_relation_with_vector(
                subject=subject,
                predicate=predicate,
                obj=obj,
                confidence=float(row.get("confidence", 1.0) or 1.0),
                source_paragraph=paragraph_hash,
                metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else {"external_id": external_token, "source_type": source_type},
                write_vector=self.relation_vectors_enabled,
            )
            self.metadata_store.link_paragraph_relation(paragraph_hash, result.hash_value)
            stored_relations.append(result.hash_value)

        self.metadata_store.upsert_external_memory_ref(
            external_id=external_token,
            paragraph_hash=paragraph_hash,
            source_type=source_type,
            metadata={"chat_id": chat_id, "person_ids": person_tokens},
        )
        self.metadata_store.enqueue_episode_pending(paragraph_hash, source=source)
        self._persist()
        await self.process_episode_pending_batch(
            limit=max(1, int(self._cfg("episode.pending_batch_size", 12))),
            max_retry=max(1, int(self._cfg("episode.pending_max_retry", 3))),
        )
        for person_id in person_tokens:
            self._mark_person_active(person_id)
            await self.refresh_person_profile(person_id)
        payload = {"stored_ids": [paragraph_hash, *stored_relations], "skipped_ids": []}
        if warnings:
            payload["warnings"] = warnings
            payload["detail"] = "vector_degraded_write"
        return payload

    async def process_episode_pending_batch(self, *, limit: int = 20, max_retry: int = 3) -> Dict[str, Any]:
        await self.initialize()
        assert self.metadata_store is not None
        assert self.episode_service is not None

        pending_rows = self.metadata_store.fetch_episode_pending_batch(limit=max(1, int(limit)), max_retry=max(1, int(max_retry)))
        if not pending_rows:
            return {"processed": 0, "episode_count": 0, "fallback_count": 0, "failed": 0}

        source_to_hashes: Dict[str, List[str]] = {}
        pending_hashes = [str(row.get("paragraph_hash", "") or "").strip() for row in pending_rows if str(row.get("paragraph_hash", "") or "").strip()]
        for row in pending_rows:
            paragraph_hash = str(row.get("paragraph_hash", "") or "").strip()
            source = str(row.get("source", "") or "").strip()
            if not paragraph_hash or not source:
                continue
            source_to_hashes.setdefault(source, []).append(paragraph_hash)

        if pending_hashes:
            self.metadata_store.mark_episode_pending_running(pending_hashes)

        result = await self.episode_service.process_pending_rows(pending_rows)
        done_hashes = [str(item or "").strip() for item in result.get("done_hashes", []) if str(item or "").strip()]
        failed_hashes = {
            str(hash_value or "").strip(): str(error or "").strip()
            for hash_value, error in (result.get("failed_hashes", {}) or {}).items()
            if str(hash_value or "").strip()
        }

        if done_hashes:
            self.metadata_store.mark_episode_pending_done(done_hashes)
        for hash_value, error in failed_hashes.items():
            self.metadata_store.mark_episode_pending_failed(hash_value, error)

        untouched = [hash_value for hash_value in pending_hashes if hash_value not in set(done_hashes) and hash_value not in failed_hashes]
        for hash_value in untouched:
            self.metadata_store.mark_episode_pending_failed(hash_value, "episode processing finished without explicit status")

        for source, paragraph_hashes in source_to_hashes.items():
            counts = self.metadata_store.get_episode_pending_status_counts(source)
            if counts.get("failed", 0) > 0:
                source_error = next(
                    (
                        failed_hashes.get(hash_value)
                        for hash_value in paragraph_hashes
                        if failed_hashes.get(hash_value)
                    ),
                    "episode pending source contains failed rows",
                )
                self.metadata_store.mark_episode_source_failed(source, str(source_error or "episode pending source contains failed rows"))
            elif counts.get("pending", 0) == 0 and counts.get("running", 0) == 0:
                self.metadata_store.mark_episode_source_done(source)

        self._persist()
        return {
            "processed": len(done_hashes) + len(failed_hashes),
            "episode_count": int(result.get("episode_count") or 0),
            "fallback_count": int(result.get("fallback_count") or 0),
            "failed": len(failed_hashes) + len(untouched),
            "group_count": int(result.get("group_count") or 0),
            "missing_count": int(result.get("missing_count") or 0),
        }

    @staticmethod
    def _build_source(source_type: str, chat_id: str, person_ids: Sequence[str]) -> str:
        clean_type = str(source_type or "").strip() or "memory"
        if clean_type == "chat_summary" and chat_id:
            return f"chat_summary:{chat_id}"
        if clean_type == "person_fact" and person_ids:
            return f"person_fact:{person_ids[0]}"
        return f"{clean_type}:{chat_id}" if chat_id else clean_type

    @staticmethod
    def _chat_source(chat_id: str) -> Optional[str]:
        clean = str(chat_id or "").strip()
        return f"chat_summary:{clean}" if clean else None

    @staticmethod
    def _resolve_knowledge_type(source_type: str) -> str:
        clean_type = str(source_type or "").strip().lower()
        if clean_type == "person_fact":
            return "factual"
        if clean_type == "chat_summary":
            return "narrative"
        return "mixed"

    @staticmethod
    def _time_meta(timestamp: Optional[float], time_start: Optional[float], time_end: Optional[float]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if timestamp is not None:
            payload["event_time"] = float(timestamp)
        if time_start is not None:
            payload["event_time_start"] = float(time_start)
        if time_end is not None:
            payload["event_time_end"] = float(time_end)
        if payload:
            payload["time_granularity"] = "minute"
            payload["time_confidence"] = 0.95
        return payload
