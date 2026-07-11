from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Sequence
from json_repair import repair_json
from src.common.logger import get_logger
from src.config.config import global_config
from src.services import message_service as message_api
from src.services.llm_service import LLMServiceClient
from ..utils.hash import compute_hash, normalize_text
from ..utils.person_profile_service import PersonProfileService

logger = get_logger("A_Memorix.SDKMemoryKernel")


class KernelFeedbackTaskMixin:
    def _relation_status_is_inactive(status: Optional[Dict[str, Any]]) -> bool:
        if status is None:
            return True
        return bool(status.get("is_inactive"))

    def _load_paragraph_stale_marks(
        self,
        paragraph_hashes: Sequence[str],
    ) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
        if self.metadata_store is None:
            return {}, {}
        normalized = self._tokens(paragraph_hashes)
        if not normalized:
            return {}, {}
        marks_by_paragraph = self.metadata_store.get_paragraph_stale_relation_marks_batch(normalized)
        relation_hashes = self._tokens(
            mark.get("relation_hash", "")
            for marks in marks_by_paragraph.values()
            for mark in marks
            if isinstance(mark, dict)
        )
        status_map = self.metadata_store.get_relation_status_batch(relation_hashes) if relation_hashes else {}
        return marks_by_paragraph, status_map

    def _paragraph_hidden_by_stale_marks(
        self,
        paragraph_hash: str,
        *,
        marks_by_paragraph: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        relation_status_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> bool:
        token = str(paragraph_hash or "").strip()
        if not token or self.metadata_store is None or not self._feedback_cfg_paragraph_hard_filter_enabled():
            return False

        marks_map = marks_by_paragraph if isinstance(marks_by_paragraph, dict) else {}
        status_map = relation_status_map if isinstance(relation_status_map, dict) else {}
        if not marks_map:
            marks_map, status_map = self._load_paragraph_stale_marks([token])
        elif not status_map:
            relation_hashes = self._tokens(
                mark.get("relation_hash", "")
                for mark in marks_map.get(token, [])
                if isinstance(mark, dict)
            )
            status_map = self.metadata_store.get_relation_status_batch(relation_hashes) if relation_hashes else {}

        for mark in marks_map.get(token, []):
            relation_hash = str((mark or {}).get("relation_hash", "") or "").strip()
            if not relation_hash:
                continue
            if self._relation_status_is_inactive(status_map.get(relation_hash)):
                return True
        return False

    def _resolve_feedback_related_person_ids(
        self,
        *,
        old_relation_rows: Sequence[Dict[str, Any]],
        corrected_relations: Sequence[Dict[str, Any]],
    ) -> List[str]:
        candidates = self._tokens(
            value
            for row in list(old_relation_rows) + list(corrected_relations)
            if isinstance(row, dict)
            for value in (row.get("subject"), row.get("object"))
        )
        resolved: List[str] = []
        seen = set()
        for candidate in candidates:
            person_id = PersonProfileService.resolve_person_id(candidate)
            if not person_id or person_id in seen:
                continue
            seen.add(person_id)
            resolved.append(person_id)
        return resolved

    def _mark_feedback_stale_paragraphs(
        self,
        *,
        task_id: int,
        query_tool_id: str,
        relation_hashes: Sequence[str],
        reason: str,
    ) -> Dict[str, List[str]]:
        if self.metadata_store is None or not self._feedback_cfg_paragraph_mark_enabled():
            return {}

        relation_tokens = self._tokens(relation_hashes)
        paragraph_map = self.metadata_store.get_paragraph_hashes_by_relation_hashes(relation_tokens)
        for relation_hash, paragraph_hashes in paragraph_map.items():
            for paragraph_hash in paragraph_hashes:
                self.metadata_store.upsert_paragraph_stale_relation_mark(
                    paragraph_hash=paragraph_hash,
                    relation_hash=relation_hash,
                    query_tool_id=query_tool_id,
                    task_id=task_id,
                    reason=reason,
                )
        return paragraph_map

    def _enqueue_feedback_episode_rebuilds(
        self,
        *,
        paragraph_hashes: Sequence[str],
        session_id: str,
        include_correction_source: bool,
    ) -> List[str]:
        if self.metadata_store is None or not self._feedback_cfg_episode_rebuild_enabled():
            return []

        sources = self._tokens(
            row.get("source", "")
            for row in self._load_paragraph_rows(paragraph_hashes)
            if isinstance(row, dict)
        )
        correction_source = self._chat_source(session_id)
        if include_correction_source and correction_source:
            sources = self._merge_tokens(sources, [correction_source])

        queued: List[str] = []
        for source in sources:
            if self.metadata_store.enqueue_episode_source_rebuild(source, reason="feedback_correction"):
                queued.append(source)
        return queued

    def _enqueue_feedback_profile_refreshes(
        self,
        *,
        person_ids: Sequence[str],
        query_tool_id: str,
    ) -> List[str]:
        if self.metadata_store is None or not self._feedback_cfg_profile_refresh_enabled():
            return []
        queued: List[str] = []
        for person_id in self._tokens(person_ids):
            payload = self.metadata_store.enqueue_person_profile_refresh(
                person_id=person_id,
                reason="feedback_correction",
                source_query_tool_id=query_tool_id,
            )
            if isinstance(payload, dict):
                queued.append(person_id)
        return queued

    def _feedback_affected_counts(task: Dict[str, Any]) -> Dict[str, int]:
        decision_payload = task.get("decision_payload") if isinstance(task.get("decision_payload"), dict) else {}
        apply_result = decision_payload.get("apply_result") if isinstance(decision_payload.get("apply_result"), dict) else {}
        rollback_plan = task.get("rollback_plan") if isinstance(task.get("rollback_plan"), dict) else {}
        corrected_write = rollback_plan.get("corrected_write") if isinstance(rollback_plan.get("corrected_write"), dict) else {}
        return {
            "relations": len(list(apply_result.get("relation_hashes") or rollback_plan.get("forgotten_relations") or [])),
            "stale_paragraphs": len(list(apply_result.get("stale_paragraph_hashes") or rollback_plan.get("stale_marks") or [])),
            "episode_sources": len(list(apply_result.get("episode_rebuild_sources") or rollback_plan.get("episode_sources") or [])),
            "profile_person_ids": len(list(apply_result.get("profile_refresh_person_ids") or rollback_plan.get("profile_person_ids") or [])),
            "correction_paragraphs": len(list(corrected_write.get("paragraph_hashes") or [])),
            "corrected_relations": len(list(corrected_write.get("corrected_relations") or [])),
        }

    def _build_feedback_rollback_plan_summary(self, rollback_plan: Dict[str, Any]) -> Dict[str, Any]:
        corrected_write = rollback_plan.get("corrected_write") if isinstance(rollback_plan.get("corrected_write"), dict) else {}
        return {
            "forgotten_relations": list(rollback_plan.get("forgotten_relations") or []),
            "corrected_write": corrected_write,
            "stale_marks": list(rollback_plan.get("stale_marks") or []),
            "episode_sources": self._tokens(rollback_plan.get("episode_sources")),
            "profile_person_ids": self._tokens(rollback_plan.get("profile_person_ids")),
            "affected_counts": {
                "forgotten_relations": len(list(rollback_plan.get("forgotten_relations") or [])),
                "corrected_relations": len(list(corrected_write.get("corrected_relations") or [])),
                "correction_paragraphs": len(list(corrected_write.get("paragraph_hashes") or [])),
                "stale_marks": len(list(rollback_plan.get("stale_marks") or [])),
                "episode_sources": len(self._tokens(rollback_plan.get("episode_sources"))),
                "profile_person_ids": len(self._tokens(rollback_plan.get("profile_person_ids"))),
            },
        }

    def _build_feedback_task_summary(self, task: Dict[str, Any]) -> Dict[str, Any]:
        query_snapshot = task.get("query_snapshot") if isinstance(task.get("query_snapshot"), dict) else {}
        decision_payload = task.get("decision_payload") if isinstance(task.get("decision_payload"), dict) else {}
        return {
            "task_id": int(task.get("id", 0) or 0),
            "query_tool_id": str(task.get("query_tool_id", "") or "").strip(),
            "session_id": str(task.get("session_id", "") or "").strip(),
            "query_text": str(query_snapshot.get("query", "") or "").strip(),
            "query_timestamp": task.get("query_timestamp"),
            "task_status": str(task.get("status", "") or "").strip().lower(),
            "decision": str(decision_payload.get("decision", "") or "").strip().lower(),
            "decision_confidence": float(decision_payload.get("confidence", 0.0) or 0.0),
            "feedback_message_count": int(decision_payload.get("feedback_message_count", 0) or 0),
            "rollback_status": str(task.get("rollback_status", "") or "none").strip().lower() or "none",
            "affected_counts": self._feedback_affected_counts(task),
            "created_at": task.get("created_at"),
            "updated_at": task.get("updated_at"),
        }

    def _build_feedback_task_detail(self, task: Dict[str, Any]) -> Dict[str, Any]:
        detail = self._build_feedback_task_summary(task)
        detail.update(
            {
                "query_snapshot": task.get("query_snapshot") if isinstance(task.get("query_snapshot"), dict) else {},
                "decision_payload": task.get("decision_payload") if isinstance(task.get("decision_payload"), dict) else {},
                "rollback_plan_summary": self._build_feedback_rollback_plan_summary(
                    task.get("rollback_plan") if isinstance(task.get("rollback_plan"), dict) else {}
                ),
                "rollback_result": task.get("rollback_result") if isinstance(task.get("rollback_result"), dict) else {},
                "rollback_error": str(task.get("rollback_error", "") or "").strip(),
                "rollback_requested_by": str(task.get("rollback_requested_by", "") or "").strip(),
                "rollback_reason": str(task.get("rollback_reason", "") or "").strip(),
                "rollback_requested_at": task.get("rollback_requested_at"),
                "rolled_back_at": task.get("rolled_back_at"),
                "action_logs": self.metadata_store.list_feedback_action_logs(int(task.get("id", 0) or 0))
                if self.metadata_store is not None
                else [],
            }
        )
        return detail

    def _soft_delete_feedback_correction_paragraphs(self, paragraph_hashes: Sequence[str]) -> Dict[str, Any]:
        assert self.metadata_store is not None
        hashes = self._tokens(paragraph_hashes)
        if not hashes:
            return {"deleted_hashes": [], "deleted_external_refs": []}

        paragraph_rows = {hash_value: self.metadata_store.get_paragraph(hash_value) for hash_value in hashes}
        self.metadata_store.mark_as_deleted(hashes, "paragraph")
        conn = self.metadata_store.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"DELETE FROM paragraph_entities WHERE paragraph_hash IN ({','.join(['?'] * len(hashes))})",
            tuple(hashes),
        )
        cursor.execute(
            f"DELETE FROM paragraph_relations WHERE paragraph_hash IN ({','.join(['?'] * len(hashes))})",
            tuple(hashes),
        )
        conn.commit()
        deleted_external_refs = self.metadata_store.delete_external_memory_refs_by_paragraphs(hashes)
        return {
            "deleted_hashes": hashes,
            "paragraph_rows": paragraph_rows,
            "deleted_external_refs": deleted_external_refs,
        }

    async def _rollback_feedback_task(
        self,
        *,
        task_id: int,
        requested_by: str,
        reason: str,
    ) -> Dict[str, Any]:
        await self.initialize()
        assert self.metadata_store is not None

        task = self.metadata_store.get_feedback_task_by_id(task_id)
        if task is None:
            return {"success": False, "error": "反馈纠错任务不存在"}
        if str(task.get("status", "") or "").strip().lower() != "applied":
            return {"success": False, "error": "仅 applied 的反馈纠错任务允许回退"}
        rollback_status = str(task.get("rollback_status", "") or "none").strip().lower()
        if rollback_status == "rolled_back":
            return {
                "success": True,
                "already_rolled_back": True,
                "task": self._build_feedback_task_detail(task),
                "result": task.get("rollback_result") if isinstance(task.get("rollback_result"), dict) else {},
            }
        if rollback_status == "running":
            return {"success": False, "error": "该反馈纠错任务正在回退中", "task": self._build_feedback_task_detail(task)}

        query_tool_id = str(task.get("query_tool_id", "") or "").strip()
        rollback_plan = task.get("rollback_plan") if isinstance(task.get("rollback_plan"), dict) else {}
        if not rollback_plan:
            running_task = self.metadata_store.mark_feedback_task_rollback_running(
                task_id=task_id,
                requested_by=requested_by,
                reason=reason,
            )
            if running_task is None:
                latest_task = self.metadata_store.get_feedback_task_by_id(task_id)
                latest_status = str((latest_task or {}).get("rollback_status", "") or "none").strip().lower()
                if latest_status == "running":
                    return {
                        "success": False,
                        "error": "该反馈纠错任务正在回退中",
                        "task": self._build_feedback_task_detail(latest_task) if isinstance(latest_task, dict) else None,
                    }
                if latest_status == "rolled_back":
                    return {
                        "success": True,
                        "already_rolled_back": True,
                        "task": self._build_feedback_task_detail(latest_task) if isinstance(latest_task, dict) else None,
                        "result": (latest_task or {}).get("rollback_result") if isinstance((latest_task or {}).get("rollback_result"), dict) else {},
                    }
                return {
                    "success": False,
                    "error": "无法进入回退状态",
                    "task": self._build_feedback_task_detail(latest_task) if isinstance(latest_task, dict) else None,
                }
            self.metadata_store.append_feedback_action_log(
                task_id=task_id,
                query_tool_id=query_tool_id,
                action_type="rollback_error",
                reason="rollback_plan_missing",
            )
            failed = self.metadata_store.finalize_feedback_task_rollback(
                task_id=task_id,
                rollback_status="error",
                rollback_error="rollback_plan_missing",
            )
            return {"success": False, "error": "缺少 rollback_plan，无法回退", "task": failed}

        running_task = self.metadata_store.mark_feedback_task_rollback_running(
            task_id=task_id,
            requested_by=requested_by,
            reason=reason,
        )
        if running_task is None:
            latest_task = self.metadata_store.get_feedback_task_by_id(task_id)
            latest_status = str((latest_task or {}).get("rollback_status", "") or "none").strip().lower()
            if latest_status == "running":
                return {
                    "success": False,
                    "error": "该反馈纠错任务正在回退中",
                    "task": self._build_feedback_task_detail(latest_task) if isinstance(latest_task, dict) else None,
                }
            if latest_status == "rolled_back":
                return {
                    "success": True,
                    "already_rolled_back": True,
                    "task": self._build_feedback_task_detail(latest_task) if isinstance(latest_task, dict) else None,
                    "result": (latest_task or {}).get("rollback_result") if isinstance((latest_task or {}).get("rollback_result"), dict) else {},
                }
            return {
                "success": False,
                "error": "无法进入回退状态",
                "task": self._build_feedback_task_detail(latest_task) if isinstance(latest_task, dict) else None,
            }

        result: Dict[str, Any] = {
            "task_id": task_id,
            "query_tool_id": query_tool_id,
            "restored_relation_hashes": [],
            "reverted_corrected_relation_hashes": [],
            "deleted_correction_paragraph_hashes": [],
            "cleared_stale_mark_count": 0,
            "episode_sources_queued": [],
            "profile_person_ids_queued": [],
            "warnings": [],
        }
        try:
            forgotten_relations = rollback_plan.get("forgotten_relations") if isinstance(rollback_plan.get("forgotten_relations"), list) else []
            for item in forgotten_relations:
                if not isinstance(item, dict):
                    continue
                relation_hash = str(item.get("hash", "") or "").strip()
                snapshot = item.get("before_status") if isinstance(item.get("before_status"), dict) else {}
                if not relation_hash or not snapshot:
                    continue
                before_status = self.metadata_store.get_relation_status_batch([relation_hash]).get(relation_hash, {})
                after_status = self.metadata_store.restore_relation_status_from_snapshot(relation_hash, snapshot)
                if after_status is None:
                    result["warnings"].append(f"restore_old_relation_failed:{relation_hash}")
                    continue
                result["restored_relation_hashes"].append(relation_hash)
                self.metadata_store.append_feedback_action_log(
                    task_id=task_id,
                    query_tool_id=query_tool_id,
                    action_type="rollback_restore_relation",
                    target_hash=relation_hash,
                    before_payload=before_status,
                    after_payload=after_status,
                    reason=reason,
                )

            corrected_write = rollback_plan.get("corrected_write") if isinstance(rollback_plan.get("corrected_write"), dict) else {}
            correction_paragraph_hashes = self._tokens(corrected_write.get("paragraph_hashes"))
            deleted_paragraphs = self._soft_delete_feedback_correction_paragraphs(correction_paragraph_hashes)
            result["deleted_correction_paragraph_hashes"] = deleted_paragraphs.get("deleted_hashes", [])
            paragraph_rows = deleted_paragraphs.get("paragraph_rows") if isinstance(deleted_paragraphs.get("paragraph_rows"), dict) else {}
            deleted_external_refs = deleted_paragraphs.get("deleted_external_refs") if isinstance(deleted_paragraphs.get("deleted_external_refs"), list) else []
            deleted_ref_map: Dict[str, List[Dict[str, Any]]] = {}
            for ref in deleted_external_refs:
                if not isinstance(ref, dict):
                    continue
                paragraph_hash = str(ref.get("paragraph_hash", "") or "").strip()
                if not paragraph_hash:
                    continue
                deleted_ref_map.setdefault(paragraph_hash, []).append(ref)
            for paragraph_hash in result["deleted_correction_paragraph_hashes"]:
                self.metadata_store.append_feedback_action_log(
                    task_id=task_id,
                    query_tool_id=query_tool_id,
                    action_type="rollback_delete_correction_paragraph",
                    target_hash=paragraph_hash,
                    before_payload={
                        "paragraph": paragraph_rows.get(paragraph_hash) if isinstance(paragraph_rows.get(paragraph_hash), dict) else {},
                        "external_refs": deleted_ref_map.get(paragraph_hash, []),
                    },
                    reason=reason,
                )

            corrected_relations = corrected_write.get("corrected_relations") if isinstance(corrected_write.get("corrected_relations"), list) else []
            for item in corrected_relations:
                if not isinstance(item, dict):
                    continue
                relation_hash = str(item.get("hash", "") or "").strip()
                if not relation_hash:
                    continue
                before_status = self.metadata_store.get_relation_status_batch([relation_hash]).get(relation_hash, {})
                if bool(item.get("existed_before")):
                    snapshot = item.get("before_status") if isinstance(item.get("before_status"), dict) else {}
                    after_status = self.metadata_store.restore_relation_status_from_snapshot(relation_hash, snapshot)
                else:
                    self.metadata_store.update_relations_protection([relation_hash], protected_until=0.0, is_pinned=False)
                    self.metadata_store.mark_relations_inactive([relation_hash], inactive_since=time.time())
                    after_status = self.metadata_store.get_relation_status_batch([relation_hash]).get(relation_hash)
                if after_status is None:
                    result["warnings"].append(f"revert_corrected_relation_failed:{relation_hash}")
                    continue
                result["reverted_corrected_relation_hashes"].append(relation_hash)
                self.metadata_store.append_feedback_action_log(
                    task_id=task_id,
                    query_tool_id=query_tool_id,
                    action_type="rollback_revert_corrected_relation",
                    target_hash=relation_hash,
                    before_payload=before_status,
                    after_payload=after_status,
                    reason=reason,
                )

            stale_marks_raw = rollback_plan.get("stale_marks") if isinstance(rollback_plan.get("stale_marks"), list) else []
            stale_marks: List[tuple[str, str]] = []
            for item in stale_marks_raw:
                if not isinstance(item, dict):
                    continue
                paragraph_hash = str(item.get("paragraph_hash", "") or "").strip()
                relation_hash = str(item.get("relation_hash", "") or "").strip()
                if not paragraph_hash or not relation_hash:
                    continue
                stale_marks.append((paragraph_hash, relation_hash))
            result["cleared_stale_mark_count"] = self.metadata_store.delete_paragraph_stale_relation_marks(stale_marks)
            for paragraph_hash, relation_hash in stale_marks:
                self.metadata_store.append_feedback_action_log(
                    task_id=task_id,
                    query_tool_id=query_tool_id,
                    action_type="rollback_clear_stale_mark",
                    target_hash=paragraph_hash,
                    after_payload={"relation_hash": relation_hash},
                    reason=reason,
                )

            for source in self._tokens(rollback_plan.get("episode_sources")):
                if self.metadata_store.enqueue_episode_source_rebuild(source, reason="feedback_correction_rollback"):
                    result["episode_sources_queued"].append(source)
                    self.metadata_store.append_feedback_action_log(
                        task_id=task_id,
                        query_tool_id=query_tool_id,
                        action_type="rollback_enqueue_episode_rebuild",
                        target_hash=source,
                        reason=reason,
                    )

            for person_id in self._tokens(rollback_plan.get("profile_person_ids")):
                payload = self.metadata_store.enqueue_person_profile_refresh(
                    person_id=person_id,
                    reason="feedback_correction_rollback",
                    source_query_tool_id=query_tool_id,
                )
                if not isinstance(payload, dict):
                    continue
                result["profile_person_ids_queued"].append(person_id)
                self.metadata_store.append_feedback_action_log(
                    task_id=task_id,
                    query_tool_id=query_tool_id,
                    action_type="rollback_enqueue_profile_refresh",
                    target_hash=person_id,
                    reason=reason,
                )

            self._rebuild_graph_from_metadata()
            self._persist()
            final_task = self.metadata_store.finalize_feedback_task_rollback(
                task_id=task_id,
                rollback_status="rolled_back",
                rollback_result=result,
            )
            return {"success": True, "result": result, "task": self._build_feedback_task_detail(final_task or running_task)}
        except Exception as exc:
            logger.warning(f"反馈纠错回退失败: task_id={task_id} err={exc}", exc_info=True)
            self.metadata_store.append_feedback_action_log(
                task_id=task_id,
                query_tool_id=query_tool_id,
                action_type="rollback_error",
                reason=str(exc),
                after_payload=result if result else None,
            )
            final_task = self.metadata_store.finalize_feedback_task_rollback(
                task_id=task_id,
                rollback_status="error",
                rollback_result=result if result else None,
                rollback_error=str(exc),
            )
            return {
                "success": False,
                "error": str(exc),
                "result": result,
                "task": self._build_feedback_task_detail(final_task or running_task),
            }

    async def _process_feedback_profile_refresh_batch(self, *, limit: int) -> Dict[str, Any]:
        if self.metadata_store is None or self.person_profile_service is None:
            return {"processed": 0, "refreshed": 0, "failed": 0, "items": [], "failures": []}

        rows = self.metadata_store.fetch_person_profile_refresh_batch(
            limit=max(1, int(limit or 1)),
            max_retry=max(1, int(self._cfg("person_profile.max_retry", 3) or 3)),
        )
        items: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        for row in rows:
            person_id = str(row.get("person_id", "") or "").strip()
            requested_at = row.get("requested_at")
            if not person_id:
                continue
            if not self.metadata_store.mark_person_profile_refresh_running(person_id, requested_at=requested_at):
                continue
            try:
                profile = await self.refresh_person_profile(
                    person_id,
                    limit=max(4, int(self._cfg("person_profile.top_k_evidence", 12) or 12)),
                    mark_active=False,
                )
                if isinstance(profile, dict) and bool(profile.get("success")):
                    self.metadata_store.mark_person_profile_refresh_done(person_id, requested_at=requested_at)
                    items.append(
                        {
                            "person_id": person_id,
                            "profile_version": int(profile.get("profile_version", 0) or 0),
                            "profile_source": str(profile.get("profile_source", "") or ""),
                        }
                    )
                else:
                    error = str((profile or {}).get("error", "") or "person profile refresh failed")
                    self.metadata_store.mark_person_profile_refresh_failed(person_id, error, requested_at=requested_at)
                    failures.append({"person_id": person_id, "error": error})
            except Exception as exc:
                error = str(exc)[:500]
                self.metadata_store.mark_person_profile_refresh_failed(person_id, error, requested_at=requested_at)
                failures.append({"person_id": person_id, "error": error})
        return {
            "processed": len(items) + len(failures),
            "refreshed": len(items),
            "failed": len(failures),
            "items": items,
            "failures": failures,
        }

    async def _process_feedback_episode_rebuild_batch(self, *, limit: int) -> Dict[str, Any]:
        if self.metadata_store is None or self.episode_service is None:
            return {"processed": 0, "rebuilt": 0, "failed": 0, "items": [], "failures": []}

        rows = self.metadata_store.fetch_episode_source_rebuild_batch(
            limit=max(1, int(limit or 1)),
            max_retry=max(1, int(self._cfg("episode.pending_max_retry", 3) or 3)),
        )
        items: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        for row in rows:
            source = str(row.get("source", "") or "").strip()
            requested_at = row.get("requested_at")
            if not source:
                continue
            if not self.metadata_store.mark_episode_source_running(source, requested_at=requested_at):
                continue
            try:
                result = await self.episode_service.rebuild_source(source)
                self.metadata_store.mark_episode_source_done(source, requested_at=requested_at)
                items.append(result if isinstance(result, dict) else {"source": source})
            except Exception as exc:
                error = str(exc)[:500]
                self.metadata_store.mark_episode_source_failed(source, error, requested_at=requested_at)
                failures.append({"source": source, "error": error})
        if items or failures:
            self._persist()
        return {
            "processed": len(items) + len(failures),
            "rebuilt": len(items),
            "failed": len(failures),
            "items": items,
            "failures": failures,
        }

    async def _feedback_correction_reconcile_loop(self) -> None:
        try:
            while not self._background_stopping:
                await asyncio.sleep(self._feedback_cfg_reconcile_interval_seconds())
                if self._background_stopping:
                    break
                if self.metadata_store is None or not self._feedback_cfg_enabled():
                    continue
                batch_size = self._feedback_cfg_reconcile_batch_size()
                if self._feedback_cfg_profile_refresh_enabled():
                    await self._process_feedback_profile_refresh_batch(limit=batch_size)
                if self._feedback_cfg_episode_rebuild_enabled():
                    await self._process_feedback_episode_rebuild_batch(limit=batch_size)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"feedback_correction_reconcile loop 异常: {exc}")
