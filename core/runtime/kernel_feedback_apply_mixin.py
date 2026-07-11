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


class KernelFeedbackApplyMixin:
    async def enqueue_feedback_task(
        self,
        *,
        query_tool_id: str,
        session_id: str,
        query_timestamp: Any = None,
        structured_content: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self._feedback_cfg_enabled():
            return {"success": False, "queued": False, "reason": "feedback_correction_disabled"}
        if self.metadata_store is None:
            return {"success": False, "queued": False, "reason": "metadata_store_unavailable"}

        clean_tool_id = str(query_tool_id or "").strip()
        clean_session_id = str(session_id or "").strip()
        if not clean_tool_id or not clean_session_id:
            return {"success": False, "queued": False, "reason": "missing_required_fields"}

        content = structured_content if isinstance(structured_content, dict) else {}
        hits = content.get("hits")
        if not isinstance(hits, list) or not hits:
            return {"success": False, "queued": False, "reason": "no_hits"}

        query_time = self._coerce_datetime(query_timestamp) or datetime.now()
        due_at = query_time + timedelta(hours=self._feedback_cfg_window_hours())
        saved = self.metadata_store.enqueue_feedback_task(
            query_tool_id=clean_tool_id,
            session_id=clean_session_id,
            query_timestamp=query_time.timestamp(),
            due_at=due_at.timestamp(),
            query_snapshot=content,
        )
        if not isinstance(saved, dict):
            return {"success": False, "queued": False, "reason": "db_save_failed"}

        logger.debug(
            "反馈纠错任务入队: query_tool_id=%s due_at=%s",
            clean_tool_id,
            due_at.isoformat(),
        )
        return {
            "success": True,
            "queued": True,
            "query_tool_id": clean_tool_id,
            "due_at": due_at.isoformat(),
            "task": saved,
        }

    def _extract_feedback_messages(
        *,
        session_id: str,
        query_time: datetime,
        due_time: datetime,
        max_messages: int,
    ) -> List[str]:
        raw_messages = message_api.get_messages_by_time_in_chat(
            chat_id=session_id,
            start_time=query_time.timestamp(),
            end_time=due_time.timestamp(),
            limit=max(1, int(max_messages) * 4),
            limit_mode="latest",
            filter_mai=True,
            filter_command=True,
        )
        collected: List[str] = []
        seen = set()
        for item in raw_messages:
            text = str(getattr(item, "processed_plain_text", "") or "").strip()
            if KernelFeedbackConfigMixin._feedback_noise(text):
                continue
            if text in seen:
                continue
            seen.add(text)
            collected.append(text)
        if len(collected) > max_messages:
            collected = collected[-max_messages:]
        return collected

    def _build_feedback_hit_briefs(self, hits: List[Dict[str, Any]], *, limit: int = 12) -> List[Dict[str, Any]]:
        briefs: List[Dict[str, Any]] = []
        for raw in hits[: max(1, int(limit))]:
            if not isinstance(raw, dict):
                continue
            metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
            subject = str(metadata.get("subject", "") or "").strip()
            predicate = str(metadata.get("predicate", "") or "").strip()
            obj = str(metadata.get("object", "") or "").strip()
            linked_relation_hashes: List[str] = []
            linked_relation_texts: List[str] = []

            item_type = str(raw.get("type", "") or "").strip()
            item_hash = str(raw.get("hash", "") or "").strip()
            if item_type == "paragraph" and item_hash and self.metadata_store is not None:
                linked_relations = self.metadata_store.get_paragraph_relations(item_hash)
                for relation in linked_relations:
                    relation_hash = str(relation.get("hash", "") or "").strip()
                    if not relation_hash or relation_hash in linked_relation_hashes:
                        continue
                    linked_relation_hashes.append(relation_hash)
                    rel_subject = str(relation.get("subject", "") or "").strip()
                    rel_predicate = str(relation.get("predicate", "") or "").strip()
                    rel_object = str(relation.get("object", "") or "").strip()
                    relation_text = self._format_relation_text(rel_subject, rel_predicate, rel_object)
                    if relation_text:
                        linked_relation_texts.append(relation_text)
                    if not (subject and predicate and obj):
                        subject = rel_subject
                        predicate = rel_predicate
                        obj = rel_object
            briefs.append(
                {
                    "hash": item_hash,
                    "type": item_type,
                    "content": str(raw.get("content", "") or "").strip(),
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "linked_relation_hashes": linked_relation_hashes[:6],
                    "linked_relation_texts": linked_relation_texts[:3],
                }
            )
        return briefs

    def _should_invoke_feedback_classifier(feedback_messages: List[str]) -> bool:
        if not feedback_messages:
            return False
        lowered = "\n".join(feedback_messages).lower()
        return any(token in lowered for token in KernelFeedbackConfigMixin._feedback_signal_tokens())

    async def _classify_feedback(
        self,
        *,
        query_tool_id: str,
        query_text: str,
        hit_briefs: List[Dict[str, Any]],
        feedback_messages: List[str],
    ) -> Dict[str, Any]:
        prompt = (
            "你是长期记忆纠错分类器。"
            "你会根据“记忆检索命中列表”和“用户后续反馈”判断是否需要修正记忆。"
            "请严格输出 JSON 对象，不要输出解释文字。\n\n"
            f"query_tool_id: {query_tool_id}\n"
            f"原查询: {query_text}\n"
            f"候选命中: {json.dumps(hit_briefs, ensure_ascii=False)}\n"
            f"反馈消息: {json.dumps(feedback_messages, ensure_ascii=False)}\n\n"
            "输出 JSON schema:\n"
            "{"
            "\"decision\":\"confirm|reject|correct|supplement|none\","
            "\"confidence\":0.0,"
            "\"target_hashes\":[\"命中列表中的 hash\"],"
            "\"corrected_relations\":[{\"subject\":\"\",\"predicate\":\"\",\"object\":\"\",\"confidence\":1.0}],"
            "\"reason\":\"\""
            "}\n"
            "约束:\n"
            "1. 只有当反馈明确指向错误时才输出 reject/correct。\n"
            "2. target_hashes 必须来自候选命中 hash。\n"
            "3. corrected_relations 仅在 decision=correct 时填写，且必须是明确三元组。\n"
            "4. 不确定时输出 decision=none, confidence<=0.5。"
        )
        try:
            if self._feedback_classifier is None:
                self._feedback_classifier = LLMServiceClient(
                    task_name="utils",
                    request_type="memory_feedback_correction",
                )
            response = await self._feedback_classifier.generate_response(prompt)
            payload = self._safe_json_loads(getattr(response, "response", ""))
        except Exception as exc:
            logger.warning(f"反馈分类器调用失败: {exc}")
            payload = {}
        return payload

    def _normalize_feedback_decision(
        payload: Dict[str, Any],
        *,
        hit_hashes: Sequence[str],
    ) -> Dict[str, Any]:
        allowed = {"confirm", "reject", "correct", "supplement", "none"}
        decision = str(payload.get("decision", "") or "").strip().lower()
        if decision not in allowed:
            decision = "none"
        try:
            confidence = float(payload.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))

        valid_hashes = {str(item or "").strip() for item in hit_hashes if str(item or "").strip()}
        target_hashes_raw = payload.get("target_hashes")
        if isinstance(target_hashes_raw, str):
            target_hashes_candidates = [target_hashes_raw]
        elif isinstance(target_hashes_raw, list):
            target_hashes_candidates = target_hashes_raw
        else:
            target_hashes_candidates = []
        target_hashes = [
            str(item or "").strip()
            for item in target_hashes_candidates
            if str(item or "").strip() in valid_hashes
        ]

        corrected_relations: List[Dict[str, Any]] = []
        raw_relations = payload.get("corrected_relations")
        if isinstance(raw_relations, list):
            for item in raw_relations:
                if not isinstance(item, dict):
                    continue
                subject = str(item.get("subject", "") or "").strip()
                predicate = str(item.get("predicate", "") or "").strip()
                obj = str(item.get("object", "") or "").strip()
                if not (subject and predicate and obj):
                    continue
                try:
                    rel_conf = float(item.get("confidence", 1.0) or 1.0)
                except (TypeError, ValueError):
                    rel_conf = 1.0
                corrected_relations.append(
                    {
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "confidence": min(1.0, max(0.0, rel_conf)),
                    }
                )
        corrected_relations = corrected_relations[:6]

        return {
            "decision": decision,
            "confidence": confidence,
            "target_hashes": target_hashes,
            "corrected_relations": corrected_relations,
            "reason": str(payload.get("reason", "") or "").strip(),
            "raw": payload,
        }

    def _feedback_apply_result_status(apply_result: Dict[str, Any]) -> str:
        if bool(apply_result.get("applied")):
            return "applied"

        reason = str(apply_result.get("reason", "") or "").strip().lower()
        if reason in {"low_confidence", "no_relation_targets"} or reason.startswith("decision_"):
            return "skipped"
        return "error"

    def _restore_feedback_relations_from_snapshots(
        self,
        *,
        task_id: int,
        query_tool_id: str,
        relation_hashes: Sequence[str],
        snapshots: Dict[str, Dict[str, Any]],
        current_statuses: Optional[Dict[str, Dict[str, Any]]] = None,
        reason: str,
    ) -> Dict[str, List[str]]:
        assert self.metadata_store is not None

        restored_hashes: List[str] = []
        failed_hashes: List[str] = []
        status_map = current_statuses if isinstance(current_statuses, dict) else {}

        for relation_hash in self._tokens(relation_hashes):
            snapshot = snapshots.get(relation_hash) if isinstance(snapshots, dict) else None
            if not isinstance(snapshot, dict) or not snapshot:
                failed_hashes.append(relation_hash)
                continue

            after_status = self.metadata_store.restore_relation_status_from_snapshot(relation_hash, snapshot)
            if after_status is None:
                failed_hashes.append(relation_hash)
                continue

            restored_hashes.append(relation_hash)
            self.metadata_store.append_feedback_action_log(
                task_id=task_id,
                query_tool_id=query_tool_id,
                action_type="compensate_restore_relation",
                target_hash=relation_hash,
                before_payload=status_map.get(relation_hash, {}),
                after_payload=after_status,
                reason=reason,
            )

        if restored_hashes or failed_hashes:
            self._rebuild_graph_from_metadata()
            self._persist()

        return {
            "restored_hashes": restored_hashes,
            "failed_hashes": failed_hashes,
        }

    async def _ingest_feedback_relations(
        self,
        *,
        query_tool_id: str,
        session_id: str,
        relation_hashes: List[str],
        corrected_relations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        supersedes_hash = relation_hashes[0] if relation_hashes else ""
        relation_rows: List[Dict[str, Any]] = []
        for row in corrected_relations:
            relation_rows.append(
                {
                    "subject": str(row.get("subject", "") or "").strip(),
                    "predicate": str(row.get("predicate", "") or "").strip(),
                    "object": str(row.get("object", "") or "").strip(),
                    "confidence": float(row.get("confidence", 1.0) or 1.0),
                    "metadata": {
                        "supersedes_hash": supersedes_hash,
                        "supersedes_hashes": relation_hashes,
                        "from_query_tool_id": query_tool_id,
                        "feedback_window": self._feedback_cfg_window_label(),
                    },
                }
            )
        plain_text = "；".join(
            f"{item['subject']} {item['predicate']} {item['object']}"
            for item in relation_rows
            if item.get("subject") and item.get("predicate") and item.get("object")
        )
        external_id = compute_hash(
            "feedback_correction:"
            + query_tool_id
            + ":"
            + json.dumps(relation_rows, ensure_ascii=False, sort_keys=True)
        )
        payload = await self.ingest_text(
            external_id=external_id,
            source_type="chat_summary",
            text=plain_text,
            chat_id=session_id,
            relations=relation_rows,
            metadata={
                "from_query_tool_id": query_tool_id,
                "feedback_window": self._feedback_cfg_window_label(),
                "supersedes_hashes": relation_hashes,
                "feedback_correction_source": True,
            },
            respect_filter=False,
        )
        if isinstance(payload, dict):
            stored_ids = self._tokens(payload.get("stored_ids"))
            corrected_relation_hashes = stored_ids[1:]
            payload["external_id"] = external_id
            payload["source"] = self._chat_source(session_id)
            payload["paragraph_hashes"] = stored_ids[:1]
            payload["corrected_relation_hashes"] = corrected_relation_hashes
            base_success = bool(payload.get("success")) if "success" in payload else True
            payload["success"] = base_success and bool(corrected_relation_hashes)
            if not payload["success"] and not str(payload.get("error", "") or "").strip():
                payload["error"] = "missing_corrected_relations"
            return payload
        return {"success": False, "error": "invalid_ingest_payload"}

    async def _apply_feedback_decision(
        self,
        *,
        task_id: int,
        query_tool_id: str,
        session_id: str,
        decision: Dict[str, Any],
        hit_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        threshold = self._feedback_cfg_auto_apply_threshold()
        confidence = float(decision.get("confidence", 0.0) or 0.0)
        if confidence < threshold:
            return {
                "applied": False,
                "reason": "low_confidence",
                "threshold": threshold,
                "confidence": confidence,
            }

        decision_type = str(decision.get("decision", "none") or "none").strip().lower()
        if decision_type not in {"reject", "correct"}:
            return {
                "applied": False,
                "reason": f"decision_{decision_type}_no_auto_apply",
            }

        target_hashes = [
            str(item or "").strip()
            for item in (decision.get("target_hashes") or [])
            if str(item or "").strip()
        ]
        relation_hashes = self._resolve_feedback_relation_hashes(
            target_hashes=target_hashes,
            hit_map=hit_map,
        )
        if not relation_hashes:
            return {
                "applied": False,
                "reason": "no_relation_targets",
            }

        corrected_relations = [
            dict(item)
            for item in (decision.get("corrected_relations") or [])
            if isinstance(item, dict)
        ]
        if decision_type == "correct" and not corrected_relations:
            return {
                "applied": False,
                "reason": "missing_corrected_relations",
                "relation_hashes": relation_hashes,
                "stale_paragraph_hashes": [],
                "episode_rebuild_sources": [],
                "profile_refresh_person_ids": [],
                "rollback_plan_summary": {},
            }

        assert self.metadata_store is not None
        old_relation_rows = self._query_relation_rows_by_hashes(relation_hashes, include_inactive=True)
        before_status = self.metadata_store.get_relation_status_batch(relation_hashes)
        forget_result = self._apply_v5_relation_action(action="forget", hashes=relation_hashes, strength=1.0)
        forget_success = bool(forget_result.get("success"))
        after_status = self.metadata_store.get_relation_status_batch(relation_hashes)
        for hash_value in relation_hashes:
            self.metadata_store.append_feedback_action_log(
                task_id=task_id,
                query_tool_id=query_tool_id,
                action_type="forget_relation",
                target_hash=hash_value,
                before_payload=before_status.get(hash_value) if isinstance(before_status, dict) else {},
                after_payload=after_status.get(hash_value) if isinstance(after_status, dict) else {},
                reason=str(decision.get("reason", "") or ""),
            )

        ingest_result = None
        corrected_relation_hash_candidates: List[str] = []
        corrected_relation_specs_by_hash: Dict[str, Dict[str, Any]] = {}
        if decision_type == "correct" and corrected_relations and self.metadata_store is not None:
            for item in corrected_relations:
                try:
                    relation_hash = self.metadata_store.compute_relation_hash(
                        str(item.get("subject", "") or "").strip(),
                        str(item.get("predicate", "") or "").strip(),
                        str(item.get("object", "") or "").strip(),
                    )
                except Exception:
                    continue
                if not relation_hash:
                    continue
                corrected_relation_hash_candidates.append(relation_hash)
                corrected_relation_specs_by_hash[relation_hash] = {
                    "subject": str(item.get("subject", "") or "").strip(),
                    "predicate": str(item.get("predicate", "") or "").strip(),
                    "object": str(item.get("object", "") or "").strip(),
                }
        corrected_relation_before_status = (
            self.metadata_store.get_relation_status_batch(corrected_relation_hash_candidates)
            if corrected_relation_hash_candidates
            else {}
        )
        if not forget_success:
            return {
                "applied": False,
                "reason": "forget_failed",
                "error": str(forget_result.get("error", "") or "forget_failed"),
                "forget": forget_result,
                "ingest": ingest_result,
                "relation_hashes": relation_hashes,
                "stale_paragraph_hashes": [],
                "episode_rebuild_sources": [],
                "profile_refresh_person_ids": [],
                "rollback_plan_summary": {},
            }

        stale_paragraph_map: Dict[str, List[str]] = {}
        stale_paragraph_hashes: List[str] = []
        episode_rebuild_sources: List[str] = []
        profile_refresh_person_ids: List[str] = []
        rollback_plan: Dict[str, Any] = {}
        if decision_type == "correct" and corrected_relations:
            ingest_result = await self._ingest_feedback_relations(
                query_tool_id=query_tool_id,
                session_id=session_id,
                relation_hashes=relation_hashes,
                corrected_relations=corrected_relations,
            )
            self.metadata_store.append_feedback_action_log(
                task_id=task_id,
                query_tool_id=query_tool_id,
                action_type="ingest_correction",
                target_hash=relation_hashes[0] if relation_hashes else "",
                before_payload={"target_hashes": relation_hashes},
                after_payload=ingest_result,
                reason=str(decision.get("reason", "") or ""),
            )

            ingest_success = bool((ingest_result or {}).get("success")) if isinstance(ingest_result, dict) else False
            if not ingest_success:
                compensation_result = self._restore_feedback_relations_from_snapshots(
                    task_id=task_id,
                    query_tool_id=query_tool_id,
                    relation_hashes=relation_hashes,
                    snapshots=before_status if isinstance(before_status, dict) else {},
                    current_statuses=after_status if isinstance(after_status, dict) else {},
                    reason=str(decision.get("reason", "") or "") or "feedback_correction_ingest_failed",
                )
                restore_failed_hashes = compensation_result.get("failed_hashes", [])
                return {
                    "applied": False,
                    "reason": "correction_restore_failed" if restore_failed_hashes else "correction_ingest_failed",
                    "error": str((ingest_result or {}).get("error", "") or "correction_ingest_failed"),
                    "forget": forget_result,
                    "ingest": ingest_result,
                    "relation_hashes": relation_hashes,
                    "stale_paragraph_hashes": [],
                    "episode_rebuild_sources": [],
                    "profile_refresh_person_ids": [],
                    "restored_relation_hashes": compensation_result.get("restored_hashes", []),
                    "restore_failed_hashes": restore_failed_hashes,
                    "rollback_plan_summary": {},
                }
        else:
            ingest_success = False

        applied = forget_success if decision_type == "reject" else (forget_success and ingest_success)
        if applied:
            stale_paragraph_map = self._mark_feedback_stale_paragraphs(
                task_id=task_id,
                query_tool_id=query_tool_id,
                relation_hashes=relation_hashes,
                reason=str(decision.get("reason", "") or "") or "feedback_correction",
            )
            stale_paragraph_hashes = self._merge_tokens(
                *[
                    paragraph_hashes
                    for paragraph_hashes in stale_paragraph_map.values()
                    if isinstance(paragraph_hashes, list)
                ]
            )
            episode_rebuild_sources = self._enqueue_feedback_episode_rebuilds(
                paragraph_hashes=stale_paragraph_hashes,
                session_id=session_id,
                include_correction_source=bool(ingest_success),
            )
            profile_refresh_person_ids = self._enqueue_feedback_profile_refreshes(
                person_ids=self._resolve_feedback_related_person_ids(
                    old_relation_rows=old_relation_rows,
                    corrected_relations=corrected_relations,
                ),
                query_tool_id=query_tool_id,
            )
            for relation_hash, paragraph_hashes in stale_paragraph_map.items():
                for paragraph_hash in paragraph_hashes:
                    self.metadata_store.append_feedback_action_log(
                        task_id=task_id,
                        query_tool_id=query_tool_id,
                        action_type="mark_stale_paragraph",
                        target_hash=paragraph_hash,
                        after_payload={"relation_hash": relation_hash},
                        reason=str(decision.get("reason", "") or ""),
                    )
            for source in episode_rebuild_sources:
                self.metadata_store.append_feedback_action_log(
                    task_id=task_id,
                    query_tool_id=query_tool_id,
                    action_type="enqueue_episode_rebuild",
                    target_hash=source,
                    reason=str(decision.get("reason", "") or ""),
                )
            for person_id in profile_refresh_person_ids:
                self.metadata_store.append_feedback_action_log(
                    task_id=task_id,
                    query_tool_id=query_tool_id,
                    action_type="enqueue_profile_refresh",
                    target_hash=person_id,
                    reason=str(decision.get("reason", "") or ""),
                )
            forgotten_relations = []
            for row in old_relation_rows:
                relation_hash = str(row.get("hash", "") or "").strip()
                if not relation_hash:
                    continue
                forgotten_relations.append(
                    {
                        "hash": relation_hash,
                        "subject": str(row.get("subject", "") or "").strip(),
                        "predicate": str(row.get("predicate", "") or "").strip(),
                        "object": str(row.get("object", "") or "").strip(),
                        "before_status": before_status.get(relation_hash) if isinstance(before_status, dict) else {},
                    }
                )

            corrected_write: Dict[str, Any] = {}
            if isinstance(ingest_result, dict):
                stored_relation_hashes = self._tokens(ingest_result.get("corrected_relation_hashes"))
                corrected_write = {
                    "external_id": str(ingest_result.get("external_id", "") or "").strip(),
                    "source": str(ingest_result.get("source", "") or "").strip(),
                    "paragraph_hashes": self._tokens(ingest_result.get("paragraph_hashes")),
                    "corrected_relation_hashes": stored_relation_hashes,
                    "corrected_relations": [
                        {
                            "hash": relation_hash,
                            **corrected_relation_specs_by_hash.get(relation_hash, {}),
                            "existed_before": relation_hash in corrected_relation_before_status,
                            "before_status": corrected_relation_before_status.get(relation_hash, {}),
                        }
                        for relation_hash in stored_relation_hashes
                    ],
                }

            rollback_plan = {
                "task_id": task_id,
                "query_tool_id": query_tool_id,
                "session_id": session_id,
                "decision_type": decision_type,
                "forgotten_relations": forgotten_relations,
                "corrected_write": corrected_write,
                "stale_marks": [
                    {"paragraph_hash": paragraph_hash, "relation_hash": relation_hash}
                    for relation_hash, paragraph_hashes in stale_paragraph_map.items()
                    for paragraph_hash in (paragraph_hashes or [])
                    if str(paragraph_hash or "").strip()
                ],
                "episode_sources": episode_rebuild_sources,
                "profile_person_ids": profile_refresh_person_ids,
                "created_at": time.time(),
            }
            update_rollback_plan = getattr(self.metadata_store, "update_feedback_task_rollback_plan", None)
            if callable(update_rollback_plan):
                update_rollback_plan(
                    task_id=task_id,
                    rollback_plan=rollback_plan,
                )
        return {
            "applied": applied,
            "forget": forget_result,
            "ingest": ingest_result,
            "relation_hashes": relation_hashes,
            "stale_paragraph_hashes": stale_paragraph_hashes,
            "episode_rebuild_sources": episode_rebuild_sources,
            "profile_refresh_person_ids": profile_refresh_person_ids,
            "rollback_plan_summary": self._build_feedback_rollback_plan_summary(rollback_plan) if rollback_plan else {},
        }

    def _resolve_feedback_relation_hashes(
        self,
        *,
        target_hashes: Sequence[str],
        hit_map: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        resolved: List[str] = []
        seen: set[str] = set()
        for target_hash in target_hashes:
            token = str(target_hash or "").strip()
            if not token:
                continue
            hit = hit_map.get(token) if isinstance(hit_map, dict) else None
            item_type = str((hit or {}).get("type", "") or "").strip()
            if item_type == "relation":
                if token not in seen:
                    seen.add(token)
                    resolved.append(token)
                continue
            if item_type != "paragraph":
                continue

            linked_candidates = self._tokens((hit or {}).get("linked_relation_hashes"))
            if not linked_candidates and self.metadata_store is not None:
                for relation in self.metadata_store.get_paragraph_relations(token):
                    linked_hash = str(relation.get("hash", "") or "").strip()
                    if linked_hash:
                        linked_candidates.append(linked_hash)

            for linked_hash in linked_candidates:
                if linked_hash in seen:
                    continue
                seen.add(linked_hash)
                resolved.append(linked_hash)
        return resolved

    async def _process_feedback_task(self, task: Dict[str, Any]) -> None:
        task_id = int(task.get("id") or 0)
        query_tool_id = str(task.get("query_tool_id", "") or "").strip()
        if task_id <= 0 or not query_tool_id:
            return

        assert self.metadata_store is not None
        self.metadata_store.mark_feedback_task_running(task_id)

        decision_payload: Dict[str, Any] = {}
        session_id = str(task.get("session_id", "") or "").strip()
        try:
            structured = task.get("query_snapshot") if isinstance(task.get("query_snapshot"), dict) else {}
            if not session_id:
                session_id = str(structured.get("chat_id", "") or "").strip()
            if not session_id:
                raise RuntimeError("反馈任务缺少 session_id")
            hits_raw = structured.get("hits")
            if not isinstance(hits_raw, list) or not hits_raw:
                decision_payload = {"decision": "none", "confidence": 1.0, "reason": "no_hits"}
                self.metadata_store.finalize_feedback_task(
                    task_id=task_id,
                    status="skipped",
                    decision_payload=decision_payload,
                )
                return

            query_timestamp = self._coerce_datetime(task.get("query_timestamp")) or datetime.now()
            due_at = self._coerce_datetime(task.get("due_at")) or (
                query_timestamp + timedelta(hours=self._feedback_cfg_window_hours())
            )
            if due_at <= query_timestamp:
                due_at = query_timestamp + timedelta(hours=self._feedback_cfg_window_hours())

            feedback_messages = self._extract_feedback_messages(
                session_id=session_id,
                query_time=query_timestamp,
                due_time=due_at,
                max_messages=self._feedback_cfg_max_messages(),
            )
            if not feedback_messages:
                decision_payload = {"decision": "none", "confidence": 1.0, "reason": "no_feedback_messages"}
                self.metadata_store.finalize_feedback_task(
                    task_id=task_id,
                    status="skipped",
                    decision_payload=decision_payload,
                )
                return

            if self._feedback_cfg_prefilter_enabled() and not self._should_invoke_feedback_classifier(feedback_messages):
                decision_payload = {"decision": "none", "confidence": 1.0, "reason": "prefilter_skipped"}
                self.metadata_store.append_feedback_action_log(
                    task_id=task_id,
                    query_tool_id=query_tool_id,
                    action_type="skip",
                    reason="prefilter_skipped",
                    after_payload={"feedback_messages": feedback_messages},
                )
                self.metadata_store.finalize_feedback_task(
                    task_id=task_id,
                    status="skipped",
                    decision_payload=decision_payload,
                )
                return

            hit_briefs = self._build_feedback_hit_briefs(hits_raw)
            hit_map = {str(item.get("hash", "") or "").strip(): item for item in hit_briefs if str(item.get("hash", "") or "").strip()}
            raw_decision = await self._classify_feedback(
                query_tool_id=query_tool_id,
                query_text=str(structured.get("query", "") or ""),
                hit_briefs=hit_briefs,
                feedback_messages=feedback_messages,
            )
            decision_payload = self._normalize_feedback_decision(raw_decision, hit_hashes=list(hit_map.keys()))
            decision_payload["feedback_message_count"] = len(feedback_messages)
            self.metadata_store.append_feedback_action_log(
                task_id=task_id,
                query_tool_id=query_tool_id,
                action_type="classification",
                after_payload=decision_payload,
                reason=str(decision_payload.get("reason", "") or ""),
            )

            apply_result = await self._apply_feedback_decision(
                task_id=task_id,
                query_tool_id=query_tool_id,
                session_id=session_id,
                decision=decision_payload,
                hit_map=hit_map,
            )
            decision_payload["apply_result"] = apply_result
            final_status = self._feedback_apply_result_status(apply_result)
            self.metadata_store.finalize_feedback_task(
                task_id=task_id,
                status=final_status,
                decision_payload=decision_payload,
                last_error=str(apply_result.get("error", "") or "") if final_status == "error" else "",
            )
        except Exception as exc:
            logger.warning(f"反馈纠错任务处理失败: task_id={task_id} err={exc}", exc_info=True)
            self.metadata_store.append_feedback_action_log(
                task_id=task_id,
                query_tool_id=query_tool_id,
                action_type="error",
                reason=str(exc),
                after_payload=decision_payload if decision_payload else None,
            )
            self.metadata_store.finalize_feedback_task(
                task_id=task_id,
                status="error",
                decision_payload=decision_payload if decision_payload else None,
                last_error=str(exc),
            )

    async def _feedback_correction_loop(self) -> None:
        try:
            while not self._background_stopping:
                interval_seconds = self._feedback_cfg_check_interval_seconds()
                if not self._feedback_cfg_enabled():
                    await asyncio.sleep(interval_seconds)
                    continue
                if self.metadata_store is None:
                    await asyncio.sleep(interval_seconds)
                    continue
                tasks = self.metadata_store.fetch_due_feedback_tasks(
                    limit=self._feedback_cfg_batch_size(),
                    now=datetime.now().timestamp(),
                )
                if not tasks:
                    await asyncio.sleep(interval_seconds)
                    continue
                for task in tasks:
                    if self._background_stopping:
                        break
                    if not isinstance(task, dict):
                        continue
                    await self._process_feedback_task(task)
                await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"feedback_correction loop 异常: {exc}")
