from __future__ import annotations

import time
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Sequence
from src.common.logger import get_logger
from ..utils.retrieval_tuning_manager import RetrievalTuningManager
from ..utils.web_import_manager import ImportTaskManager

logger = get_logger("A_Memorix.SDKMemoryKernel")


class KernelAdminOpsMixin:
    async def memory_runtime_admin(self, *, action: str, **kwargs) -> Dict[str, Any]:
        await self.initialize()
        act = str(action or "").strip().lower()

        if act == "save":
            self._persist()
            return {"success": True, "saved": True, "data_dir": str(self.data_dir)}

        if act == "get_config":
            degraded = self._embedding_degraded_snapshot()
            backfill_counts = self._paragraph_vector_backfill_counts()
            return {
                "success": True,
                "config": self.config,
                "data_dir": str(self.data_dir),
                "embedding_dimension": int(self.embedding_dimension),
                "auto_save": bool(self._cfg("advanced.enable_auto_save", True)),
                "relation_vectors_enabled": bool(self.relation_vectors_enabled),
                "runtime_ready": self.is_runtime_ready(),
                "embedding_degraded": bool(degraded.get("active", False)),
                "embedding_degraded_reason": str(degraded.get("reason", "") or ""),
                "embedding_degraded_since": degraded.get("since"),
                "embedding_last_check": degraded.get("last_check"),
                "paragraph_vector_backfill_pending": int(backfill_counts.get("pending", 0) or 0),
                "paragraph_vector_backfill_running": int(backfill_counts.get("running", 0) or 0),
                "paragraph_vector_backfill_failed": int(backfill_counts.get("failed", 0) or 0),
                "paragraph_vector_backfill_done": int(backfill_counts.get("done", 0) or 0),
            }

        if act in {"self_check", "refresh_self_check"}:
            report = await self._refresh_runtime_self_check(
                sample_text=str(kwargs.get("sample_text", "") or "A_Memorix runtime self check")
            )
            checked_at = float(report.get("checked_at") or time.time())
            if bool(report.get("ok", False)):
                self._set_embedding_degraded(active=False, checked_at=checked_at)
            elif self._embedding_fallback_enabled():
                self._set_embedding_degraded(
                    active=True,
                    reason=str(report.get("message", "runtime self-check failed") or "runtime self-check failed"),
                    checked_at=checked_at,
                )
            return {"success": bool(report.get("ok", False)), "report": report}

        if act == "set_auto_save":
            enabled = bool(kwargs.get("enabled", False))
            self._set_cfg("advanced.enable_auto_save", enabled)
            return {"success": True, "auto_save": enabled}

        if act == "recover_embedding":
            result = await self._recover_embedding_once(
                sample_text=str(kwargs.get("sample_text", "") or "A_Memorix runtime self check")
            )
            result["embedding_degraded"] = self._is_embedding_degraded()
            result["embedding_state"] = self._embedding_degraded_snapshot()
            result["backfill_counts"] = self._paragraph_vector_backfill_counts()
            return result

        if act == "paragraph_backfill_once":
            result = await self._run_paragraph_backfill_once(
                limit=self._optional_int(kwargs.get("limit")),
                max_retry=self._optional_int(kwargs.get("max_retry")),
                trigger="manual",
            )
            result["embedding_degraded"] = self._is_embedding_degraded()
            result["backfill_counts"] = self._paragraph_vector_backfill_counts()
            return result

        return {"success": False, "error": f"不支持的 runtime action: {act}"}

    async def memory_import_admin(self, *, action: str, **kwargs) -> Dict[str, Any]:
        await self.initialize()
        manager = self.import_task_manager
        if manager is None:
            return {"success": False, "error": "import manager 未初始化"}

        act = str(action or "").strip().lower()
        if act in {"settings", "get_settings", "get_guide"}:
            return {"success": True, "settings": await manager.get_runtime_settings()}
        if act in {"path_aliases", "get_path_aliases"}:
            return {"success": True, "path_aliases": manager.get_path_aliases()}
        if act in {"resolve_path", "resolve"}:
            return await manager.resolve_path_request(kwargs)
        if act == "create_upload":
            task = await manager.create_upload_task(
                list(kwargs.get("staged_files") or kwargs.get("files") or kwargs.get("uploads") or []),
                kwargs,
            )
            return {"success": True, "task": task}
        if act == "create_paste":
            return {"success": True, "task": await manager.create_paste_task(kwargs)}
        if act == "create_raw_scan":
            return {"success": True, "task": await manager.create_raw_scan_task(kwargs)}
        if act == "create_lpmm_openie":
            return {"success": True, "task": await manager.create_lpmm_openie_task(kwargs)}
        if act == "create_lpmm_convert":
            return {"success": True, "task": await manager.create_lpmm_convert_task(kwargs)}
        if act == "create_temporal_backfill":
            return {"success": True, "task": await manager.create_temporal_backfill_task(kwargs)}
        if act == "create_maibot_migration":
            return {"success": True, "task": await manager.create_maibot_migration_task(kwargs)}
        if act == "list":
            items = await manager.list_tasks(limit=max(1, int(kwargs.get("limit", 50) or 50)))
            return {"success": True, "items": items, "count": len(items)}
        if act == "get":
            task = await manager.get_task(
                str(kwargs.get("task_id", "") or ""),
                include_chunks=bool(kwargs.get("include_chunks", False)),
            )
            return {"success": task is not None, "task": task, "error": "" if task is not None else "任务不存在"}
        if act in {"chunks", "get_chunks"}:
            payload = await manager.get_chunks(
                str(kwargs.get("task_id", "") or ""),
                str(kwargs.get("file_id", "") or ""),
                offset=max(0, int(kwargs.get("offset", 0) or 0)),
                limit=max(1, int(kwargs.get("limit", 50) or 50)),
            )
            return {"success": payload is not None, **(payload or {}), "error": "" if payload is not None else "任务或文件不存在"}
        if act == "cancel":
            task = await manager.cancel_task(str(kwargs.get("task_id", "") or ""))
            return {"success": task is not None, "task": task, "error": "" if task is not None else "任务不存在"}
        if act == "retry_failed":
            overrides = kwargs.get("overrides") if isinstance(kwargs.get("overrides"), dict) else kwargs
            task = await manager.retry_failed(str(kwargs.get("task_id", "") or ""), overrides=overrides)
            return {"success": task is not None, "task": task, "error": "" if task is not None else "任务不存在"}
        return {"success": False, "error": f"不支持的 import action: {act}"}

    async def memory_tuning_admin(self, *, action: str, **kwargs) -> Dict[str, Any]:
        await self.initialize()
        manager = self.retrieval_tuning_manager
        if manager is None:
            return {"success": False, "error": "tuning manager 未初始化"}

        act = str(action or "").strip().lower()
        if act in {"settings", "get_settings"}:
            return {"success": True, "settings": manager.get_runtime_settings()}
        if act == "get_profile":
            profile = manager.get_profile_snapshot()
            return {"success": True, "profile": profile, "toml": manager.export_toml_snippet(profile)}
        if act == "apply_profile":
            profile_raw = kwargs.get("profile")
            if isinstance(profile_raw, dict):
                profile_payload: Dict[str, Any] = dict(profile_raw)
            else:
                profile_payload = {
                    key: value
                    for key, value in kwargs.items()
                    if key not in {"reason", "profile"}
                }
            return {
                "success": True,
                **await manager.apply_profile(
                    profile_payload,
                    reason=str(kwargs.get("reason", "manual") or "manual"),
                ),
            }
        if act == "rollback_profile":
            return {"success": True, **await manager.rollback_profile()}
        if act == "export_profile":
            profile = manager.get_profile_snapshot()
            return {"success": True, "profile": profile, "toml": manager.export_toml_snippet(profile)}
        if act == "create_task":
            payload = kwargs.get("payload") if isinstance(kwargs.get("payload"), dict) else kwargs
            return {"success": True, "task": await manager.create_task(payload)}
        if act == "list_tasks":
            items = await manager.list_tasks(limit=max(1, int(kwargs.get("limit", 50) or 50)))
            return {"success": True, "items": items, "count": len(items)}
        if act == "get_task":
            task = await manager.get_task(
                str(kwargs.get("task_id", "") or ""),
                include_rounds=bool(kwargs.get("include_rounds", False)),
            )
            return {"success": task is not None, "task": task, "error": "" if task is not None else "任务不存在"}
        if act == "get_rounds":
            payload = await manager.get_rounds(
                str(kwargs.get("task_id", "") or ""),
                offset=max(0, int(kwargs.get("offset", 0) or 0)),
                limit=max(1, int(kwargs.get("limit", 50) or 50)),
            )
            return {"success": payload is not None, **(payload or {}), "error": "" if payload is not None else "任务不存在"}
        if act == "cancel":
            task = await manager.cancel_task(str(kwargs.get("task_id", "") or ""))
            return {"success": task is not None, "task": task, "error": "" if task is not None else "任务不存在"}
        if act == "apply_best":
            return {"success": True, **await manager.apply_best(str(kwargs.get("task_id", "") or ""))}
        if act == "get_report":
            report = await manager.get_report(str(kwargs.get("task_id", "") or ""), fmt=str(kwargs.get("format", "md") or "md"))
            return {"success": report is not None, "report": report, "error": "" if report is not None else "任务不存在"}
        return {"success": False, "error": f"不支持的 tuning action: {act}"}

    async def memory_v5_admin(self, *, action: str, **kwargs) -> Dict[str, Any]:
        await self.initialize()
        assert self.metadata_store

        act = str(action or "").strip().lower()
        target = str(kwargs.get("target", "") or kwargs.get("query", "") or "").strip()
        reason = str(kwargs.get("reason", "") or "").strip()
        updated_by = str(kwargs.get("updated_by", "") or kwargs.get("requested_by", "") or "").strip()
        limit = max(1, int(kwargs.get("limit", 50) or 50))

        if act == "recycle_bin":
            items = self.metadata_store.get_deleted_relations(limit=limit)
            return {"success": True, "items": items, "count": len(items)}

        if act == "status":
            return self._memory_v5_status(target=target, limit=limit)

        if act == "restore":
            hashes = self._resolve_deleted_relation_hashes(target)
            if not hashes:
                return {"success": False, "error": "未命中可恢复关系"}
            result = await self._restore_relation_hashes(hashes)
            operation = self.metadata_store.record_v5_operation(
                action=act,
                target=target,
                resolved_hashes=hashes,
                reason=reason,
                updated_by=updated_by,
                result=result,
            )
            return {"success": bool(result.get("restored_count", 0) > 0), "operation": operation, **result}

        hashes = self._resolve_relation_hashes(target)
        if not hashes:
            return {"success": False, "error": "未命中可维护关系"}

        result = self._apply_v5_relation_action(
            action=act,
            hashes=hashes,
            strength=float(kwargs.get("strength", 1.0) or 1.0),
        )
        operation = self.metadata_store.record_v5_operation(
            action=act,
            target=target,
            resolved_hashes=hashes,
            reason=reason,
            updated_by=updated_by,
            result=result,
        )
        return {"success": bool(result.get("success", False)), "operation": operation, **result}

    async def memory_delete_admin(self, *, action: str, **kwargs) -> Dict[str, Any]:
        await self.initialize()
        act = str(action or "").strip().lower()
        mode = str(kwargs.get("mode", "") or "").strip().lower()
        selector = kwargs.get("selector")
        if selector is None:
            selector = {
                key: value
                for key, value in kwargs.items()
                if key
                not in {
                    "action",
                    "mode",
                    "dry_run",
                    "cascade",
                    "operation_id",
                    "reason",
                    "requested_by",
                }
            }
        reason = str(kwargs.get("reason", "") or "").strip()
        requested_by = str(kwargs.get("requested_by", "") or "").strip()

        if act == "preview":
            return await self._preview_delete_action(mode=mode, selector=selector)
        if act == "execute":
            return await self._execute_delete_action(
                mode=mode,
                selector=selector,
                requested_by=requested_by,
                reason=reason,
            )
        if act == "restore":
            return await self._restore_delete_action(
                mode=mode,
                selector=selector,
                operation_id=str(kwargs.get("operation_id", "") or "").strip(),
                requested_by=requested_by,
                reason=reason,
            )
        if act == "get_operation":
            operation = self.metadata_store.get_delete_operation(str(kwargs.get("operation_id", "") or "").strip())
            return {"success": operation is not None, "operation": operation, "error": "" if operation is not None else "operation 不存在"}
        if act == "list_operations":
            items = self.metadata_store.list_delete_operations(
                limit=max(1, int(kwargs.get("limit", 50) or 50)),
                mode=mode,
            )
            return {"success": True, "items": items, "count": len(items)}
        if act == "purge":
            return await self._purge_deleted_memory(
                grace_hours=self._optional_float(kwargs.get("grace_hours")),
                limit=max(1, int(kwargs.get("limit", 1000) or 1000)),
            )
        return {"success": False, "error": f"不支持的 delete action: {act}"}

    def get_import_task_manager(self) -> Optional[ImportTaskManager]:
        return self.import_task_manager

    def get_retrieval_tuning_manager(self) -> Optional[RetrievalTuningManager]:
        return self.retrieval_tuning_manager

    def _memory_v5_status(self, *, target: str = "", limit: int = 50) -> Dict[str, Any]:
        assert self.metadata_store
        now = time.time()
        summary = self.metadata_store.get_memory_status_summary(now)
        payload: Dict[str, Any] = {
            "success": True,
            **summary,
            "config": {
                "half_life_hours": float(self._cfg("memory.half_life_hours", 24.0) or 24.0),
                "base_decay_interval_hours": float(self._cfg("memory.base_decay_interval_hours", 1.0) or 1.0),
                "prune_threshold": float(self._cfg("memory.prune_threshold", 0.1) or 0.1),
                "freeze_duration_hours": float(self._cfg("memory.freeze_duration_hours", 24.0) or 24.0),
            },
            "last_maintenance_at": self._last_maintenance_at,
        }
        token = str(target or "").strip()
        if not token:
            return payload

        active_hashes = self._resolve_relation_hashes(token)[:limit]
        deleted_hashes = self._resolve_deleted_relation_hashes(token)[:limit]
        active_statuses = self.metadata_store.get_relation_status_batch(active_hashes)
        items: List[Dict[str, Any]] = []
        for hash_value in active_hashes:
            relation = self.metadata_store.get_relation(hash_value) or {}
            status = active_statuses.get(hash_value, {})
            items.append(
                {
                    "hash": hash_value,
                    "subject": str(relation.get("subject", "") or ""),
                    "predicate": str(relation.get("predicate", "") or ""),
                    "object": str(relation.get("object", "") or ""),
                    "state": "inactive" if bool(status.get("is_inactive")) else "active",
                    "is_pinned": bool(status.get("is_pinned", False)),
                    "temp_protected": bool(float(status.get("protected_until") or 0.0) > now),
                    "protected_until": status.get("protected_until"),
                    "last_reinforced": status.get("last_reinforced"),
                    "weight": float(status.get("weight", relation.get("confidence", 0.0)) or 0.0),
                }
            )
        for hash_value in deleted_hashes:
            relation = self.metadata_store.get_deleted_relation(hash_value) or {}
            items.append(
                {
                    "hash": hash_value,
                    "subject": str(relation.get("subject", "") or ""),
                    "predicate": str(relation.get("predicate", "") or ""),
                    "object": str(relation.get("object", "") or ""),
                    "state": "deleted",
                    "is_pinned": bool(relation.get("is_pinned", False)),
                    "temp_protected": False,
                    "protected_until": relation.get("protected_until"),
                    "last_reinforced": relation.get("last_reinforced"),
                    "weight": float(relation.get("confidence", 0.0) or 0.0),
                    "deleted_at": relation.get("deleted_at"),
                }
            )
        payload["items"] = items[:limit]
        payload["count"] = len(payload["items"])
        payload["target"] = token
        return payload

    def _adjust_relation_confidence(self, hashes: List[str], *, delta: float) -> Dict[str, float]:
        assert self.metadata_store
        normalized = [str(item or "").strip() for item in hashes if str(item or "").strip()]
        if not normalized:
            return {}
        conn = self.metadata_store.get_connection()
        cursor = conn.cursor()
        chunk_size = 200
        for index in range(0, len(normalized), chunk_size):
            chunk = normalized[index : index + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            cursor.execute(
                f"""
                UPDATE relations
                SET confidence = MAX(0.0, COALESCE(confidence, 0.0) + ?)
                WHERE hash IN ({placeholders})
                """,
                tuple([float(delta)] + chunk),
            )
        conn.commit()
        statuses = self.metadata_store.get_relation_status_batch(normalized)
        return {hash_value: float((statuses.get(hash_value) or {}).get("weight", 0.0) or 0.0) for hash_value in normalized}

    def _apply_v5_relation_action(self, *, action: str, hashes: List[str], strength: float = 1.0) -> Dict[str, Any]:
        assert self.metadata_store
        act = str(action or "").strip().lower()
        normalized = [str(item or "").strip() for item in hashes if str(item or "").strip()]
        if not normalized:
            return {"success": False, "error": "未命中可维护关系"}

        now = time.time()
        strength_value = max(0.1, float(strength or 1.0))
        prune_threshold = max(0.0, float(self._cfg("memory.prune_threshold", 0.1) or 0.1))
        detail = ""

        if act == "reinforce":
            weights = self._adjust_relation_confidence(normalized, delta=0.5 * strength_value)
            protect_hours = max(1.0, 24.0 * strength_value)
            self.metadata_store.reinforce_relations(normalized)
            self.metadata_store.mark_relations_active(normalized, boost_weight=max(prune_threshold, 0.1))
            self.metadata_store.update_relations_protection(
                normalized,
                protected_until=now + protect_hours * 3600.0,
                last_reinforced=now,
            )
            detail = f"reinforce {len(normalized)} 条关系"
        elif act == "weaken":
            weights = self._adjust_relation_confidence(normalized, delta=-0.5 * strength_value)
            to_freeze = [hash_value for hash_value, weight in weights.items() if weight <= prune_threshold]
            if to_freeze:
                self.metadata_store.mark_relations_inactive(to_freeze, inactive_since=now)
            detail = f"weaken {len(normalized)} 条关系"
        elif act == "remember_forever":
            self.metadata_store.mark_relations_active(normalized, boost_weight=max(prune_threshold, 0.1))
            self.metadata_store.update_relations_protection(normalized, protected_until=0.0, is_pinned=True)
            weights = {hash_value: float((self.metadata_store.get_relation_status_batch([hash_value]).get(hash_value) or {}).get("weight", 0.0) or 0.0) for hash_value in normalized}
            detail = f"remember_forever {len(normalized)} 条关系"
        elif act == "forget":
            weights = self._adjust_relation_confidence(normalized, delta=-2.0 * strength_value)
            self.metadata_store.update_relations_protection(normalized, protected_until=0.0, is_pinned=False)
            self.metadata_store.mark_relations_inactive(normalized, inactive_since=now)
            detail = f"forget {len(normalized)} 条关系"
        else:
            return {"success": False, "error": f"不支持的 V5 动作: {act}"}

        self._rebuild_graph_from_metadata()
        self._last_maintenance_at = now
        self._persist()
        statuses = self.metadata_store.get_relation_status_batch(normalized)
        return {
            "success": True,
            "detail": detail,
            "hashes": normalized,
            "count": len(normalized),
            "weights": weights,
            "statuses": statuses,
        }
