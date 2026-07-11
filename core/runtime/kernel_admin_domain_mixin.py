from __future__ import annotations

import time
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Sequence
from src.common.logger import get_logger
from ..utils.retrieval_tuning_manager import RetrievalTuningManager
from ..utils.web_import_manager import ImportTaskManager

logger = get_logger("A_Memorix.SDKMemoryKernel")


class KernelAdminDomainMixin:
    async def memory_graph_admin(self, *, action: str, **kwargs) -> Dict[str, Any]:
        await self.initialize()
        assert self.metadata_store is not None
        assert self.graph_store is not None

        act = str(action or "").strip().lower()
        if act == "get_graph":
            return {"success": True, **self._serialize_graph(limit=max(1, int(kwargs.get("limit", 200) or 200)))}
        if act == "search":
            return self._search_graph(
                query=str(kwargs.get("query", "") or "").strip(),
                limit=max(1, min(200, int(kwargs.get("limit", 50) or 50))),
            )
        if act == "node_detail":
            detail = self._build_graph_node_detail(
                node_id=str(kwargs.get("node_id", "") or kwargs.get("node", "") or "").strip(),
                relation_limit=max(1, int(kwargs.get("relation_limit", 20) or 20)),
                paragraph_limit=max(1, int(kwargs.get("paragraph_limit", 20) or 20)),
                evidence_node_limit=max(12, int(kwargs.get("evidence_node_limit", 80) or 80)),
            )
            return detail
        if act == "edge_detail":
            detail = self._build_graph_edge_detail(
                source=str(kwargs.get("source", "") or "").strip(),
                target=str(kwargs.get("target", "") or kwargs.get("object", "") or "").strip(),
                paragraph_limit=max(1, int(kwargs.get("paragraph_limit", 20) or 20)),
                evidence_node_limit=max(12, int(kwargs.get("evidence_node_limit", 80) or 80)),
            )
            return detail

        if act == "create_node":
            name = str(kwargs.get("name", "") or kwargs.get("node", "") or "").strip()
            if not name:
                return {"success": False, "error": "node name 不能为空"}
            entity_hash = self.metadata_store.add_entity(name=name, metadata=kwargs.get("metadata") or {})
            self._rebuild_graph_from_metadata()
            self._persist()
            return {"success": True, "node": {"name": name, "hash": entity_hash}}

        if act == "delete_node":
            name = str(kwargs.get("name", "") or kwargs.get("node", "") or kwargs.get("hash_or_name", "") or "").strip()
            if not name:
                return {"success": False, "error": "node name 不能为空"}
            result = await self._execute_delete_action(
                mode="entity",
                selector={"query": name},
                requested_by=str(kwargs.get("requested_by", "") or "memory_graph_admin"),
                reason=str(kwargs.get("reason", "") or "graph_delete_node"),
            )
            return {
                **result,
                "deleted": bool(result.get("deleted_entity_count", 0) or result.get("deleted_count", 0)),
                "node": name,
            }

        if act == "rename_node":
            old_name = str(kwargs.get("name", "") or kwargs.get("old_name", "") or kwargs.get("node", "") or "").strip()
            new_name = str(kwargs.get("new_name", "") or kwargs.get("target_name", "") or "").strip()
            return self._rename_node(old_name, new_name)

        if act == "create_edge":
            subject = str(kwargs.get("subject", "") or kwargs.get("source", "") or "").strip()
            predicate = str(kwargs.get("predicate", "") or kwargs.get("label", "") or "").strip()
            obj = str(kwargs.get("object", "") or kwargs.get("target", "") or "").strip()
            if not all([subject, predicate, obj]):
                return {"success": False, "error": "subject/predicate/object 不能为空"}
            if self.relation_write_service is not None:
                result = await self.relation_write_service.upsert_relation_with_vector(
                    subject=subject,
                    predicate=predicate,
                    obj=obj,
                    confidence=float(kwargs.get("confidence", 1.0) or 1.0),
                    source_paragraph=str(kwargs.get("source_paragraph", "") or "") or None,
                    metadata=kwargs.get("metadata") or {},
                    write_vector=self.relation_vectors_enabled,
                )
                relation_hash = result.hash_value
            else:
                relation_hash = self.metadata_store.add_relation(
                    subject=subject,
                    predicate=predicate,
                    obj=obj,
                    confidence=float(kwargs.get("confidence", 1.0) or 1.0),
                    source_paragraph=kwargs.get("source_paragraph"),
                    metadata=kwargs.get("metadata") or {},
                )
            self._rebuild_graph_from_metadata()
            self._persist()
            return {
                "success": True,
                "edge": {
                    "hash": relation_hash,
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "weight": float(kwargs.get("confidence", 1.0) or 1.0),
                },
            }

        if act == "delete_edge":
            relation_hash = str(kwargs.get("hash", "") or kwargs.get("relation_hash", "") or "").strip()
            if relation_hash:
                result = await self._execute_delete_action(
                    mode="relation",
                    selector={"query": relation_hash},
                    requested_by=str(kwargs.get("requested_by", "") or "memory_graph_admin"),
                    reason=str(kwargs.get("reason", "") or "graph_delete_edge"),
                )
                return {
                    **result,
                    "deleted": int(result.get("deleted_relation_count", 0) or result.get("deleted_count", 0)),
                    "hash": relation_hash,
                }

            subject = str(kwargs.get("subject", "") or kwargs.get("source", "") or "").strip()
            obj = str(kwargs.get("object", "") or kwargs.get("target", "") or "").strip()
            deleted_hashes = [
                str(row.get("hash", "") or "")
                for row in self.metadata_store.get_relations(subject=subject)
                if str(row.get("object", "") or "").strip() == obj
            ]
            result = await self._execute_delete_action(
                mode="relation",
                selector={"hashes": deleted_hashes, "subject": subject, "object": obj},
                requested_by=str(kwargs.get("requested_by", "") or "memory_graph_admin"),
                reason=str(kwargs.get("reason", "") or "graph_delete_edge"),
            )
            return {
                **result,
                "deleted": int(result.get("deleted_relation_count", 0) or result.get("deleted_count", 0)),
                "subject": subject,
                "object": obj,
            }

        if act == "update_edge_weight":
            return self._update_edge_weight(
                relation_hash=str(kwargs.get("hash", "") or kwargs.get("relation_hash", "") or "").strip(),
                subject=str(kwargs.get("subject", "") or kwargs.get("source", "") or "").strip(),
                obj=str(kwargs.get("object", "") or kwargs.get("target", "") or "").strip(),
                weight=float(kwargs.get("weight", kwargs.get("confidence", 1.0)) or 1.0),
            )

        return {"success": False, "error": f"不支持的 graph action: {act}"}

    async def memory_source_admin(self, *, action: str, **kwargs) -> Dict[str, Any]:
        await self.initialize()
        assert self.metadata_store

        act = str(action or "").strip().lower()
        if act == "list":
            sources = self.metadata_store.get_all_sources()
            items = []
            for row in sources:
                source_name = str(row.get("source", "") or "").strip()
                items.append(
                    {
                        **row,
                        "episode_rebuild_blocked": self.metadata_store.is_episode_source_query_blocked(source_name),
                    }
                )
            return {"success": True, "items": items, "count": len(items)}

        if act == "delete":
            source = str(kwargs.get("source", "") or "").strip()
            return await self._execute_delete_action(
                mode="source",
                selector={"sources": [source]},
                requested_by=str(kwargs.get("requested_by", "") or "memory_source_admin"),
                reason=str(kwargs.get("reason", "") or "source_delete"),
            )

        if act == "batch_delete":
            return await self._execute_delete_action(
                mode="source",
                selector={"sources": list(kwargs.get("sources") or [])},
                requested_by=str(kwargs.get("requested_by", "") or "memory_source_admin"),
                reason=str(kwargs.get("reason", "") or "source_batch_delete"),
            )

        return {"success": False, "error": f"不支持的 source action: {act}"}

    async def memory_episode_admin(self, *, action: str, **kwargs) -> Dict[str, Any]:
        await self.initialize()
        assert self.metadata_store

        act = str(action or "").strip().lower()
        if act in {"query", "list"}:
            items = self.metadata_store.query_episodes(
                query=str(kwargs.get("query", "") or "").strip(),
                time_from=self._optional_float(kwargs.get("time_start", kwargs.get("time_from"))),
                time_to=self._optional_float(kwargs.get("time_end", kwargs.get("time_to"))),
                person=str(kwargs.get("person_id", "") or kwargs.get("person", "") or "").strip() or None,
                source=str(kwargs.get("source", "") or "").strip() or None,
                limit=max(1, int(kwargs.get("limit", 20) or 20)),
            )
            return {"success": True, "items": items, "count": len(items)}

        if act == "get":
            episode_id = str(kwargs.get("episode_id", "") or "").strip()
            if not episode_id:
                return {"success": False, "error": "episode_id 不能为空"}
            episode = self.metadata_store.get_episode_by_id(episode_id)
            if episode is None:
                return {"success": False, "error": "episode 不存在"}
            episode["paragraphs"] = self.metadata_store.get_episode_paragraphs(
                episode_id,
                limit=max(1, int(kwargs.get("paragraph_limit", 100) or 100)),
            )
            return {"success": True, "episode": episode}

        if act == "status":
            summary = self.metadata_store.get_episode_source_rebuild_summary(
                failed_limit=max(1, int(kwargs.get("limit", 20) or 20))
            )
            summary["pending_queue"] = self.metadata_store.query(
                "SELECT COUNT(*) AS c FROM episode_pending_paragraphs WHERE status IN ('pending', 'running', 'failed')"
            )[0]["c"]
            return {"success": True, **summary}

        if act == "rebuild":
            sources = self._tokens(kwargs.get("sources"))
            if not sources:
                source = str(kwargs.get("source", "") or "").strip()
                if source:
                    sources = [source]
            if not sources and bool(kwargs.get("all", False)):
                sources = self.metadata_store.list_episode_sources_for_rebuild()
                if not sources:
                    sources = [str(row.get("source", "") or "").strip() for row in self.metadata_store.get_all_sources()]
            if not sources:
                return {"success": False, "error": "未提供可重建的 source"}
            result = await self.rebuild_episodes_for_sources(sources)
            return {"success": len(result.get("failures", [])) == 0, **result}

        if act == "process_pending":
            result = await self.process_episode_pending_batch(
                limit=max(1, int(kwargs.get("limit", 20) or 20)),
                max_retry=max(1, int(kwargs.get("max_retry", 3) or 3)),
            )
            return {"success": True, **result}

        return {"success": False, "error": f"不支持的 episode action: {act}"}

    async def memory_profile_admin(self, *, action: str, **kwargs) -> Dict[str, Any]:
        await self.initialize()
        assert self.metadata_store is not None
        assert self.person_profile_service is not None

        act = str(action or "").strip().lower()
        if act == "query":
            profile = await self._query_person_profile_with_feedback_refresh(
                person_id=str(kwargs.get("person_id", "") or "").strip(),
                person_keyword=str(kwargs.get("person_keyword", "") or kwargs.get("keyword", "") or "").strip(),
                limit=max(1, int(kwargs.get("limit", kwargs.get("top_k", 12)) or 12)),
                force_refresh=bool(kwargs.get("force_refresh", False)),
                source_note="sdk_memory_kernel.memory_profile_admin.query",
            )
            return profile if isinstance(profile, dict) else {"success": False, "error": "invalid profile payload"}

        if act == "status":
            summary = self.metadata_store.get_person_profile_refresh_summary(
                failed_limit=max(1, int(kwargs.get("limit", 20) or 20))
            )
            return {"success": True, **summary}

        if act == "process_pending":
            result = await self._process_feedback_profile_refresh_batch(
                limit=max(1, int(kwargs.get("limit", self._feedback_cfg_reconcile_batch_size()) or self._feedback_cfg_reconcile_batch_size()))
            )
            return {"success": True, **result}

        if act == "list":
            limit = max(1, int(kwargs.get("limit", 50) or 50))
            rows = self.metadata_store.query(
                """
                SELECT s.person_id, s.profile_version, s.profile_text, s.updated_at, s.expires_at, s.source_note
                FROM person_profile_snapshots s
                JOIN (
                    SELECT person_id, MAX(profile_version) AS max_version
                    FROM person_profile_snapshots
                    GROUP BY person_id
                ) latest
                  ON latest.person_id = s.person_id
                 AND latest.max_version = s.profile_version
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            items = []
            for row in rows:
                person_id = str(row.get("person_id", "") or "").strip()
                override = self.metadata_store.get_person_profile_override(person_id)
                items.append(
                    {
                        "person_id": person_id,
                        "profile_version": int(row.get("profile_version", 0) or 0),
                        "profile_text": str(row.get("profile_text", "") or ""),
                        "updated_at": row.get("updated_at"),
                        "expires_at": row.get("expires_at"),
                        "source_note": str(row.get("source_note", "") or ""),
                        "has_manual_override": bool(override),
                        "manual_override": override,
                    }
                )
            return {"success": True, "items": items, "count": len(items)}

        if act == "set_override":
            person_id = str(kwargs.get("person_id", "") or "").strip()
            override = self.metadata_store.set_person_profile_override(
                person_id=person_id,
                override_text=str(kwargs.get("override_text", "") or kwargs.get("text", "") or ""),
                updated_by=str(kwargs.get("updated_by", "") or ""),
                source=str(kwargs.get("source", "") or "memory_profile_admin"),
            )
            return {"success": True, "override": override}

        if act == "delete_override":
            person_id = str(kwargs.get("person_id", "") or "").strip()
            deleted = self.metadata_store.delete_person_profile_override(person_id)
            return {"success": bool(deleted), "deleted": bool(deleted), "person_id": person_id}

        return {"success": False, "error": f"不支持的 profile action: {act}"}

    async def memory_feedback_admin(self, *, action: str, **kwargs) -> Dict[str, Any]:
        await self.initialize()
        assert self.metadata_store is not None

        act = str(action or "").strip().lower()
        if act == "list":
            items = self.metadata_store.list_feedback_tasks(
                limit=max(1, int(kwargs.get("limit", 50) or 50)),
                statuses=self._tokens(kwargs.get("status") or kwargs.get("statuses")),
                rollback_statuses=self._tokens(kwargs.get("rollback_status") or kwargs.get("rollback_statuses")),
                query=str(kwargs.get("query", "") or "").strip(),
            )
            return {
                "success": True,
                "items": [self._build_feedback_task_summary(task) for task in items],
                "count": len(items),
            }

        if act == "get":
            task = self.metadata_store.get_feedback_task_by_id(int(kwargs.get("task_id", 0) or 0))
            if task is None:
                return {"success": False, "error": "反馈纠错任务不存在"}
            return {"success": True, "task": self._build_feedback_task_detail(task)}

        if act == "rollback":
            return await self._rollback_feedback_task(
                task_id=int(kwargs.get("task_id", 0) or 0),
                requested_by=str(kwargs.get("requested_by", "") or "").strip(),
                reason=str(kwargs.get("reason", "") or "").strip(),
            )

        return {"success": False, "error": f"不支持的 feedback action: {act}"}
