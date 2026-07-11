from __future__ import annotations

import time
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Sequence
from src.common.logger import get_logger
from ..utils.hash import compute_hash, normalize_text

logger = get_logger("A_Memorix.SDKMemoryKernel")


class KernelDeleteActionMixin:
    async def _build_delete_plan(self, *, mode: str, selector: Any) -> Dict[str, Any]:
        assert self.metadata_store
        act_mode = str(mode or "").strip().lower()
        normalized_selector = self._selector_dict(selector)
        items: List[Dict[str, Any]] = []
        counts = {"relations": 0, "paragraphs": 0, "entities": 0, "sources": 0}
        vector_ids: List[str] = []
        sources: List[str] = []
        target_hashes: Dict[str, List[str]] = {
            "relations": [],
            "paragraphs": [],
            "entities": [],
            "sources": [],
            "matched_sources": [],
        }
        seen_items: set[tuple[str, str]] = set()
        relation_hashes: List[str] = []
        paragraph_hashes: List[str] = []
        entity_hashes: List[str] = []
        paragraph_relation_candidates: List[str] = []

        def append_item(snapshot: Optional[Dict[str, Any]]) -> None:
            if not isinstance(snapshot, dict):
                return
            item_type = str(snapshot.get("item_type", "") or "").strip()
            item_hash = str(snapshot.get("item_hash", "") or snapshot.get("item_key", "") or "").strip()
            if not item_type or not item_hash:
                return
            key = (item_type, item_hash)
            if key in seen_items:
                return
            seen_items.add(key)
            items.append(snapshot)

        def append_relation_hash(hash_value: str) -> None:
            token = str(hash_value or "").strip()
            if not token or token in relation_hashes:
                return
            row = self.metadata_store.get_relation(token)
            if row is None:
                return
            relation_hashes.append(token)
            append_item(self._snapshot_relation_item(token))
            vector_ids.append(token)

        def append_paragraph_row(row: Optional[Dict[str, Any]]) -> None:
            if not isinstance(row, dict):
                return
            paragraph_hash = str(row.get("hash", "") or "").strip()
            if not paragraph_hash or paragraph_hash in paragraph_hashes or bool(row.get("is_deleted", 0)):
                return
            paragraph_hashes.append(paragraph_hash)
            snapshot = self._snapshot_paragraph_item(paragraph_hash)
            append_item(snapshot)
            vector_ids.append(paragraph_hash)
            paragraph = (snapshot or {}).get("payload", {}).get("paragraph") if isinstance((snapshot or {}).get("payload"), dict) else {}
            source = str((paragraph or {}).get("source", "") or "").strip()
            if source:
                sources.append(source)
            paragraph_relation_candidates.extend(self._tokens(((snapshot or {}).get("payload") or {}).get("relation_hashes")))

        def append_entity_row(row: Optional[Dict[str, Any]]) -> None:
            if not isinstance(row, dict):
                return
            entity_hash = str(row.get("hash", "") or "").strip()
            if not entity_hash or entity_hash in entity_hashes or bool(row.get("is_deleted", 0)):
                return
            entity_hashes.append(entity_hash)
            append_item(self._snapshot_entity_item(entity_hash))
            vector_ids.append(entity_hash)

        if act_mode == "relation":
            direct_hashes = self._merge_tokens(
                normalized_selector.get("hashes"),
                normalized_selector.get("items"),
                [normalized_selector.get("hash")],
            )
            query_hashes = self._resolve_relation_hashes(str(normalized_selector.get("query", "") or ""))
            for hash_value in direct_hashes or query_hashes:
                append_relation_hash(hash_value)
            counts["relations"] = len(relation_hashes)
            target_hashes["relations"] = list(relation_hashes)

        elif act_mode in {"paragraph", "source"}:
            paragraph_rows: List[Dict[str, Any]] = []
            if act_mode == "source":
                source_tokens = self._resolve_source_targets(normalized_selector)
                target_hashes["sources"] = source_tokens
                counts["requested_sources"] = len(source_tokens)
                matched_source_tokens: List[str] = []
                for source in source_tokens:
                    source_rows = self.metadata_store.query(
                        """
                        SELECT *
                        FROM paragraphs
                        WHERE source = ?
                          AND (is_deleted IS NULL OR is_deleted = 0)
                        ORDER BY created_at ASC
                        """,
                        (source,),
                    )
                    if source_rows:
                        matched_source_tokens.append(source)
                        sources.append(source)
                        paragraph_rows.extend(source_rows)
                target_hashes["matched_sources"] = matched_source_tokens
                counts["sources"] = len(matched_source_tokens)
                counts["matched_sources"] = len(matched_source_tokens)
            else:
                paragraph_rows = self._resolve_paragraph_targets(normalized_selector, include_deleted=False)
            for row in paragraph_rows:
                append_paragraph_row(row)
            target_hashes["paragraphs"] = list(paragraph_hashes)
            counts["paragraphs"] = len(paragraph_hashes)

            for relation_hash in self._tokens(paragraph_relation_candidates):
                if not self._relation_has_remaining_paragraphs(relation_hash, paragraph_hashes):
                    append_relation_hash(relation_hash)
            target_hashes["relations"] = list(relation_hashes)
            counts["relations"] = len(relation_hashes)

        elif act_mode == "entity":
            entity_rows = self._resolve_entity_targets(normalized_selector, include_deleted=False)
            for row in entity_rows:
                append_entity_row(row)
            target_hashes["entities"] = list(entity_hashes)
            counts["entities"] = len(entity_hashes)
            entity_names = [str(row.get("name", "") or "").strip() for row in entity_rows if str(row.get("name", "") or "").strip()]
            for entity_name in entity_names:
                for relation in self.metadata_store.get_relations(subject=entity_name) + self.metadata_store.get_relations(object=entity_name):
                    append_relation_hash(str(relation.get("hash", "") or "").strip())
            target_hashes["relations"] = list(relation_hashes)
            counts["relations"] = len(relation_hashes)
        elif act_mode == "mixed":
            source_tokens = self._merge_tokens(normalized_selector.get("sources"), [normalized_selector.get("source")])
            target_hashes["sources"] = list(source_tokens)
            counts["requested_sources"] = len(source_tokens)
            matched_source_tokens: List[str] = []

            for row in self._resolve_entity_targets({"hashes": normalized_selector.get("entity_hashes")}, include_deleted=False):
                append_entity_row(row)
            target_hashes["entities"] = list(entity_hashes)
            counts["entities"] = len(entity_hashes)

            for row in self._resolve_paragraph_targets({"hashes": normalized_selector.get("paragraph_hashes")}, include_deleted=False):
                append_paragraph_row(row)

            for source in source_tokens:
                source_rows = self.metadata_store.query(
                    """
                    SELECT *
                    FROM paragraphs
                    WHERE source = ?
                      AND (is_deleted IS NULL OR is_deleted = 0)
                    ORDER BY created_at ASC
                    """,
                    (source,),
                )
                if source_rows:
                    matched_source_tokens.append(source)
                    sources.append(source)
                    for row in source_rows:
                        append_paragraph_row(row)

            target_hashes["paragraphs"] = list(paragraph_hashes)
            counts["paragraphs"] = len(paragraph_hashes)
            target_hashes["matched_sources"] = matched_source_tokens
            counts["sources"] = len(matched_source_tokens)
            counts["matched_sources"] = len(matched_source_tokens)

            for hash_value in self._tokens(normalized_selector.get("relation_hashes")):
                append_relation_hash(hash_value)

            entity_names = [
                str(row.get("name", "") or "").strip()
                for row in self._resolve_entity_targets({"hashes": entity_hashes}, include_deleted=False)
                if str(row.get("name", "") or "").strip()
            ]
            for entity_name in entity_names:
                for relation in self.metadata_store.get_relations(subject=entity_name) + self.metadata_store.get_relations(object=entity_name):
                    append_relation_hash(str(relation.get("hash", "") or "").strip())

            for relation_hash in self._tokens(paragraph_relation_candidates):
                if not self._relation_has_remaining_paragraphs(relation_hash, paragraph_hashes):
                    append_relation_hash(relation_hash)

            target_hashes["relations"] = list(relation_hashes)
            counts["relations"] = len(relation_hashes)
        else:
            return {"success": False, "error": f"不支持的 delete mode: {act_mode}"}

        sources = self._tokens(sources)
        vector_ids = self._tokens(vector_ids)
        primary_count = counts.get(f"{act_mode}s", 0) if act_mode not in {"source", "mixed"} else counts.get("matched_sources", 0)
        success = (
            primary_count > 0 or counts.get("paragraphs", 0) > 0 or counts.get("relations", 0) > 0 or counts.get("entities", 0) > 0
            if act_mode != "source"
            else (counts.get("matched_sources", 0) > 0 and counts.get("paragraphs", 0) > 0)
        )
        return {
            "success": success,
            "mode": act_mode,
            "selector": normalized_selector,
            "items": items,
            "counts": counts,
            "vector_ids": vector_ids,
            "sources": sources,
            "target_hashes": target_hashes,
            "requested_source_count": counts.get("requested_sources", 0) if act_mode == "source" else 0,
            "matched_source_count": counts.get("matched_sources", 0) if act_mode == "source" else 0,
            "error": "" if success else "未命中可删除内容",
        }

    async def _preview_delete_action(self, *, mode: str, selector: Any) -> Dict[str, Any]:
        plan = await self._build_delete_plan(mode=mode, selector=selector)
        if not plan.get("success", False):
            return {"success": False, "error": plan.get("error", "未命中可删除内容")}
        preview_items = [self._build_delete_preview_item(item) for item in plan.get("items", [])[:100]]
        return {
            "success": True,
            "mode": plan.get("mode"),
            "selector": plan.get("selector"),
            "counts": plan.get("counts", {}),
            "requested_source_count": int(plan.get("requested_source_count", 0) or 0),
            "matched_source_count": int(plan.get("matched_source_count", 0) or 0),
            "sources": plan.get("sources", []),
            "vector_ids": plan.get("vector_ids", []),
            "items": preview_items,
            "item_count": len(plan.get("items", [])),
            "dry_run": True,
        }

    async def _execute_delete_action(
        self,
        *,
        mode: str,
        selector: Any,
        requested_by: str = "",
        reason: str = "",
    ) -> Dict[str, Any]:
        assert self.metadata_store
        plan = await self._build_delete_plan(mode=mode, selector=selector)
        if not plan.get("success", False):
            return {"success": False, "error": plan.get("error", "未命中可删除内容")}

        act_mode = str(plan.get("mode", "") or "").strip().lower()
        conn = self.metadata_store.get_connection()
        cursor = conn.cursor()
        paragraph_hashes = self._tokens((plan.get("target_hashes") or {}).get("paragraphs"))
        entity_hashes = self._tokens((plan.get("target_hashes") or {}).get("entities"))
        relation_hashes = self._tokens((plan.get("target_hashes") or {}).get("relations"))
        requested_source_tokens = self._tokens((plan.get("target_hashes") or {}).get("sources"))
        matched_source_tokens = self._tokens((plan.get("target_hashes") or {}).get("matched_sources"))

        try:
            if paragraph_hashes:
                self.metadata_store.mark_as_deleted(paragraph_hashes, "paragraph")
                cursor.execute(
                    f"DELETE FROM paragraph_entities WHERE paragraph_hash IN ({','.join(['?'] * len(paragraph_hashes))})",
                    tuple(paragraph_hashes),
                )
                cursor.execute(
                    f"DELETE FROM paragraph_relations WHERE paragraph_hash IN ({','.join(['?'] * len(paragraph_hashes))})",
                    tuple(paragraph_hashes),
                )
                self.metadata_store.delete_external_memory_refs_by_paragraphs(paragraph_hashes)
            if act_mode == "source" and matched_source_tokens:
                for source in matched_source_tokens:
                    self.metadata_store.replace_episodes_for_source(source, [])

            if entity_hashes:
                self.metadata_store.mark_as_deleted(entity_hashes, "entity")
                cursor.execute(
                    f"DELETE FROM paragraph_entities WHERE entity_hash IN ({','.join(['?'] * len(entity_hashes))})",
                    tuple(entity_hashes),
                )

            conn.commit()

            deleted_relations = self.metadata_store.backup_and_delete_relations(relation_hashes)
            deleted_vectors = 0
            if self.vector_store is not None and plan.get("vector_ids"):
                deleted_vectors = self.vector_store.delete(list(plan.get("vector_ids") or []))

            operation = self.metadata_store.create_delete_operation(
                mode=act_mode,
                selector=plan.get("selector"),
                items=plan.get("items", []),
                reason=reason,
                requested_by=requested_by,
                summary={
                    "counts": plan.get("counts", {}),
                    "sources": plan.get("sources", []),
                    "vector_ids": plan.get("vector_ids", []),
                    "deleted_relation_rows": deleted_relations,
                },
            )

            if plan.get("sources"):
                self.metadata_store._enqueue_episode_source_rebuilds(list(plan.get("sources") or []), reason="delete_admin_execute")
            self._rebuild_graph_from_metadata()
            self._persist()
            return self._build_standard_delete_result(
                mode=act_mode,
                operation_id=str(operation.get("operation_id", "") or ""),
                counts=plan.get("counts", {}),
                sources=plan.get("sources", []),
                deleted_entity_count=len(entity_hashes),
                deleted_relation_count=len(relation_hashes),
                deleted_paragraph_count=len(paragraph_hashes),
                deleted_source_count=len(matched_source_tokens),
                deleted_vector_count=int(deleted_vectors or 0),
                requested_source_count=len(requested_source_tokens),
                matched_source_count=len(matched_source_tokens),
                error="" if (entity_hashes or relation_hashes or paragraph_hashes or matched_source_tokens) else "未命中可删除内容",
            )
        except Exception as exc:
            conn.rollback()
            logger.warning(f"delete_admin execute 失败: {exc}")
            return self._build_standard_delete_result(mode=act_mode, error=str(exc))

    async def _restore_delete_action(
        self,
        *,
        mode: str,
        selector: Any,
        operation_id: str = "",
        requested_by: str = "",
        reason: str = "",
    ) -> Dict[str, Any]:
        del requested_by
        del reason
        assert self.metadata_store

        op_id = str(operation_id or "").strip()
        if op_id:
            operation = self.metadata_store.get_delete_operation(op_id)
            if operation is None:
                return {"success": False, "error": "operation 不存在"}
            return await self._restore_delete_operation(operation)

        act_mode = str(mode or "").strip().lower()
        if act_mode != "relation":
            return {"success": False, "error": "paragraph/entity/source 恢复必须提供 operation_id"}

        raw = self._selector_dict(selector)
        target = str(raw.get("query", "") or raw.get("target", "") or raw.get("hash", "") or "").strip()
        hashes = self._resolve_deleted_relation_hashes(target)
        if not hashes:
            return {"success": False, "error": "未命中可恢复关系"}
        result = await self._restore_relation_hashes(hashes)
        return {"success": bool(result.get("restored_count", 0) > 0), **result}

    async def _restore_delete_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        assert self.metadata_store
        items = operation.get("items") if isinstance(operation.get("items"), list) else []
        entity_payloads: Dict[str, Dict[str, Any]] = {}
        paragraph_payloads: Dict[str, Dict[str, Any]] = {}
        relation_payloads: Dict[str, Dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("item_type", "") or "").strip()
            item_hash = str(item.get("item_hash", "") or "").strip()
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
            if item_type == "entity" and item_hash:
                entity_payloads[item_hash] = payload
            elif item_type == "paragraph" and item_hash:
                paragraph_payloads[item_hash] = payload
            elif item_type == "relation" and item_hash:
                relation_payloads[item_hash] = payload

        restored_entities: List[str] = []
        restored_paragraphs: List[str] = []
        for hash_value, payload in entity_payloads.items():
            entity_row = payload.get("entity") if isinstance(payload.get("entity"), dict) else {}
            if entity_row:
                self.metadata_store.restore_entity_by_hash(hash_value)
                await self._ensure_entity_vector(entity_row)
                restored_entities.append(hash_value)
        for hash_value, payload in paragraph_payloads.items():
            paragraph_row = payload.get("paragraph") if isinstance(payload.get("paragraph"), dict) else {}
            if paragraph_row:
                self.metadata_store.restore_paragraph_by_hash(hash_value)
                await self._ensure_paragraph_vector(paragraph_row)
                restored_paragraphs.append(hash_value)

        restored_relations = await self._restore_relation_hashes(list(relation_payloads.keys()), payloads=relation_payloads, rebuild_graph=False, persist=False)

        conn = self.metadata_store.get_connection()
        cursor = conn.cursor()
        for payload in entity_payloads.values():
            for link in payload.get("paragraph_links") or []:
                paragraph_hash = str(link.get("paragraph_hash", "") or "").strip()
                entity_hash = str(link.get("entity_hash", "") or "").strip()
                mention_count = max(1, int(link.get("mention_count", 1) or 1))
                if not paragraph_hash or not entity_hash:
                    continue
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO paragraph_entities (paragraph_hash, entity_hash, mention_count)
                    VALUES (?, ?, ?)
                    """,
                    (paragraph_hash, entity_hash, mention_count),
                )
        for payload in paragraph_payloads.values():
            for link in payload.get("entity_links") or []:
                paragraph_hash = str(link.get("paragraph_hash", "") or "").strip()
                entity_hash = str(link.get("entity_hash", "") or "").strip()
                mention_count = max(1, int(link.get("mention_count", 1) or 1))
                if not paragraph_hash or not entity_hash:
                    continue
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO paragraph_entities (paragraph_hash, entity_hash, mention_count)
                    VALUES (?, ?, ?)
                    """,
                    (paragraph_hash, entity_hash, mention_count),
                )
            for relation_hash in self._tokens(payload.get("relation_hashes")):
                paragraph_hash = str((payload.get("paragraph") or {}).get("hash", "") or "").strip()
                if not paragraph_hash or not relation_hash:
                    continue
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO paragraph_relations (paragraph_hash, relation_hash)
                    VALUES (?, ?)
                    """,
                    (paragraph_hash, relation_hash),
                )
            self.metadata_store.restore_external_memory_refs(list(payload.get("external_refs") or []))
        conn.commit()

        sources = self._tokens(
            [
                str(((payload.get("paragraph") or {}).get("source", "") or "")).strip()
                for payload in paragraph_payloads.values()
            ]
        )
        if sources:
            self.metadata_store._enqueue_episode_source_rebuilds(sources, reason="delete_admin_restore")
        self._rebuild_graph_from_metadata()
        self._persist()
        summary = {
            "restored_entities": restored_entities,
            "restored_paragraphs": restored_paragraphs,
            "restored_relations": restored_relations.get("restored_hashes", []),
            "sources": sources,
        }
        self.metadata_store.mark_delete_operation_restored(str(operation.get("operation_id", "") or ""), summary=summary)
        return {
            "success": True,
            "operation_id": str(operation.get("operation_id", "") or ""),
            **summary,
            "restored_relation_count": restored_relations.get("restored_count", 0),
            "relation_failures": restored_relations.get("failures", []),
        }

    async def _purge_deleted_memory(self, *, grace_hours: Optional[float], limit: int) -> Dict[str, Any]:
        assert self.metadata_store
        orphan_cfg = self._cfg("memory.orphan", {}) or {}
        grace = float(grace_hours) if grace_hours is not None else max(
            1.0,
            float(orphan_cfg.get("sweep_grace_hours", 24.0) or 24.0),
        )
        cutoff = time.time() - grace * 3600.0
        deleted_relation_hashes = self.metadata_store.purge_deleted_relations(cutoff_time=cutoff, limit=limit)
        dead_paragraphs = self.metadata_store.sweep_deleted_items("paragraph", grace * 3600.0)
        paragraph_hashes = [str(item[0] or "").strip() for item in dead_paragraphs if str(item[0] or "").strip()]
        dead_entities = self.metadata_store.sweep_deleted_items("entity", grace * 3600.0)
        entity_hashes = [str(item[0] or "").strip() for item in dead_entities if str(item[0] or "").strip()]
        entity_names = [str(item[1] or "").strip() for item in dead_entities if str(item[1] or "").strip()]

        if paragraph_hashes:
            self.metadata_store.physically_delete_paragraphs(paragraph_hashes)
        if entity_hashes:
            self.metadata_store.physically_delete_entities(entity_hashes)
        if entity_names:
            self.graph_store.delete_nodes(entity_names)
        if self.vector_store is not None:
            vector_ids = self._merge_tokens(paragraph_hashes, entity_hashes, deleted_relation_hashes)
            if vector_ids:
                self.vector_store.delete(vector_ids)
        self._rebuild_graph_from_metadata()
        self._persist()
        return {
            "success": True,
            "grace_hours": grace,
            "purged_deleted_relations": deleted_relation_hashes,
            "purged_paragraph_hashes": paragraph_hashes,
            "purged_entity_hashes": entity_hashes,
            "purged_counts": {
                "relations": len(deleted_relation_hashes),
                "paragraphs": len(paragraph_hashes),
                "entities": len(entity_hashes),
            },
        }
