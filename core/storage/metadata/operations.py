"""V5 删除操作记录。"""

from __future__ import annotations

import json
import pickle
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.common.logger import get_logger

from ...utils.hash import compute_hash, normalize_text
from ...utils.time_parser import normalize_time_meta
from ..knowledge_types import (
    KnowledgeType,
    allowed_knowledge_type_values,
    resolve_stored_knowledge_type,
    validate_stored_knowledge_type,
)
from .constants import RUNTIME_AUTO_MIGRATION_MIN_SCHEMA_VERSION, SCHEMA_VERSION

try:
    import jieba  # type: ignore

    HAS_JIEBA = True
except Exception:
    jieba = None
    HAS_JIEBA = False

logger = get_logger("A_Memorix.MetadataStore")


class MetadataOperationsMixin:
    """V5 删除操作记录。"""

    def record_v5_operation(
        self,
        *,
        action: str,
        target: str,
        resolved_hashes: List[str],
        reason: str = "",
        updated_by: str = "",
        result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        operation_id = f"v5_{uuid.uuid4().hex}"
        created_at = datetime.now().timestamp()
        payload = {
            "operation_id": operation_id,
            "action": str(action or "").strip(),
            "target": str(target or "").strip(),
            "reason": str(reason or "").strip(),
            "updated_by": str(updated_by or "").strip(),
            "created_at": created_at,
            "resolved_hashes": [str(item or "").strip() for item in (resolved_hashes or []) if str(item or "").strip()],
            "result": result or {},
        }
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO memory_v5_operations (
                operation_id, action, target, reason, updated_by, created_at, resolved_hashes_json, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                operation_id,
                payload["action"],
                payload["target"] or None,
                payload["reason"] or None,
                payload["updated_by"] or None,
                created_at,
                self._json_dumps(payload["resolved_hashes"]),
                self._json_dumps(payload["result"]),
            ),
        )
        self._conn.commit()
        return payload

    def create_delete_operation(
        self,
        *,
        mode: str,
        selector: Any,
        items: List[Dict[str, Any]],
        reason: str = "",
        requested_by: str = "",
        status: str = "executed",
        summary: Optional[Dict[str, Any]] = None,
        operation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        op_id = str(operation_id or f"del_{uuid.uuid4().hex}").strip()
        created_at = datetime.now().timestamp()
        normalized_items: List[Dict[str, Any]] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("item_type", "") or "").strip()
            if not item_type:
                continue
            normalized_items.append(
                {
                    "item_type": item_type,
                    "item_hash": str(item.get("item_hash", "") or "").strip() or None,
                    "item_key": str(item.get("item_key", "") or item.get("item_hash", "") or "").strip() or None,
                    "payload": item.get("payload") if isinstance(item.get("payload"), dict) else {},
                }
            )

        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO delete_operations (
                operation_id, mode, selector, reason, requested_by, status, created_at, restored_at, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """,
            (
                op_id,
                str(mode or "").strip(),
                self._json_dumps(selector if selector is not None else {}),
                str(reason or "").strip() or None,
                str(requested_by or "").strip() or None,
                str(status or "executed").strip(),
                created_at,
                self._json_dumps(summary or {}),
            ),
        )
        if normalized_items:
            cursor.executemany(
                """
                INSERT INTO delete_operation_items (
                    operation_id, item_type, item_hash, item_key, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        op_id,
                        item["item_type"],
                        item["item_hash"],
                        item["item_key"],
                        self._json_dumps(item["payload"]),
                        created_at,
                    )
                    for item in normalized_items
                ],
            )
        self._conn.commit()
        return self.get_delete_operation(op_id) or {
            "operation_id": op_id,
            "mode": str(mode or "").strip(),
            "selector": selector,
            "reason": str(reason or "").strip(),
            "requested_by": str(requested_by or "").strip(),
            "status": str(status or "executed").strip(),
            "created_at": created_at,
            "summary": summary or {},
            "items": normalized_items,
        }

    def mark_delete_operation_restored(
        self,
        operation_id: str,
        *,
        summary: Optional[Dict[str, Any]] = None,
    ) -> bool:
        token = str(operation_id or "").strip()
        if not token:
            return False
        cursor = self._conn.cursor()
        cursor.execute(
            """
            UPDATE delete_operations
            SET status = ?, restored_at = ?, summary_json = ?
            WHERE operation_id = ?
            """,
            (
                "restored",
                datetime.now().timestamp(),
                self._json_dumps(summary or {}),
                token,
            ),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def list_delete_operations(self, *, limit: int = 50, mode: str = "") -> List[Dict[str, Any]]:
        cursor = self._conn.cursor()
        params: List[Any] = []
        where = ""
        mode_token = str(mode or "").strip().lower()
        if mode_token:
            where = "WHERE LOWER(mode) = ?"
            params.append(mode_token)
        params.append(max(1, int(limit or 50)))
        cursor.execute(
            f"""
            SELECT operation_id, mode, selector, reason, requested_by, status, created_at, restored_at, summary_json
            FROM delete_operations
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        items: List[Dict[str, Any]] = []
        for row in cursor.fetchall():
            payload = dict(row)
            payload["selector"] = self._json_loads(payload.get("selector"), {})
            payload["summary"] = self._json_loads(payload.get("summary_json"), {})
            items.append(payload)
        return items

    def get_delete_operation(self, operation_id: str) -> Optional[Dict[str, Any]]:
        token = str(operation_id or "").strip()
        if not token:
            return None
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT operation_id, mode, selector, reason, requested_by, status, created_at, restored_at, summary_json
            FROM delete_operations
            WHERE operation_id = ?
            LIMIT 1
            """,
            (token,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        payload = dict(row)
        payload["selector"] = self._json_loads(payload.get("selector"), {})
        payload["summary"] = self._json_loads(payload.get("summary_json"), {})

        cursor.execute(
            """
            SELECT item_type, item_hash, item_key, payload_json, created_at
            FROM delete_operation_items
            WHERE operation_id = ?
            ORDER BY id ASC
            """,
            (token,),
        )
        payload["items"] = [
            {
                "item_type": str(item["item_type"] or ""),
                "item_hash": str(item["item_hash"] or ""),
                "item_key": str(item["item_key"] or ""),
                "payload": self._json_loads(item["payload_json"], {}),
                "created_at": item["created_at"],
            }
            for item in cursor.fetchall()
        ]
        return payload
