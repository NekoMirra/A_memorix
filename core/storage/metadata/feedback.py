"""记忆反馈任务与动作日志。"""

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


class MetadataFeedbackMixin:
    """记忆反馈任务与动作日志。"""

    def _feedback_task_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["query_snapshot"] = self._json_loads(data.pop("query_snapshot_json", None), {})
        data["decision_payload"] = self._json_loads(data.get("decision_json"), {})
        data["rollback_status"] = str(data.get("rollback_status", "") or "none").strip().lower() or "none"
        data["rollback_plan"] = self._json_loads(data.pop("rollback_plan_json", None), {})
        data["rollback_result"] = self._json_loads(data.pop("rollback_result_json", None), {})
        data["rollback_error"] = str(data.get("rollback_error", "") or "").strip()
        data["rollback_requested_by"] = str(data.get("rollback_requested_by", "") or "").strip()
        data["rollback_reason"] = str(data.get("rollback_reason", "") or "").strip()
        return data

    def _feedback_action_log_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["id"] = int(data.get("id", 0) or 0)
        data["task_id"] = int(data.get("task_id", 0) or 0)
        data["query_tool_id"] = str(data.get("query_tool_id", "") or "").strip()
        data["action_type"] = str(data.get("action_type", "") or "").strip()
        data["target_hash"] = str(data.get("target_hash", "") or "").strip()
        data["reason"] = str(data.get("reason", "") or "").strip()
        data["before_payload"] = self._json_loads(data.pop("before_json", None), {})
        data["after_payload"] = self._json_loads(data.pop("after_json", None), {})
        return data

    def get_feedback_task(self, query_tool_id: str) -> Optional[Dict[str, Any]]:
        token = str(query_tool_id or "").strip()
        if not token:
            return None
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM memory_feedback_tasks
            WHERE query_tool_id = ?
            LIMIT 1
            """,
            (token,),
        )
        row = cursor.fetchone()
        return self._feedback_task_row_to_dict(row) if row is not None else None

    def get_feedback_task_by_id(self, task_id: int) -> Optional[Dict[str, Any]]:
        if int(task_id or 0) <= 0:
            return None
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM memory_feedback_tasks
            WHERE id = ?
            LIMIT 1
            """,
            (int(task_id),),
        )
        row = cursor.fetchone()
        return self._feedback_task_row_to_dict(row) if row is not None else None

    def list_feedback_tasks(
        self,
        *,
        limit: int = 50,
        statuses: Optional[List[str]] = None,
        rollback_statuses: Optional[List[str]] = None,
        query: str = "",
    ) -> List[Dict[str, Any]]:
        safe_limit = max(1, int(limit or 50))
        params: List[Any] = []
        conditions: List[str] = []

        normalized_statuses = [
            str(item or "").strip().lower()
            for item in (statuses or [])
            if str(item or "").strip().lower() in {"pending", "running", "applied", "skipped", "error"}
        ]
        if normalized_statuses:
            placeholders = ",".join(["?"] * len(normalized_statuses))
            conditions.append(f"LOWER(COALESCE(status, '')) IN ({placeholders})")
            params.extend(normalized_statuses)

        normalized_rollback_statuses = [
            str(item or "").strip().lower()
            for item in (rollback_statuses or [])
            if str(item or "").strip().lower() in {"none", "running", "rolled_back", "error"}
        ]
        if normalized_rollback_statuses:
            placeholders = ",".join(["?"] * len(normalized_rollback_statuses))
            conditions.append(f"LOWER(COALESCE(rollback_status, 'none')) IN ({placeholders})")
            params.extend(normalized_rollback_statuses)

        query_token = str(query or "").strip().lower()
        if query_token:
            like_value = f"%{query_token}%"
            conditions.append(
                """
                (
                    LOWER(COALESCE(query_tool_id, '')) LIKE ?
                    OR LOWER(COALESCE(session_id, '')) LIKE ?
                    OR LOWER(COALESCE(query_snapshot_json, '')) LIKE ?
                    OR LOWER(COALESCE(decision_json, '')) LIKE ?
                    OR LOWER(COALESCE(last_error, '')) LIKE ?
                    OR LOWER(COALESCE(rollback_reason, '')) LIKE ?
                    OR LOWER(COALESCE(rollback_error, '')) LIKE ?
                )
                """
            )
            params.extend([like_value] * 7)

        where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(safe_limit)
        cursor = self._conn.cursor()
        cursor.execute(
            f"""
            SELECT *
            FROM memory_feedback_tasks
            {where_sql}
            ORDER BY query_timestamp DESC, id DESC
            LIMIT ?
            """,
            tuple(params),
        )
        return [self._feedback_task_row_to_dict(row) for row in cursor.fetchall()]

    def enqueue_feedback_task(
        self,
        *,
        query_tool_id: str,
        session_id: str,
        query_timestamp: float,
        due_at: float,
        query_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        tool_token = str(query_tool_id or "").strip()
        session_token = str(session_id or "").strip()
        if not tool_token or not session_token:
            return None

        now = datetime.now().timestamp()
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO memory_feedback_tasks (
                query_tool_id, session_id, query_timestamp, due_at, status, attempt_count,
                query_snapshot_json, decision_json, last_error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'pending', 0, ?, NULL, NULL, ?, ?)
            """,
            (
                tool_token,
                session_token,
                float(query_timestamp),
                float(due_at),
                self._json_dumps(query_snapshot or {}),
                now,
                now,
            ),
        )
        self._conn.commit()
        return self.get_feedback_task(tool_token)

    def update_feedback_task_rollback_plan(
        self,
        *,
        task_id: int,
        rollback_plan: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if int(task_id or 0) <= 0:
            return None
        cursor = self._conn.cursor()
        cursor.execute(
            """
            UPDATE memory_feedback_tasks
            SET rollback_plan_json = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                self._json_dumps(rollback_plan or {}),
                datetime.now().timestamp(),
                int(task_id),
            ),
        )
        self._conn.commit()
        return self.get_feedback_task_by_id(int(task_id))

    def fetch_due_feedback_tasks(
        self,
        *,
        limit: int = 20,
        now: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        safe_limit = max(1, int(limit))
        now_ts = self._as_optional_float(now)
        if now_ts is None:
            now_ts = datetime.now().timestamp()

        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM memory_feedback_tasks
            WHERE due_at <= ?
              AND status IN ('pending', 'running')
            ORDER BY due_at ASC, id ASC
            LIMIT ?
            """,
            (now_ts, safe_limit),
        )
        return [self._feedback_task_row_to_dict(row) for row in cursor.fetchall()]

    def mark_feedback_task_running(self, task_id: int) -> Optional[Dict[str, Any]]:
        if int(task_id or 0) <= 0:
            return None
        now = datetime.now().timestamp()
        cursor = self._conn.cursor()
        cursor.execute(
            """
            UPDATE memory_feedback_tasks
            SET status = 'running',
                attempt_count = COALESCE(attempt_count, 0) + 1,
                updated_at = ?
            WHERE id = ?
              AND status IN ('pending', 'running')
            """,
            (now, int(task_id)),
        )
        self._conn.commit()
        cursor.execute(
            """
            SELECT *
            FROM memory_feedback_tasks
            WHERE id = ?
            LIMIT 1
            """,
            (int(task_id),),
        )
        row = cursor.fetchone()
        return self._feedback_task_row_to_dict(row) if row is not None else None

    def finalize_feedback_task(
        self,
        *,
        task_id: int,
        status: str,
        decision_payload: Optional[Dict[str, Any]] = None,
        last_error: str = "",
    ) -> Optional[Dict[str, Any]]:
        final_status = str(status or "").strip().lower()
        if final_status not in {"applied", "skipped", "error"}:
            raise ValueError(f"不支持的反馈任务结束状态: {status}")
        if int(task_id or 0) <= 0:
            return None

        now = datetime.now().timestamp()
        cursor = self._conn.cursor()
        cursor.execute(
            """
            UPDATE memory_feedback_tasks
            SET status = ?,
                decision_json = ?,
                last_error = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                final_status,
                self._json_dumps(decision_payload or {}),
                str(last_error or "").strip() or None,
                now,
                int(task_id),
            ),
        )
        self._conn.commit()
        cursor.execute(
            """
            SELECT *
            FROM memory_feedback_tasks
            WHERE id = ?
            LIMIT 1
            """,
            (int(task_id),),
        )
        row = cursor.fetchone()
        return self._feedback_task_row_to_dict(row) if row is not None else None

    def mark_feedback_task_rollback_running(
        self,
        *,
        task_id: int,
        requested_by: str = "",
        reason: str = "",
    ) -> Optional[Dict[str, Any]]:
        if int(task_id or 0) <= 0:
            return None
        now = datetime.now().timestamp()
        cursor = self._conn.cursor()
        cursor.execute(
            """
            UPDATE memory_feedback_tasks
            SET rollback_status = 'running',
                rollback_requested_by = ?,
                rollback_reason = ?,
                rollback_error = NULL,
                rollback_requested_at = ?,
                updated_at = ?
            WHERE id = ?
              AND LOWER(COALESCE(status, '')) = 'applied'
              AND LOWER(COALESCE(rollback_status, 'none')) IN ('none', 'error')
            """,
            (
                str(requested_by or "").strip() or None,
                str(reason or "").strip() or None,
                now,
                now,
                int(task_id),
            ),
        )
        self._conn.commit()
        if int(cursor.rowcount or 0) <= 0:
            return None
        return self.get_feedback_task_by_id(int(task_id))

    def finalize_feedback_task_rollback(
        self,
        *,
        task_id: int,
        rollback_status: str,
        rollback_result: Optional[Dict[str, Any]] = None,
        rollback_error: str = "",
    ) -> Optional[Dict[str, Any]]:
        if int(task_id or 0) <= 0:
            return None
        final_status = str(rollback_status or "").strip().lower()
        if final_status not in {"none", "rolled_back", "error"}:
            raise ValueError(f"不支持的反馈任务回退状态: {rollback_status}")
        now = datetime.now().timestamp()
        cursor = self._conn.cursor()
        cursor.execute(
            """
            UPDATE memory_feedback_tasks
            SET rollback_status = ?,
                rollback_result_json = ?,
                rollback_error = ?,
                rolled_back_at = CASE WHEN ? = 'rolled_back' THEN ? ELSE rolled_back_at END,
                updated_at = ?
            WHERE id = ?
            """,
            (
                final_status,
                self._json_dumps(rollback_result or {}),
                str(rollback_error or "").strip() or None,
                final_status,
                now,
                now,
                int(task_id),
            ),
        )
        self._conn.commit()
        return self.get_feedback_task_by_id(int(task_id))

    def append_feedback_action_log(
        self,
        *,
        task_id: int,
        query_tool_id: str,
        action_type: str,
        target_hash: str = "",
        before_payload: Optional[Dict[str, Any]] = None,
        after_payload: Optional[Dict[str, Any]] = None,
        reason: str = "",
    ) -> Optional[Dict[str, Any]]:
        if int(task_id or 0) <= 0:
            return None
        query_token = str(query_tool_id or "").strip()
        if not query_token:
            return None

        now = datetime.now().timestamp()
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO memory_feedback_action_logs (
                task_id, query_tool_id, action_type, target_hash,
                before_json, after_json, reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(task_id),
                query_token,
                str(action_type or "").strip() or "unknown",
                str(target_hash or "").strip() or None,
                self._json_dumps(before_payload) if isinstance(before_payload, dict) else None,
                self._json_dumps(after_payload) if isinstance(after_payload, dict) else None,
                str(reason or "").strip() or None,
                now,
            ),
        )
        self._conn.commit()
        return {
            "id": int(cursor.lastrowid or 0),
            "task_id": int(task_id),
            "query_tool_id": query_token,
            "action_type": str(action_type or "").strip() or "unknown",
            "target_hash": str(target_hash or "").strip(),
            "before_json": self._json_dumps(before_payload) if isinstance(before_payload, dict) else None,
            "after_json": self._json_dumps(after_payload) if isinstance(after_payload, dict) else None,
            "reason": str(reason or "").strip(),
            "created_at": now,
        }

    def list_feedback_action_logs(self, task_id: int) -> List[Dict[str, Any]]:
        if int(task_id or 0) <= 0:
            return []
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, task_id, query_tool_id, action_type, target_hash, before_json, after_json, reason, created_at
            FROM memory_feedback_action_logs
            WHERE task_id = ?
            ORDER BY id ASC
            """,
            (int(task_id),),
        )
        return [self._feedback_action_log_row_to_dict(row) for row in cursor.fetchall()]
