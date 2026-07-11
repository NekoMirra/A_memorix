"""外部记忆引用映射。"""

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


class MetadataExternalRefsMixin:
    """外部记忆引用映射。"""

    def get_external_memory_ref(self, external_id: str) -> Optional[Dict[str, Any]]:
        """按 external_id 查询外部记忆映射。"""
        token = str(external_id or "").strip()
        if not token:
            return None

        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT external_id, paragraph_hash, source_type, created_at, metadata_json
            FROM external_memory_refs
            WHERE external_id = ?
            LIMIT 1
            """,
            (token,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        payload = dict(row)
        raw_metadata = payload.get("metadata_json")
        if raw_metadata:
            try:
                payload["metadata"] = json.loads(raw_metadata)
            except Exception:
                payload["metadata"] = {}
        else:
            payload["metadata"] = {}
        return payload

    def upsert_external_memory_ref(
        self,
        *,
        external_id: str,
        paragraph_hash: str,
        source_type: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """注册 external_id 到段落哈希的幂等映射。"""
        external_token = str(external_id or "").strip()
        paragraph_token = str(paragraph_hash or "").strip()
        if not external_token:
            raise ValueError("external_id 不能为空")
        if not paragraph_token:
            raise ValueError("paragraph_hash 不能为空")

        now = datetime.now().timestamp()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO external_memory_refs (
                external_id, paragraph_hash, source_type, created_at, metadata_json
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(external_id) DO UPDATE SET
                paragraph_hash = excluded.paragraph_hash,
                source_type = excluded.source_type,
                metadata_json = excluded.metadata_json
            """,
            (
                external_token,
                paragraph_token,
                str(source_type or "").strip() or None,
                now,
                metadata_json,
            ),
        )
        self._conn.commit()
        return self.get_external_memory_ref(external_token) or {
            "external_id": external_token,
            "paragraph_hash": paragraph_token,
            "source_type": str(source_type or "").strip(),
            "created_at": now,
            "metadata": metadata or {},
        }

    def list_external_memory_refs_by_paragraphs(self, paragraph_hashes: List[str]) -> List[Dict[str, Any]]:
        hashes = [str(item or "").strip() for item in (paragraph_hashes or []) if str(item or "").strip()]
        if not hashes:
            return []
        placeholders = ",".join(["?"] * len(hashes))
        cursor = self._conn.cursor()
        cursor.execute(
            f"""
            SELECT external_id, paragraph_hash, source_type, created_at, metadata_json
            FROM external_memory_refs
            WHERE paragraph_hash IN ({placeholders})
            ORDER BY created_at ASC, external_id ASC
            """,
            tuple(hashes),
        )
        items: List[Dict[str, Any]] = []
        for row in cursor.fetchall():
            payload = dict(row)
            payload["metadata"] = self._json_loads(payload.get("metadata_json"), {})
            items.append(payload)
        return items

    def delete_external_memory_refs_by_paragraphs(self, paragraph_hashes: List[str]) -> List[Dict[str, Any]]:
        items = self.list_external_memory_refs_by_paragraphs(paragraph_hashes)
        hashes = [str(item or "").strip() for item in (paragraph_hashes or []) if str(item or "").strip()]
        if not hashes:
            return items
        placeholders = ",".join(["?"] * len(hashes))
        cursor = self._conn.cursor()
        cursor.execute(
            f"DELETE FROM external_memory_refs WHERE paragraph_hash IN ({placeholders})",
            tuple(hashes),
        )
        self._conn.commit()
        return items

    def restore_external_memory_refs(self, refs: List[Dict[str, Any]]) -> int:
        count = 0
        for item in refs or []:
            external_id = str(item.get("external_id", "") or "").strip()
            paragraph_hash = str(item.get("paragraph_hash", "") or "").strip()
            if not external_id or not paragraph_hash:
                continue
            created_at = float(item.get("created_at") or datetime.now().timestamp())
            metadata_json = self._json_dumps(item.get("metadata") or {})
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT INTO external_memory_refs (
                    external_id, paragraph_hash, source_type, created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(external_id) DO UPDATE SET
                    paragraph_hash = excluded.paragraph_hash,
                    source_type = excluded.source_type,
                    created_at = excluded.created_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    external_id,
                    paragraph_hash,
                    str(item.get("source_type", "") or "").strip() or None,
                    created_at,
                    metadata_json,
                ),
            )
            count += max(0, int(cursor.rowcount or 0))
        self._conn.commit()
        return count
