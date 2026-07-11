"""统计汇总。"""

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


class MetadataStatsMixin:
    """统计汇总。"""

    def get_statistics(self) -> Dict[str, int]:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        cursor = self._conn.cursor()

        stats = {}

        # 段落数量
        cursor.execute("SELECT COUNT(*) FROM paragraphs")
        stats["paragraph_count"] = cursor.fetchone()[0]

        # 实体数量
        cursor.execute("SELECT COUNT(*) FROM entities")
        stats["entity_count"] = cursor.fetchone()[0]

        # 关系数量
        cursor.execute("SELECT COUNT(*) FROM relations")
        stats["relation_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM paragraph_stale_relation_marks")
        stats["stale_paragraph_mark_count"] = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM person_profile_refresh_queue WHERE status IN ('pending', 'running', 'failed')"
        )
        stats["person_profile_refresh_pending_count"] = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM person_profile_refresh_queue WHERE status = 'failed'"
        )
        stats["person_profile_refresh_failed_count"] = cursor.fetchone()[0]

        # 总词数
        cursor.execute("SELECT SUM(word_count) FROM paragraphs")
        result = cursor.fetchone()[0]
        stats["total_words"] = result if result else 0

        return stats

    def get_knowledge_type_distribution(self) -> Dict[str, int]:
        """获取段落知识类型分布。"""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT knowledge_type, COUNT(*) as count
            FROM paragraphs
            WHERE is_deleted = 0
            GROUP BY knowledge_type
            """
        )
        result: Dict[str, int] = {}
        for row in cursor.fetchall():
            type_name = row[0] if row[0] else "未分类"
            result[str(type_name)] = int(row[1] or 0)
        return result

    def get_memory_status_summary(self, now_ts: Optional[float] = None) -> Dict[str, int]:
        """聚合 memory status 统计。"""
        now_ts = float(now_ts) if now_ts is not None else datetime.now().timestamp()
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM relations WHERE is_inactive = 0")
        active_count = int(cursor.fetchone()[0] or 0)
        cursor.execute("SELECT COUNT(*) FROM relations WHERE is_inactive = 1")
        inactive_count = int(cursor.fetchone()[0] or 0)
        cursor.execute("SELECT COUNT(*) FROM deleted_relations")
        deleted_count = int(cursor.fetchone()[0] or 0)
        cursor.execute("SELECT COUNT(*) FROM relations WHERE is_pinned = 1")
        pinned_count = int(cursor.fetchone()[0] or 0)
        cursor.execute("SELECT COUNT(*) FROM relations WHERE protected_until > ?", (now_ts,))
        ttl_count = int(cursor.fetchone()[0] or 0)
        return {
            "active_count": active_count,
            "inactive_count": inactive_count,
            "deleted_count": deleted_count,
            "pinned_count": pinned_count,
            "temp_protected_count": ttl_count,
        }
