"""段落元数据 CRUD 与相关队列。"""

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


class MetadataParagraphsMixin:
    """段落元数据 CRUD 与相关队列。"""

    def add_paragraph(
        self,
        content: str,
        vector_index: Optional[int] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        knowledge_type: str = "mixed",
        time_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        添加段落

        Args:
            content: 段落内容
            vector_index: 向量索引
            source: 来源
            metadata: 额外元数据
            knowledge_type: 知识类型 (narrative/factual/quote/structured/mixed)
            time_meta: 时间元信息 (event_time/event_time_start/event_time_end/...)

        Returns:
            段落哈希值
        """
        content_normalized = normalize_text(content)
        hash_value = compute_hash(content_normalized)
        resolved_knowledge_type = validate_stored_knowledge_type(knowledge_type)

        now = datetime.now().timestamp()
        word_count = len(content_normalized.split())
        normalized_time = normalize_time_meta(time_meta)

        cursor = self._conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO paragraphs
                (
                    hash, content, vector_index, created_at, updated_at, metadata, source, word_count,
                    event_time, event_time_start, event_time_end, time_granularity, time_confidence,
                    knowledge_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                hash_value,
                content,
                vector_index,
                now,
                now,
                pickle.dumps(metadata or {}),
                source,
                word_count,
                normalized_time.get("event_time"),
                normalized_time.get("event_time_start"),
                normalized_time.get("event_time_end"),
                normalized_time.get("time_granularity"),
                normalized_time.get("time_confidence", 1.0),
                resolved_knowledge_type.value,
            ))
            self._conn.commit()
            try:
                self.enqueue_episode_source_rebuild(
                    source=source,
                    reason="paragraph_added",
                )
            except Exception as e:
                logger.warning(f"Episode source 重建入队失败: hash={hash_value[:16]}..., err={e}")
            logger.debug(
                f"添加段落: hash={hash_value[:16]}..., words={word_count}, type={resolved_knowledge_type.value}"
            )
            return hash_value
        except sqlite3.IntegrityError:
            logger.debug(f"段落已存在: {hash_value[:16]}...")
            # 尝试复活
            self.revive_if_deleted(paragraph_hashes=[hash_value])
            return hash_value

    def get_paragraph(self, hash_value: str) -> Optional[Dict[str, Any]]:
        """
        获取段落

        Args:
            hash_value: 段落哈希

        Returns:
            段落信息字典，不存在则返回None
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT * FROM paragraphs WHERE hash = ?
        """, (hash_value,))
        row = cursor.fetchone()

        if row:
            return self._row_to_dict(row, "paragraph")
        return None

    def update_paragraph_time_meta(
        self,
        paragraph_hash: str,
        time_meta: Dict[str, Any],
    ) -> bool:
        """
        更新段落时间元信息。
        """
        normalized = normalize_time_meta(time_meta)
        if not normalized:
            return False
        source_to_rebuild = self._get_sources_for_paragraph_hashes(
            [paragraph_hash],
            include_deleted=True,
        )

        updates: List[str] = []
        params: List[Any] = []
        for key in [
            "event_time",
            "event_time_start",
            "event_time_end",
            "time_granularity",
            "time_confidence",
        ]:
            if key in normalized:
                updates.append(f"{key} = ?")
                params.append(normalized[key])

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.now().timestamp())
        params.append(paragraph_hash)

        cursor = self._conn.cursor()
        cursor.execute(
            f"UPDATE paragraphs SET {', '.join(updates)} WHERE hash = ?",
            tuple(params),
        )
        self._conn.commit()
        changed = cursor.rowcount > 0
        if changed:
            self._enqueue_episode_source_rebuilds(
                source_to_rebuild,
                reason="paragraph_time_updated",
            )
        return changed

    def query_paragraphs_temporal(
        self,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        person: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
        allow_created_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        查询时序命中的段落（区间相交语义）。
        """
        if limit <= 0:
            return []

        effective_start = "COALESCE(p.event_time_start, p.event_time, p.event_time_end"
        effective_end = "COALESCE(p.event_time_end, p.event_time, p.event_time_start"
        if allow_created_fallback:
            effective_start += ", p.created_at)"
            effective_end += ", p.created_at)"
        else:
            effective_start += ")"
            effective_end += ")"

        conditions = ["(p.is_deleted IS NULL OR p.is_deleted = 0)"]
        params: List[Any] = []

        if source:
            conditions.append("p.source = ?")
            params.append(source)

        if person:
            conditions.append(
                """
                EXISTS (
                    SELECT 1
                    FROM paragraph_entities pe
                    JOIN entities e ON e.hash = pe.entity_hash
                    WHERE pe.paragraph_hash = p.hash
                      AND LOWER(e.name) LIKE ?
                )
                """
            )
            params.append(f"%{str(person).strip().lower()}%")

        if start_ts is not None and end_ts is not None:
            conditions.append(f"({effective_end} >= ? AND {effective_start} <= ?)")
            params.extend([start_ts, end_ts])
        elif start_ts is not None:
            conditions.append(f"({effective_end} >= ?)")
            params.append(start_ts)
        elif end_ts is not None:
            conditions.append(f"({effective_start} <= ?)")
            params.append(end_ts)

        where_sql = " AND ".join(conditions)
        sql = f"""
            SELECT p.*
            FROM paragraphs p
            WHERE {where_sql}
            ORDER BY {effective_end} DESC, p.updated_at DESC
            LIMIT ?
        """
        params.append(limit)

        cursor = self._conn.cursor()
        cursor.execute(sql, tuple(params))
        return [self._row_to_dict(row, "paragraph") for row in cursor.fetchall()]

    def get_paragraphs_by_source(self, source: str) -> List[Dict[str, Any]]:
        """
        按来源获取段落

        Args:
            source: 来源标识符

        Returns:
            段落列表
        """
        return self.query("SELECT * FROM paragraphs WHERE source = ?", (source,))

    def get_all_sources(self) -> List[Dict[str, Any]]:
        """
        获取所有来源文件统计信息
    
        Returns:
            来源列表 [{'source': 'name', 'count': int, 'last_updated': timestamp}]
        """
        cursor = self._conn.cursor()
        # 排除 source 为 NULL 或空的记录
        cursor.execute("""
            SELECT source, COUNT(*) as count, MAX(created_at) as last_updated 
            FROM paragraphs 
            WHERE source IS NOT NULL AND source != ''
              AND (is_deleted IS NULL OR is_deleted = 0)
            GROUP BY source
            ORDER BY last_updated DESC
        """)
    
        results = []
        for row in cursor.fetchall():
            results.append({
                "source": row[0],
                "count": row[1],
                "last_updated": row[2]
            })
        return results

    def search_paragraphs_by_content(self, content_query: str) -> List[Dict[str, Any]]:
        """按内容模糊搜索段落"""
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT * FROM paragraphs WHERE content LIKE ?
        """, (f"%{content_query}%",))
        return [self._row_to_dict(row, "paragraph") for row in cursor.fetchall()]

    def delete_paragraph(self, hash_value: str) -> bool:
        """
        删除段落（级联删除相关关联）

        Args:
            hash_value: 段落哈希

        Returns:
            是否成功删除
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            DELETE FROM paragraphs WHERE hash = ?
        """, (hash_value,))
        self._conn.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"删除段落: {hash_value[:16]}...")

        return deleted

    def delete_paragraph_atomic(self, paragraph_hash: str) -> Dict[str, Any]:
        """
        两阶段删除段落：DB 事务内计算 + 提交后执行清理

        Args:
            paragraph_hash: 段落哈希

        Returns:
            cleanup_plan: 包含需要后续从 Vector/GraphStore 中移除的 ID 列表
        """
        cleanup_plan = {
            "paragraph_hash": paragraph_hash,
            "vector_id_to_remove": None,
            "edges_to_remove": [],  # (src, tgt) 元组列表 (fallback)
            "relation_prune_ops": [],  # (subject, object, relation_hash) 精准裁剪
            "episode_sources_to_rebuild": [],
        }

        cursor = self._conn.cursor()
        try:
            # === Phase 1: DB Transaction (可回滚) ===
            # 使用 IMMEDIATE 模式，一旦开启事务立即锁定 DB (防止其他写操作插队导致幻读)
            cursor.execute("BEGIN IMMEDIATE")

            # 1. [快照] 获取候选关系
            cursor.execute("SELECT relation_hash FROM paragraph_relations WHERE paragraph_hash = ?", (paragraph_hash,))
            candidate_relations = [row[0] for row in cursor.fetchall()]

            # 2. [快照] 确认该段落存在并记录 ID 用于向量删除
            cursor.execute("SELECT hash, source FROM paragraphs WHERE hash = ?", (paragraph_hash,))
            paragraph_row = cursor.fetchone()
            if paragraph_row:
                cleanup_plan["vector_id_to_remove"] = paragraph_hash
                cleanup_plan["episode_sources_to_rebuild"] = self._dedupe_episode_sources(
                    [paragraph_row["source"]]
                )

            # 3. [主删除] 删除段落 (触发 CASCADE 删 paragraph_relations)
            cursor.execute("DELETE FROM paragraphs WHERE hash = ?", (paragraph_hash,))

            # 4. [计算孤儿]
            orphaned_hashes = []
            for rel_hash in candidate_relations:
                count = cursor.execute(
                    "SELECT count(*) FROM paragraph_relations WHERE relation_hash = ?",
                    (rel_hash,)
                ).fetchone()[0]

                if count == 0:
                    # 是孤儿：记录边信息以便后续删 Graph
                    cursor.execute("SELECT subject, object FROM relations WHERE hash = ?", (rel_hash,))
                    rel_info = cursor.fetchone()
                    if rel_info:
                        s_val, o_val = rel_info[0], rel_info[1]
                        cleanup_plan["relation_prune_ops"].append((s_val, o_val, rel_hash))

                        # 仅当 (subject, object) 不再有任何关系时，才计划删整条边（兼容旧实现）。
                        sibling_count = cursor.execute(
                            """
                            SELECT count(*) FROM relations
                            WHERE LOWER(TRIM(subject)) = LOWER(TRIM(?))
                              AND LOWER(TRIM(object)) = LOWER(TRIM(?))
                              AND hash != ?
                            """,
                            (s_val, o_val, rel_hash)
                        ).fetchone()[0]
                        if sibling_count == 0:
                            cleanup_plan["edges_to_remove"].append((s_val, o_val))

                    orphaned_hashes.append(rel_hash)

            # 5. [DB清理] 删除孤儿关系记录
            if orphaned_hashes:
                placeholders = ','.join(['?'] * len(orphaned_hashes))
                cursor.execute(f"DELETE FROM relations WHERE hash IN ({placeholders})", orphaned_hashes)

            self._conn.commit()
            if cleanup_plan["episode_sources_to_rebuild"]:
                self._enqueue_episode_source_rebuilds(
                    cleanup_plan["episode_sources_to_rebuild"],
                    reason="paragraph_deleted",
                )
            if cleanup_plan["vector_id_to_remove"]:
                logger.debug(f"原子删除段落成功: {paragraph_hash}, 计划清理 {len(orphaned_hashes)} 个孤儿关系")
            return cleanup_plan

        except Exception as e:
            self._conn.rollback()
            logger.error(f"DB Transaction failed: {e}")
            raise e

    def set_permanence(self, hash_value: str, item_type: str, is_permanent: bool) -> bool:
        """设置永久记忆标记"""
        table_map = {
            "paragraph": "paragraphs",
            "relation": "relations",
        }
        if item_type not in table_map:
            raise ValueError(f"类型 {item_type} 不支持设置永久性")
        
        cursor = self._conn.cursor()
        cursor.execute(f"""
            UPDATE {table_map[item_type]}
            SET is_permanent = ?
            WHERE hash = ?
        """, (1 if is_permanent else 0, hash_value))
        self._conn.commit()
    
        if cursor.rowcount > 0:
            logger.debug(f"设置永久记忆: {item_type}/{hash_value[:8]} -> {is_permanent}")
            return True
        return False

    def record_access(self, hash_value: str, item_type: str) -> bool:
        """记录访问（更新时间和次数）"""
        table_map = {
            "paragraph": "paragraphs",
            "relation": "relations",
        }
        if item_type not in table_map:
            return False
        
        now = datetime.now().timestamp()
        cursor = self._conn.cursor()
        cursor.execute(f"""
            UPDATE {table_map[item_type]}
            SET last_accessed = ?, access_count = access_count + 1
            WHERE hash = ?
        """, (now, hash_value))
        self._conn.commit()
        return cursor.rowcount > 0

    def update_vector_index(
        self,
        item_type: str,
        hash_value: str,
        vector_index: int,
    ) -> bool:
        """
        更新向量索引

        Args:
            item_type: 类型（paragraph/entity/relation）
            hash_value: 哈希值
            vector_index: 向量索引

        Returns:
            是否成功更新
        """
        valid_types = ["paragraph", "entity", "relation"]
        if item_type not in valid_types:
            raise ValueError(f"无效的类型: {item_type}")

        table_map = {
            "paragraph": "paragraphs",
            "entity": "entities",
            "relation": "relations",
        }

        cursor = self._conn.cursor()
        cursor.execute(f"""
            UPDATE {table_map[item_type]}
            SET vector_index = ?
            WHERE hash = ?
        """, (vector_index, hash_value))
        self._conn.commit()

        return cursor.rowcount > 0

    def query(
        self,
        sql: str,
        params: Optional[Tuple] = None,
    ) -> List[Dict[str, Any]]:
        """
        执行自定义查询

        Args:
            sql: SQL语句
            params: 参数

        Returns:
            查询结果列表
        """
        cursor = self._conn.cursor()
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)

        return [dict(row) for row in cursor.fetchall()]

    def count_paragraphs(self, include_deleted: bool = False, only_deleted: bool = False) -> int:
        """
        获取段落数量
        """
        cursor = self._conn.cursor()
        if only_deleted:
            cursor.execute("SELECT COUNT(*) FROM paragraphs WHERE is_deleted = 1")
            return cursor.fetchone()[0]
        if include_deleted:
            cursor.execute("SELECT COUNT(*) FROM paragraphs")
            return cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM paragraphs WHERE is_deleted = 0")
        return cursor.fetchone()[0]

    def backfill_temporal_metadata_from_created_at(
        self,
        *,
        limit: int = 100000,
        dry_run: bool = False,
        no_created_fallback: bool = False,
    ) -> Dict[str, int]:
        """回填段落 event_time 字段（created_at 兜底）。"""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT hash, created_at, source
            FROM paragraphs
            WHERE (event_time IS NULL AND event_time_start IS NULL AND event_time_end IS NULL)
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(max(1, limit)),),
        )
        rows = cursor.fetchall()
        candidates = len(rows)
        if dry_run:
            return {"candidates": candidates, "updated": 0}
        if no_created_fallback:
            return {"candidates": candidates, "updated": 0}

        updated = 0
        touched_sources: List[str] = []
        for row in rows:
            created_at = row["created_at"]
            if created_at is None:
                continue
            cursor.execute(
                """
                UPDATE paragraphs
                SET event_time = ?, time_granularity = ?, time_confidence = ?, updated_at = ?
                WHERE hash = ?
                """,
                (float(created_at), "day", 0.2, float(created_at), row["hash"]),
            )
            if cursor.rowcount > 0:
                updated += 1
                touched_sources.append(row["source"])
        self._conn.commit()
        if updated > 0:
            self._enqueue_episode_source_rebuilds(
                touched_sources,
                reason="paragraph_time_backfill",
            )
        return {"candidates": candidates, "updated": updated}

    def restore_paragraph_by_hash(self, paragraph_hash: str) -> bool:
        """恢复软删除段落。"""
        cursor = self._conn.cursor()
        cursor.execute(
            "UPDATE paragraphs SET is_deleted=0, deleted_at=NULL WHERE hash=?",
            (str(paragraph_hash),),
        )
        changed = cursor.rowcount > 0
        if changed:
            self._conn.commit()
        return changed

    def enqueue_paragraph_vector_backfill(
        self,
        paragraph_hash: str,
        *,
        created_at: Optional[float] = None,
        error: str = "",
    ) -> None:
        """登记段落向量回填任务。"""
        token = str(paragraph_hash or "").strip()
        if not token:
            return

        now = datetime.now().timestamp()
        created_ts = float(created_at) if created_at is not None else now
        error_text = str(error or "").strip() or None

        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO paragraph_vector_backfill (
                paragraph_hash, status, retry_count, last_error, created_at, updated_at
            ) VALUES (?, 'pending', 0, ?, ?, ?)
            ON CONFLICT(paragraph_hash) DO UPDATE SET
                status = CASE
                    WHEN paragraph_vector_backfill.status = 'done' THEN 'done'
                    ELSE 'pending'
                END,
                last_error = CASE
                    WHEN paragraph_vector_backfill.status = 'done' THEN paragraph_vector_backfill.last_error
                    ELSE excluded.last_error
                END,
                created_at = COALESCE(paragraph_vector_backfill.created_at, excluded.created_at),
                updated_at = excluded.updated_at
            """,
            (token, error_text, created_ts, now),
        )
        self._conn.commit()

    def fetch_paragraph_vector_backfill_batch(
        self,
        limit: int = 64,
        max_retry: int = 5,
    ) -> List[Dict[str, Any]]:
        """获取段落向量回填批次。"""
        safe_limit = max(1, int(limit))
        safe_retry = max(0, int(max_retry))
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT paragraph_hash, status, retry_count, last_error, created_at, updated_at
            FROM paragraph_vector_backfill
            WHERE status = 'pending'
               OR (status = 'failed' AND retry_count < ?)
            ORDER BY updated_at ASC
            LIMIT ?
            """,
            (safe_retry, safe_limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def mark_paragraph_vector_backfill_running(self, hashes: List[str]) -> None:
        """批量标记段落回填任务为 running。"""
        if not hashes:
            return
        now = datetime.now().timestamp()
        cursor = self._conn.cursor()
        uniq = list(dict.fromkeys([str(h or "").strip() for h in hashes if str(h or "").strip()]))
        if not uniq:
            return
        chunk_size = 500
        for i in range(0, len(uniq), chunk_size):
            chunk = uniq[i:i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            cursor.execute(
                f"""
                UPDATE paragraph_vector_backfill
                SET status = 'running', updated_at = ?
                WHERE paragraph_hash IN ({placeholders})
                  AND status IN ('pending', 'failed')
                """,
                [now] + chunk,
            )
        self._conn.commit()

    def mark_paragraph_vector_backfill_done(self, hashes: List[str]) -> None:
        """批量标记段落回填任务为 done。"""
        if not hashes:
            return
        now = datetime.now().timestamp()
        cursor = self._conn.cursor()
        uniq = list(dict.fromkeys([str(h or "").strip() for h in hashes if str(h or "").strip()]))
        if not uniq:
            return
        chunk_size = 500
        for i in range(0, len(uniq), chunk_size):
            chunk = uniq[i:i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            cursor.execute(
                f"""
                UPDATE paragraph_vector_backfill
                SET status = 'done',
                    last_error = NULL,
                    updated_at = ?
                WHERE paragraph_hash IN ({placeholders})
                """,
                [now] + chunk,
            )
        self._conn.commit()

    def mark_paragraph_vector_backfill_failed(self, paragraph_hash: str, error: str = "") -> None:
        """标记单个段落回填任务失败并累加重试。"""
        token = str(paragraph_hash or "").strip()
        if not token:
            return
        now = datetime.now().timestamp()
        cursor = self._conn.cursor()
        cursor.execute(
            """
            UPDATE paragraph_vector_backfill
            SET status = 'failed',
                retry_count = COALESCE(retry_count, 0) + 1,
                last_error = ?,
                updated_at = ?
            WHERE paragraph_hash = ?
            """,
            (str(error or ""), now, token),
        )
        self._conn.commit()

    def get_paragraph_vector_backfill_status_counts(self) -> Dict[str, int]:
        """统计段落回填任务状态。"""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM paragraph_vector_backfill
            GROUP BY status
            """
        )
        counts = {"pending": 0, "running": 0, "failed": 0, "done": 0}
        for row in cursor.fetchall():
            status = str(row["status"] or "").strip().lower()
            if status in counts:
                counts[status] = int(row["count"] or 0)
        return counts

    def upsert_paragraph_stale_relation_mark(
        self,
        *,
        paragraph_hash: str,
        relation_hash: str,
        query_tool_id: str = "",
        task_id: Optional[int] = None,
        reason: str = "",
    ) -> Optional[Dict[str, Any]]:
        paragraph_token = str(paragraph_hash or "").strip()
        relation_token = str(relation_hash or "").strip()
        if not paragraph_token or not relation_token:
            return None

        now = datetime.now().timestamp()
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO paragraph_stale_relation_marks (
                paragraph_hash, relation_hash, query_tool_id, task_id, reason, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paragraph_hash, relation_hash) DO UPDATE SET
                query_tool_id = excluded.query_tool_id,
                task_id = excluded.task_id,
                reason = excluded.reason,
                updated_at = excluded.updated_at
            """,
            (
                paragraph_token,
                relation_token,
                str(query_tool_id or "").strip() or None,
                int(task_id) if int(task_id or 0) > 0 else None,
                str(reason or "").strip() or None,
                now,
                now,
            ),
        )
        self._conn.commit()
        return {
            "paragraph_hash": paragraph_token,
            "relation_hash": relation_token,
            "query_tool_id": str(query_tool_id or "").strip(),
            "task_id": int(task_id or 0) if int(task_id or 0) > 0 else None,
            "reason": str(reason or "").strip(),
            "updated_at": now,
        }

    def get_paragraph_stale_relation_marks_batch(
        self,
        paragraph_hashes: Sequence[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        normalized: List[str] = []
        seen = set()
        for item in paragraph_hashes or []:
            token = str(item or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            normalized.append(token)
        if not normalized:
            return {}

        placeholders = ",".join(["?"] * len(normalized))
        cursor = self._conn.cursor()
        cursor.execute(
            f"""
            SELECT paragraph_hash, relation_hash, query_tool_id, task_id, reason, created_at, updated_at
            FROM paragraph_stale_relation_marks
            WHERE paragraph_hash IN ({placeholders})
            ORDER BY updated_at DESC, paragraph_hash ASC, relation_hash ASC
            """,
            tuple(normalized),
        )
        grouped: Dict[str, List[Dict[str, Any]]] = {token: [] for token in normalized}
        for row in cursor.fetchall():
            payload = {
                "paragraph_hash": str(row["paragraph_hash"] or "").strip(),
                "relation_hash": str(row["relation_hash"] or "").strip(),
                "query_tool_id": str(row["query_tool_id"] or "").strip(),
                "task_id": int(row["task_id"] or 0) if row["task_id"] is not None else None,
                "reason": str(row["reason"] or "").strip(),
                "created_at": self._as_optional_float(row["created_at"]),
                "updated_at": self._as_optional_float(row["updated_at"]),
            }
            grouped.setdefault(payload["paragraph_hash"], []).append(payload)
        return grouped

    def count_paragraph_stale_relation_marks(self) -> int:
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM paragraph_stale_relation_marks")
        row = cursor.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def delete_paragraph_stale_relation_marks(
        self,
        marks: Sequence[Tuple[str, str]],
    ) -> int:
        normalized: List[Tuple[str, str]] = []
        seen: set[Tuple[str, str]] = set()
        for paragraph_hash, relation_hash in marks or []:
            paragraph_token = str(paragraph_hash or "").strip()
            relation_token = str(relation_hash or "").strip()
            if not paragraph_token or not relation_token:
                continue
            key = (paragraph_token, relation_token)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(key)
        if not normalized:
            return 0

        cursor = self._conn.cursor()
        deleted = 0
        for paragraph_hash, relation_hash in normalized:
            cursor.execute(
                """
                DELETE FROM paragraph_stale_relation_marks
                WHERE paragraph_hash = ? AND relation_hash = ?
                """,
                (paragraph_hash, relation_hash),
            )
            deleted += int(cursor.rowcount or 0)
        self._conn.commit()
        return deleted
