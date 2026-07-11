"""关系元数据、向量状态与恢复。"""

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


class MetadataRelationsMixin:
    """关系元数据、向量状态与恢复。"""

    def add_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
        vector_index: Optional[int] = None,
        confidence: float = 1.0,
        source_paragraph: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        添加关系
    
        Args:
            subject: 主语
            predicate: 谓语
            obj: 宾语
            vector_index: 向量索引
            confidence: 置信度
            source_paragraph: 来源段落哈希
            metadata: 额外元数据
        
        Returns:
            关系哈希值
        """
        hash_value = self.compute_relation_hash(subject, predicate, obj)

        now = datetime.now().timestamp()
    
        # 记录原始 display name 到 metadata (如果需要的话，或者直接存到 DB 字段)
        # 这里我们直接存入 subject, predicate, object 字段，
        # 注意：如果 DB 里已存在该关系 (hash 相同)，则不会更新这些字段，保留第一次的拼写。
    
        cursor = self._conn.cursor()
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO relations
                (hash, subject, predicate, object, vector_index, confidence, created_at, source_paragraph, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                hash_value,
                subject,  # 原始拼写
                predicate,
                obj,
                vector_index,
                confidence,
                now,
                source_paragraph, # 这里的 source_paragraph 仅作为 "首次发现地" 记录，也可留空
                pickle.dumps(metadata or {}),
            ))
            self._conn.commit()
        
            if cursor.rowcount > 0:
                logger.debug(f"添加关系: {subject} -{predicate}-> {obj}")
            else:
                logger.debug(f"关系已存在: {subject} -{predicate}-> {obj}")

            # 3. 建立来源关联 (幂等)
            # 无论关系是新创建的还是已存在的，只要提供了 source_paragraph，都要建立连接
            if source_paragraph:
                self.link_paragraph_relation(source_paragraph, hash_value)
            
            return hash_value
        
        except sqlite3.IntegrityError as e:
            logger.warning(f"添加关系异常: {e}")
            return hash_value

    def compute_relation_hash(self, subject: str, predicate: str, obj: str) -> str:
        """
        计算 relation 的稳定 hash，不执行写入。
        """
        # 1. 规范化输入
        s_canon = self._canonicalize_name(subject)
        p_canon = self._canonicalize_name(predicate)
        o_canon = self._canonicalize_name(obj)
    
        if not all([s_canon, p_canon, o_canon]):
             raise ValueError("Relation components cannot be empty")

        # 2. 计算组合哈希
        # 公式: md5(s|p|o)
        relation_key = f"{s_canon}|{p_canon}|{o_canon}"
        return compute_hash(relation_key)

    def get_relation(self, hash_value: str, include_inactive: bool = True) -> Optional[Dict[str, Any]]:
        """
        获取关系

        Args:
            hash_value: 关系哈希

        Returns:
            关系信息字典，不存在则返回None
        """
        cursor = self._conn.cursor()
        if include_inactive:
            cursor.execute(
                """
                SELECT * FROM relations WHERE hash = ?
                """,
                (hash_value,),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM relations
                WHERE hash = ?
                  AND (is_inactive IS NULL OR is_inactive = 0)
                """,
                (hash_value,),
            )
        row = cursor.fetchone()

        if row:
            return self._row_to_dict(row, "relation")
        return None

    def get_relations(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        include_inactive: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        查询关系（大小写不敏感）
    
        Args:
            subject: 主语（可选）
            predicate: 谓语（可选）
            object: 宾语（可选）
        
        Returns:
            关系列表
        """
        # 构建查询条件
        conditions = []
        params = []
    
        if subject:
            conditions.append("LOWER(subject) = ?")
            params.append(self._canonicalize_name(subject))
        if predicate:
            conditions.append("LOWER(predicate) = ?")
            params.append(self._canonicalize_name(predicate))
        if object:
            conditions.append("LOWER(object) = ?")
            params.append(self._canonicalize_name(object))
        if not include_inactive:
            conditions.append("(is_inactive IS NULL OR is_inactive = 0)")
        
        sql = "SELECT * FROM relations"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        cursor = self._conn.cursor()
        cursor.execute(sql, tuple(params))
    
        return [self._row_to_dict(row, "relation") for row in cursor.fetchall()]

    def get_all_triples(self) -> List[Tuple[str, str, str, str]]:
        """
        高效获取所有三元组 (subject, predicate, object, hash)
        直接返回元组，跳过字典转换和pickle反序列化，用于构建 V5 Map 缓存。
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT subject, predicate, object, hash FROM relations")
        return list(cursor.fetchall())

    def link_paragraph_relation(
        self,
        paragraph_hash: str,
        relation_hash: str,
    ) -> bool:
        """
        关联段落和关系 (幂等)
        """
        cursor = self._conn.cursor()
        try:
            # 使用 INSERT OR IGNORE 避免重复报错
            cursor.execute("""
                INSERT OR IGNORE INTO paragraph_relations
                (paragraph_hash, relation_hash)
                VALUES (?, ?)
            """, (paragraph_hash, relation_hash))
            self._conn.commit()
            self._enqueue_episode_source_rebuilds(
                self._get_sources_for_paragraph_hashes([paragraph_hash], include_deleted=True),
                reason="paragraph_relation_linked",
            )
            return True
        except sqlite3.IntegrityError:
            return False

    def get_paragraph_relations(self, paragraph_hash: str) -> List[Dict[str, Any]]:
        """
        获取段落的所有关系

        Args:
            paragraph_hash: 段落哈希

        Returns:
            关系列表
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT r.* FROM relations r
            JOIN paragraph_relations pr ON r.hash = pr.relation_hash
            WHERE pr.paragraph_hash = ?
        """, (paragraph_hash,))

        return [self._row_to_dict(row, "relation") for row in cursor.fetchall()]

    def get_paragraph_hashes_by_relation_hashes(
        self,
        relation_hashes: List[str],
    ) -> Dict[str, List[str]]:
        normalized: List[str] = []
        seen = set()
        for item in relation_hashes or []:
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
            SELECT pr.relation_hash, pr.paragraph_hash
            FROM paragraph_relations pr
            JOIN paragraphs p ON p.hash = pr.paragraph_hash
            WHERE pr.relation_hash IN ({placeholders})
              AND (p.is_deleted IS NULL OR p.is_deleted = 0)
            ORDER BY pr.relation_hash ASC, p.updated_at DESC, p.created_at DESC, pr.paragraph_hash ASC
            """,
            tuple(normalized),
        )
        grouped: Dict[str, List[str]] = {token: [] for token in normalized}
        for row in cursor.fetchall():
            relation_hash = str(row["relation_hash"] or "").strip()
            paragraph_hash = str(row["paragraph_hash"] or "").strip()
            if not relation_hash or not paragraph_hash:
                continue
            if paragraph_hash not in grouped.setdefault(relation_hash, []):
                grouped[relation_hash].append(paragraph_hash)
        return grouped

    def get_paragraphs_by_relation(self, relation_hash: str) -> List[Dict[str, Any]]:
        """
        获取支持指定关系的所有段落

        Args:
            relation_hash: 关系哈希

        Returns:
            段落列表
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT p.*
            FROM paragraphs p
            JOIN paragraph_relations pr ON p.hash = pr.paragraph_hash
            WHERE pr.relation_hash = ?
        """, (relation_hash,))

        return [self._row_to_dict(row, "paragraph") for row in cursor.fetchall()]

    def delete_relation(self, hash_value: str) -> bool:
        """
        删除关系（级联删除相关关联）

        Args:
            hash_value: 关系哈希

        Returns:
            是否成功删除
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            DELETE FROM relations WHERE hash = ?
        """, (hash_value,))
        self._conn.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"删除关系: {hash_value[:16]}...")

        return deleted

    def set_relation_vector_state(
        self,
        hash_value: str,
        state: str,
        error: Optional[str] = None,
        bump_retry: bool = False,
    ) -> bool:
        """
        更新关系向量状态。
        """
        state_norm = str(state or "").strip().lower()
        if state_norm not in {"none", "pending", "ready", "failed"}:
            raise ValueError(f"无效 vector_state: {state}")

        now = datetime.now().timestamp()
        err_text = (str(error).strip() if error is not None else None)
        if err_text:
            err_text = err_text[:500]
        clear_error = state_norm in {"none", "pending", "ready"}

        cursor = self._conn.cursor()
        if bump_retry:
            cursor.execute(
                """
                UPDATE relations
                SET vector_state = ?,
                    vector_updated_at = ?,
                    vector_error = ?,
                    vector_retry_count = COALESCE(vector_retry_count, 0) + 1
                WHERE hash = ?
                """,
                (state_norm, now, None if clear_error else err_text, hash_value),
            )
        else:
            cursor.execute(
                """
                UPDATE relations
                SET vector_state = ?,
                    vector_updated_at = ?,
                    vector_error = ?
                WHERE hash = ?
                """,
                (state_norm, now, None if clear_error else err_text, hash_value),
            )
        self._conn.commit()
        return cursor.rowcount > 0

    def list_relations_by_vector_state(
        self,
        states: List[str],
        limit: int = 200,
        max_retry: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        根据向量状态列出关系，用于回填任务。
        """
        normalized_states = [
            str(s or "").strip().lower()
            for s in (states or [])
            if str(s or "").strip()
        ]
        normalized_states = [
            s for s in normalized_states
            if s in {"none", "pending", "ready", "failed"}
        ]
        if not normalized_states:
            return []

        placeholders = ",".join(["?"] * len(normalized_states))
        params: List[Any] = list(normalized_states)
        sql = f"""
            SELECT hash, subject, predicate, object, confidence, source_paragraph,
                   vector_state, vector_updated_at, vector_error, vector_retry_count, created_at
            FROM relations
            WHERE vector_state IN ({placeholders})
        """
        if max_retry is not None:
            sql += " AND COALESCE(vector_retry_count, 0) < ?"
            params.append(int(max_retry))
        sql += " ORDER BY COALESCE(vector_updated_at, created_at, 0) ASC LIMIT ?"
        params.append(max(1, int(limit)))

        cursor = self._conn.cursor()
        cursor.execute(sql, tuple(params))
        return [self._row_to_dict(row, "relation") for row in cursor.fetchall()]

    def count_relations_by_vector_state(self) -> Dict[str, int]:
        """
        统计关系向量状态分布。
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT COALESCE(vector_state, 'none') AS state, COUNT(*) AS cnt
            FROM relations
            GROUP BY COALESCE(vector_state, 'none')
            """
        )
        result: Dict[str, int] = {"none": 0, "pending": 0, "ready": 0, "failed": 0}
        total = 0
        for row in cursor.fetchall():
            state = str(row["state"] or "none").lower()
            count = int(row["cnt"] or 0)
            if state not in result:
                result[state] = 0
            result[state] += count
            total += count
        result["total"] = total
        return result

    def count_relations(self, include_deleted: bool = False, only_deleted: bool = False) -> int:
        """
        获取关系数量
        """
        cursor = self._conn.cursor()
        if only_deleted:
            cursor.execute("SELECT COUNT(*) FROM deleted_relations")
            return cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM relations")
        active_count = cursor.fetchone()[0]
        if not include_deleted:
            return active_count
        cursor.execute("SELECT COUNT(*) FROM deleted_relations")
        deleted_count = cursor.fetchone()[0]
        return int(active_count) + int(deleted_count)

    def get_relations_subject_object_map(self, hashes: List[str]) -> Dict[str, Tuple[str, str]]:
        """批量获取关系 hash 对应的 (subject, object)。"""
        if not hashes:
            return {}
        cursor = self._conn.cursor()
        placeholders = ",".join(["?"] * len(hashes))
        cursor.execute(
            f"SELECT hash, subject, object FROM relations WHERE hash IN ({placeholders})",
            hashes,
        )
        return {str(row[0]): (str(row[1]), str(row[2])) for row in cursor.fetchall()}

    def get_relation_db_snapshot(self) -> Tuple[int, float, str]:
        """返回关系快照：(relation_count, max_created_at, max_hash)。"""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT
                COUNT(*) AS relation_count,
                COALESCE(MAX(created_at), 0) AS max_created_at,
                COALESCE(MAX(hash), '') AS max_hash
            FROM relations
            """
        )
        row = cursor.fetchone()
        if not row:
            return (0, 0.0, "")
        return (
            int(row[0] or 0),
            float(row[1] or 0.0),
            str(row[2] or ""),
        )

    def search_relations_by_subject_or_object(
        self,
        query: str,
        *,
        limit: int = 5,
        include_deleted: bool = False,
    ) -> List[Dict[str, Any]]:
        """按 subject/object 模糊查询关系。"""
        q = str(query or "").strip()
        if not q:
            return []
        max_limit = int(max(1, limit))
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM relations
            WHERE subject LIKE ? OR object LIKE ?
            LIMIT ?
            """,
            (f"%{q}%", f"%{q}%", max_limit),
        )
        rows = [self._row_to_dict(row, "relation") for row in cursor.fetchall()]
        if rows or not include_deleted:
            return rows

        cursor.execute(
            """
            SELECT *
            FROM deleted_relations
            WHERE subject LIKE ? OR object LIKE ?
            LIMIT ?
            """,
            (f"%{q}%", f"%{q}%", max_limit),
        )
        return [self._row_to_dict(row, "relation") for row in cursor.fetchall()]

    def list_hashes(self, table: str) -> List[str]:
        """安全枚举指定表的 hash 列。"""
        allowed = {"paragraphs", "entities", "relations", "deleted_relations"}
        token = str(table or "").strip().lower()
        if token not in allowed:
            raise ValueError(f"unsupported table for list_hashes: {table}")
        cursor = self._conn.cursor()
        cursor.execute(f"SELECT hash FROM {token}")
        return [str(row[0]) for row in cursor.fetchall()]

    def get_orphan_deleted_relation_hashes(self, limit: int = 200) -> List[str]:
        """获取 deleted_relations 中已不在 relations 的孤儿 hash。"""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT d.hash
            FROM deleted_relations d
            LEFT JOIN relations r ON r.hash = d.hash
            WHERE r.hash IS NULL
            LIMIT ?
            """,
            (int(max(1, limit)),),
        )
        return [str(row[0]) for row in cursor.fetchall()]

    def resolve_relation_hash_alias(
        self,
        value: str,
        *,
        include_deleted: bool = False,
    ) -> List[str]:
        """
        解析关系哈希输入：
        - 64位：直接校验存在性
        - 32位：通过 relation_hash_aliases 唯一映射
        """
        token = str(value or "").strip().lower()
        if not token:
            return []
        if len(token) == 64 and all(ch in "0123456789abcdef" for ch in token):
            cursor = self._conn.cursor()
            cursor.execute("SELECT 1 FROM relations WHERE hash = ? LIMIT 1", (token,))
            if cursor.fetchone():
                return [token]
            if include_deleted:
                cursor.execute("SELECT 1 FROM deleted_relations WHERE hash = ? LIMIT 1", (token,))
                if cursor.fetchone():
                    return [token]
            return []

        if len(token) != 32 or not all(ch in "0123456789abcdef" for ch in token):
            return []

        cursor = self._conn.cursor()
        cursor.execute("SELECT hash FROM relation_hash_aliases WHERE alias32 = ?", (token,))
        row = cursor.fetchone()
        if not row:
            return []
        resolved = str(row[0])
        return [resolved]

    def rebuild_relation_hash_aliases(self) -> Dict[str, Any]:
        """重建 32 位 relation hash 别名映射。"""
        cursor = self._conn.cursor()
        # 历史库兜底：缺表时先创建，避免迁移过程直接中断。
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relation_hash_aliases (
                alias32 TEXT PRIMARY KEY,
                hash TEXT NOT NULL
            )
        """)
        cursor.execute("DELETE FROM relation_hash_aliases")

        cursor.execute("SELECT hash FROM relations")
        hashes = [str(r[0]) for r in cursor.fetchall()]
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='deleted_relations'"
        )
        has_deleted_relations = cursor.fetchone() is not None
        if has_deleted_relations:
            cursor.execute("SELECT hash FROM deleted_relations")
            hashes.extend(str(r[0]) for r in cursor.fetchall())

        alias_map: Dict[str, str] = {}
        conflicts: Dict[str, set[str]] = {}
        for h in hashes:
            if len(h) != 64:
                continue
            alias = h[:32]
            old = alias_map.get(alias)
            if old is None:
                alias_map[alias] = h
            elif old != h:
                conflicts.setdefault(alias, set()).update({old, h})

        for alias, full_hash in alias_map.items():
            if alias in conflicts:
                continue
            cursor.execute(
                "INSERT INTO relation_hash_aliases(alias32, hash) VALUES (?, ?)",
                (alias, full_hash),
            )
        self._conn.commit()
        return {
            "inserted": len(alias_map) - len(conflicts),
            "conflict_count": len(conflicts),
            "conflicts": sorted(conflicts.keys()),
        }

    def search_relation_hashes_by_text(self, query: str, limit: int = 5) -> List[str]:
        """按 relation 内容模糊查询 hash。"""
        q = str(query or "").strip()
        if not q:
            return []
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT hash FROM relations WHERE subject LIKE ? OR object LIKE ? LIMIT ?",
            (f"%{q}%", f"%{q}%", int(max(1, limit))),
        )
        return [str(row[0]) for row in cursor.fetchall()]

    def search_deleted_relation_hashes_by_text(self, query: str, limit: int = 5) -> List[str]:
        """按 deleted_relations 内容模糊查询 hash。"""
        q = str(query or "").strip()
        if not q:
            return []
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT hash FROM deleted_relations WHERE subject LIKE ? OR object LIKE ? LIMIT ?",
            (f"%{q}%", f"%{q}%", int(max(1, limit))),
        )
        return [str(row[0]) for row in cursor.fetchall()]

    def update_relation_timestamp(self, hash_value: str, access_count_delta: int = 1) -> None:
        """更新关系的访问时间和计数"""
        now = datetime.now().timestamp()
    
        # 同时更新 last_accessed (旧) 和 last_reinforced (V5)
    
        cursor = self._conn.cursor()
        cursor.execute("""
            UPDATE relations
            SET last_accessed = ?,
                access_count = access_count + ?
            WHERE hash = ?
        """, (now, access_count_delta, hash_value))
        self._conn.commit()

    def get_relation_status_batch(self, hashes: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        批量获取关系状态 (V5)
    
        Args:
            hashes: 关系哈希列表
        
        Returns:
            Dict[hash, status_dict]
            status_dict 包含: is_inactive, weight(confidence), is_pinned, protected_until, last_reinforced, inactive_since
        """
        if not hashes:
            return {}
        
        placeholders = ",".join(["?"] * len(hashes))
        cursor = self._conn.cursor()
        cursor.execute(f"""
            SELECT hash, is_inactive, confidence, is_pinned, protected_until, last_reinforced, inactive_since
            FROM relations
            WHERE hash IN ({placeholders})
        """, hashes)
    
        result = {}
        for row in cursor.fetchall():
            result[row["hash"]] = {
                "is_inactive": bool(row["is_inactive"]),
                "weight": row["confidence"],
                "is_pinned": bool(row["is_pinned"]),
                "protected_until": row["protected_until"],
                "last_reinforced": row["last_reinforced"],
                "inactive_since": row["inactive_since"]
            }
        return result

    def mark_relations_active(self, hashes: List[str], boost_weight: Optional[float] = None) -> None:
        """
        批量标记关系为活跃 (Active/Revive)
    
        Args:
            hashes: 关系哈希列表
            boost_weight: 如果提供，将设置 confidence = max(confidence, boost_weight)
        """
        if not hashes:
            return
        
        placeholders = ",".join(["?"] * len(hashes))
        cursor = self._conn.cursor()
    
        if boost_weight is not None:
            cursor.execute(f"""
                UPDATE relations
                SET is_inactive = 0,
                    inactive_since = NULL,
                    confidence = MAX(confidence, ?)
                WHERE hash IN ({placeholders})
            """, (boost_weight, *hashes))
        else:
             cursor.execute(f"""
                UPDATE relations
                SET is_inactive = 0,
                    inactive_since = NULL
                WHERE hash IN ({placeholders})
            """, hashes)
        
        self._conn.commit()

    def update_relations_protection(
        self, 
        hashes: List[str], 
        protected_until: Optional[float] = None, 
        is_pinned: Optional[bool] = None,
        last_reinforced: Optional[float] = None
    ) -> None:
        """
        批量更新关系保护状态
        """
        if not hashes:
            return
        
        updates = []
        params = []
    
        if protected_until is not None:
            updates.append("protected_until = ?")
            params.append(protected_until)
        if is_pinned is not None:
            updates.append("is_pinned = ?")
            params.append(1 if is_pinned else 0)
        if last_reinforced is not None:
            updates.append("last_reinforced = ?")
            params.append(last_reinforced)
        
        if not updates:
            return

        sql_set = ", ".join(updates)
        placeholders = ",".join(["?"] * len(hashes))
    
        params.extend(hashes)
    
        cursor = self._conn.cursor()
        cursor.execute(f"""
            UPDATE relations
            SET {sql_set}
            WHERE hash IN ({placeholders})
        """, params)
        self._conn.commit()

    def get_prune_candidates(self, cutoff_time: float, limit: int = 1000) -> List[str]:
        """
        获取待修剪候选 (已过冷冻保留期)
    
        Args:
            cutoff_time: 截止时间 (now - 冷冻时长)
            limit: 限制数量
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT hash FROM relations
            WHERE is_inactive = 1 
            AND inactive_since < ?
            LIMIT ?
        """, (cutoff_time, limit))
        return [row[0] for row in cursor.fetchall()]

    def backup_and_delete_relations(self, hashes: List[str]) -> int:
        """
        备份并删除关系 (Prune)
    
        Returns:
            删除的数量
        """
        if not hashes:
            return 0
        
        placeholders = ",".join(["?"] * len(hashes))
        now = datetime.now().timestamp()
    
        cursor = self._conn.cursor()
        try:
            # 1. 备份
            cursor.execute(f"""
                INSERT OR REPLACE INTO deleted_relations 
                (hash, subject, predicate, object, vector_index, confidence, created_at, 
                 vector_state, vector_updated_at, vector_error, vector_retry_count,
                 source_paragraph, metadata, is_permanent, last_accessed, access_count,
                 is_inactive, inactive_since, is_pinned, protected_until, last_reinforced, deleted_at)
                SELECT 
                 hash, subject, predicate, object, vector_index, confidence, created_at, 
                 vector_state, vector_updated_at, vector_error, vector_retry_count,
                 source_paragraph, metadata, is_permanent, last_accessed, access_count,
                 is_inactive, inactive_since, is_pinned, protected_until, last_reinforced, ?
                FROM relations
                WHERE hash IN ({placeholders})
            """, (now, *hashes))
        
            # 2. 删除 (级联删除会自动处理 paragraph_relations 关联)
            cursor.execute(f"""
                DELETE FROM relations
                WHERE hash IN ({placeholders})
            """, hashes)
        
            deleted_count = cursor.rowcount
            self._conn.commit()
            return deleted_count
        
        except Exception as e:
            logger.error(f"备份删除失败: {e}")
            self._conn.rollback()
            return 0

    def restore_relation_metadata(self, hash_value: str) -> Optional[Dict[str, Any]]:
        """
        从回收站恢复关系元数据
    
        Returns:
            恢复后的关系数据 (字典)，失败返回 None
        """
        cursor = self._conn.cursor()
        try:
            # 1. 查询备份数据
            cursor.execute("SELECT * FROM deleted_relations WHERE hash = ?", (hash_value,))
            row = cursor.fetchone()
            if not row:
                return None
            
            data = dict(row)
            # 移除 deleted_at 字段
            if "deleted_at" in data:
                del data["deleted_at"]
            
            # 2. 插入回 relations 表
            # 动态构建 SQL 以适应字段变化
            columns = list(data.keys())
            placeholders = ",".join(["?"] * len(columns))
            cols_str = ",".join(columns)
            values = list(data.values())
        
            cursor.execute(f"""
                INSERT OR REPLACE INTO relations ({cols_str})
                VALUES ({placeholders})
            """, values)
        
            # 3. 从备份表删除
            cursor.execute("DELETE FROM deleted_relations WHERE hash = ?", (hash_value,))
        
            self._conn.commit()
            return self._row_to_dict(row, "relation") # 使用助手函数将原始行转换为字典
        
        except Exception as e:
            logger.error(f"恢复关系失败: {hash_value} - {e}")
            self._conn.rollback()
            return None

    def restore_relation(self, hash_value: str) -> Optional[Dict[str, Any]]:
        """兼容旧调用名：恢复关系。"""
        return self.restore_relation_metadata(hash_value)

    def restore_relation_status_from_snapshot(
        self,
        hash_value: str,
        snapshot: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        token = str(hash_value or "").strip()
        if not token or not isinstance(snapshot, dict):
            return None

        current = self.get_relation_status_batch([token]).get(token)
        if current is None:
            restored = self.restore_relation(token)
            if restored is None:
                return None

        cursor = self._conn.cursor()
        cursor.execute(
            """
            UPDATE relations
            SET is_inactive = ?,
                confidence = ?,
                is_pinned = ?,
                protected_until = ?,
                last_reinforced = ?,
                inactive_since = ?
            WHERE hash = ?
            """,
            (
                1 if bool(snapshot.get("is_inactive")) else 0,
                float(snapshot.get("weight", 0.0) or 0.0),
                1 if bool(snapshot.get("is_pinned")) else 0,
                self._as_optional_float(snapshot.get("protected_until")),
                self._as_optional_float(snapshot.get("last_reinforced")),
                self._as_optional_float(snapshot.get("inactive_since")),
                token,
            ),
        )
        self._conn.commit()
        return self.get_relation_status_batch([token]).get(token)

    def get_protected_relations_hashes(self) -> List[str]:
        """获取所有受保护关系的哈希 (Pinned 或 Protected Until > Now)"""
        now = datetime.now().timestamp()
    
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT hash FROM relations
            WHERE is_pinned = 1 OR protected_until > ?
        """, (now,))
    
        return [row[0] for row in cursor.fetchall()]

    def get_deleted_relations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取回收站中的关系记录"""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM deleted_relations ORDER BY deleted_at DESC LIMIT ?", (limit,))
        data = []
        for row in cursor.fetchall():
             d = dict(row)
             # 是否需要解码元数据？是的，与普通行相同
             if "metadata" in d and d["metadata"]:
                 try:
                     d["metadata"] = pickle.loads(d["metadata"])
                 except Exception:
                     d["metadata"] = {}
             data.append(d)
        return data

    def get_deleted_relation(self, hash_value: str) -> Optional[Dict[str, Any]]:
        """获取单条回收站记录"""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM deleted_relations WHERE hash = ?", (hash_value,))
        row = cursor.fetchone()
        if not row: return None
    
        d = dict(row)
        if "metadata" in d and d["metadata"]:
             try:
                 d["metadata"] = pickle.loads(d["metadata"])
             except Exception:
                 d["metadata"] = {}
        return d

    def reinforce_relations(self, hashes: List[str]) -> None:
        """强化关系 (更新 last_reinforced, is_inactive=0)"""
        if not hashes: return
        now = datetime.now().timestamp()
    
        cursor = self._conn.cursor()
        # Batch update? chunking
        chunk_size = 500
        for i in range(0, len(hashes), chunk_size):
            chunk = hashes[i:i+chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            sql = f"""
                UPDATE relations 
                SET last_reinforced = ?, is_inactive = 0, inactive_since = NULL
                WHERE hash IN ({placeholders})
            """
            cursor.execute(sql, [now] + chunk)
        
        self._conn.commit()

    def mark_relations_inactive(self, hashes: List[str], inactive_since: Optional[float] = None) -> None:
        """标记关系为非活跃 (Freeze)。兼容显式 inactive_since 或默认当前时间。"""
        if not hashes:
            return
        mark_time = inactive_since if inactive_since is not None else datetime.now().timestamp()
    
        cursor = self._conn.cursor()
        chunk_size = 500
        for i in range(0, len(hashes), chunk_size):
            chunk = hashes[i:i+chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            sql = f"""
                UPDATE relations 
                SET is_inactive = 1, inactive_since = ?
                WHERE hash IN ({placeholders})
            """
            cursor.execute(sql, [mark_time] + chunk)
        
        self._conn.commit()

    def protect_relations(
        self, 
        hashes: List[str], 
        is_pinned: bool = False, 
        ttl_seconds: float = 0
    ) -> None:
        """
        设置保护状态
        """
        if not hashes: return
        now = datetime.now().timestamp()
        protected_until = (now + ttl_seconds) if ttl_seconds > 0 else 0
    
        cursor = self._conn.cursor()
        chunk_size = 500
        for i in range(0, len(hashes), chunk_size):
            chunk = hashes[i:i+chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
        
            # 由于 is_pinned 和 protected_until 是分开的，如果请求固定（pin），我们会同时更新这两项，
            # 但通常用户要么切换固定状态，要么设置 TTL。
            # 如果 is_pinned=True，TTL 通常就不重要了。
            # 但目前的逻辑是正交处理它们的。
        
            # 如果用户取消固定 (is_pinned=False)，我们是否应该尊重已设置的 TTL？
            # 当前的 API 会同时设置这两项。
        
            sql = f"""
                UPDATE relations 
                SET is_pinned = ?, protected_until = ?
                WHERE hash IN ({placeholders})
            """
            cursor.execute(sql, [is_pinned, protected_until] + chunk)
        
        self._conn.commit()

    def purge_deleted_relations(self, *, cutoff_time: float, limit: int = 1000) -> List[str]:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT hash
            FROM deleted_relations
            WHERE deleted_at IS NOT NULL AND deleted_at < ?
            ORDER BY deleted_at ASC
            LIMIT ?
            """,
            (float(cutoff_time), max(1, int(limit or 1000))),
        )
        hashes = [str(row[0] or "").strip() for row in cursor.fetchall() if str(row[0] or "").strip()]
        if not hashes:
            return []
        placeholders = ",".join(["?"] * len(hashes))
        cursor.execute(f"DELETE FROM deleted_relations WHERE hash IN ({placeholders})", tuple(hashes))
        self._conn.commit()
        return hashes
