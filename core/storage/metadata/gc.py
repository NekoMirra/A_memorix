"""软删除与垃圾回收。"""

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


class MetadataGcMixin:
    """软删除与垃圾回收。"""

    def get_entity_gc_candidates(self, isolated_hashes: List[str], retention_seconds: float) -> List[str]:
        """
        获取实体 GC 候选列表 (Soft Delete Candidates)
        条件:
        1. 在 isolated_hashes 列表中 (由 GraphStore 提供；通常是实体名称)
        2. is_deleted = 0 (未被标记)
        3. created_at < now - retention (过了新手保护期)
        4. 不被任何 active paragraph 引用 (paragraph_entities check)
    
        Args:
            isolated_hashes: 孤儿实体名称列表（兼容传入 hash）
            retention_seconds: 保留时间 (秒)
        """
        if not isolated_hashes:
            return []

        # GraphStore.get_isolated_nodes 返回节点名，这里做 canonicalize -> entity hash 映射。
        # 同时兼容历史调用直接传 hash。
        normalized_hashes: List[str] = []
        for item in isolated_hashes:
            if not item:
                continue
            v = str(item).strip()
            if len(v) == 64 and all(c in "0123456789abcdefABCDEF" for c in v):
                normalized_hashes.append(v.lower())
            else:
                canon = self._canonicalize_name(v)
                if canon:
                    normalized_hashes.append(compute_hash(canon))

        normalized_hashes = list(dict.fromkeys(normalized_hashes))
        if not normalized_hashes:
            return []
        
        now = datetime.now().timestamp()
        cutoff = now - retention_seconds
    
        candidates = []
        batch_size = 900
    
        # 分批处理 IN 查询
        for i in range(0, len(normalized_hashes), batch_size):
            batch = normalized_hashes[i:i+batch_size]
            placeholders = ",".join(["?"] * len(batch))
        
            # 使用 NOT EXISTS 子查询检查引用
            # 注意: paragraph_entities 中引用的 paragraph 如果被软删了，是否算引用？
            # 这里的语义: 只要有 rows 存在于 paragraph_entities 且该 row 对应的 paragraph 没被彻底物理删除，就算引用。
            # 更严格: ... OR (EXISTS ... AND entity_hash=... AND is_deleted=0)
            # 但 paragraph_entities 表没有 is_deleted 字段(它是关联表). 我们检查关联是否存在。
            # 如果 paragraph 本身 soft deleted, 它的引用应该失效吗？
            # 策略: 只有当 paragraph 也是 active 时，引用才有效。
            # JOIN paragraphs p ON pe.paragraph_hash = p.hash WHERE p.is_deleted = 0
        
            query = f"""
                SELECT e.hash FROM entities e
                WHERE e.hash IN ({placeholders})
                AND e.is_deleted = 0
                AND (e.created_at IS NULL OR e.created_at < ?)
                AND NOT EXISTS (
                    SELECT 1 FROM paragraph_entities pe
                    JOIN paragraphs p ON pe.paragraph_hash = p.hash
                    WHERE pe.entity_hash = e.hash
                    AND p.is_deleted = 0
                )
            """
        
            cursor = self._conn.cursor()
            cursor.execute(query, [*batch, cutoff])
            candidates.extend([row[0] for row in cursor.fetchall()])
        
        return candidates

    def get_paragraph_gc_candidates(self, retention_seconds: float) -> List[str]:
        """
        获取段落 GC 候选列表
        条件:
        1. is_deleted = 0
        2. created_at < cutoff
        3. 没有 Relations (paragraph_relations empty)
        4. 没有 Entities 引用 (paragraph_entities empty) 
           OR 引用的 Entities 全是软删状态? (太复杂，简单点: 无引用)
       
        Refined Strategy: 
        段落孤儿判定 = 
          (Left Join paragraph_relations -> NULL) AND 
          (Left Join paragraph_entities -> NULL)
        """
        now = datetime.now().timestamp()
        cutoff = now - retention_seconds
    
        query = """
            SELECT p.hash FROM paragraphs p
            LEFT JOIN paragraph_relations pr ON p.hash = pr.paragraph_hash
            LEFT JOIN paragraph_entities pe ON p.hash = pe.paragraph_hash
            WHERE p.is_deleted = 0
            AND (p.created_at IS NULL OR p.created_at < ?)
            AND pr.relation_hash IS NULL
            AND pe.entity_hash IS NULL
        """
    
        cursor = self._conn.cursor()
        cursor.execute(query, (cutoff,))
        return [row[0] for row in cursor.fetchall()]

    def mark_as_deleted(self, hashes: List[str], type_: str) -> int:
        """
        标记为软删除 (Mark Phase)
    
        Args:
            hashes: Hash 列表
            type_: 'entity' | 'paragraph'
        """
        if not hashes:
            return 0
        
        table = "entities" if type_ == "entity" else "paragraphs"
        now = datetime.now().timestamp()
        touched_sources: List[str] = []
        if type_ == "paragraph":
            touched_sources = self._get_sources_for_paragraph_hashes(hashes, include_deleted=True)
    
        count = 0
        batch_size = 900
        for i in range(0, len(hashes), batch_size):
            batch = hashes[i:i+batch_size]
            placeholders = ",".join(["?"] * len(batch))
        
            # 幂等更新: 只更那些 is_deleted=0 的
            cursor = self._conn.cursor()
            cursor.execute(f"""
                UPDATE {table}
                SET is_deleted = 1, deleted_at = ?
                WHERE is_deleted = 0 AND hash IN ({placeholders})
            """, [now] + batch)
            count += cursor.rowcount
        
        self._conn.commit()
        if type_ == "paragraph" and count > 0:
            self._enqueue_episode_source_rebuilds(
                touched_sources,
                reason="paragraph_soft_deleted",
            )
        if count > 0:
            logger.info(f"软删除标记 ({table}): {count} 项")
        return count

    def sweep_deleted_items(self, type_: str, grace_period_seconds: float) -> List[Tuple[str, str]]:
        """
        扫描可物理清理的项目 (Sweep Phase - Selection)
    
        Args:
            type_: 'entity' | 'paragraph'
            grace_period_seconds: 宽限期
        
        Returns:
            List[(hash, name)]: 待删除项列表 (paragraph name为空)
        """
        table = "entities" if type_ == "entity" else "paragraphs"
        now = datetime.now().timestamp()
        cutoff = now - grace_period_seconds
    
        cols = "hash, name" if type_ == "entity" else "hash, '' as name"
    
        cursor = self._conn.cursor()
        cursor.execute(f"""
            SELECT {cols} FROM {table}
            WHERE is_deleted = 1
            AND deleted_at < ?
        """, (cutoff,))
    
        return [(row[0], row[1]) for row in cursor.fetchall()]

    def physically_delete_entities(self, hashes: List[str]) -> int:
        """物理删除实体 (批量)"""
        if not hashes: return 0
    
        count = 0
        batch_size = 900
        for i in range(0, len(hashes), batch_size):
            batch = hashes[i:i+batch_size]
            placeholders = ",".join(["?"] * len(batch))
        
            cursor = self._conn.cursor()
            cursor.execute(f"DELETE FROM entities WHERE hash IN ({placeholders})", batch)
            count += cursor.rowcount
        
        self._conn.commit()
        return count

    def physically_delete_paragraphs(self, hashes: List[str]) -> int:
        """物理删除段落 (批量)"""
        if not hashes: return 0
        touched_sources = self._get_sources_for_paragraph_hashes(hashes, include_deleted=True)
    
        count = 0
        batch_size = 900
        for i in range(0, len(hashes), batch_size):
            batch = hashes[i:i+batch_size]
            placeholders = ",".join(["?"] * len(batch))
        
            cursor = self._conn.cursor()
            cursor.execute(f"DELETE FROM paragraphs WHERE hash IN ({placeholders})", batch)
            count += cursor.rowcount
        
        self._conn.commit()
        if count > 0:
            self._enqueue_episode_source_rebuilds(
                touched_sources,
                reason="paragraph_physically_deleted",
            )
        return count

    def revive_if_deleted(self, entity_hashes: List[str] = None, paragraph_hashes: List[str] = None) -> int:
        """
        复活已软删的项目 (Auto Revival)
        当数据被再次访问、引用或导入时调用。
        """
        count = 0
    
        if entity_hashes:
            batch_size = 900
            for i in range(0, len(entity_hashes), batch_size):
                batch = entity_hashes[i:i+batch_size]
                placeholders = ",".join(["?"] * len(batch))
            
                cursor = self._conn.cursor()
                cursor.execute(f"""
                    UPDATE entities
                    SET is_deleted = 0, deleted_at = NULL
                    WHERE is_deleted = 1 AND hash IN ({placeholders})
                """, batch)
                count += cursor.rowcount
            
        if paragraph_hashes:
            touched_sources = self._get_sources_for_paragraph_hashes(paragraph_hashes, include_deleted=True)
            batch_size = 900
            for i in range(0, len(paragraph_hashes), batch_size):
                batch = paragraph_hashes[i:i+batch_size]
                placeholders = ",".join(["?"] * len(batch))
            
                cursor = self._conn.cursor()
                cursor.execute(f"""
                    UPDATE paragraphs
                    SET is_deleted = 0, deleted_at = NULL
                    WHERE is_deleted = 1 AND hash IN ({placeholders})
                """, batch)
                count += cursor.rowcount
        else:
            touched_sources = []
    
        if count > 0:
            self._conn.commit()
            if touched_sources:
                self._enqueue_episode_source_rebuilds(
                    touched_sources,
                    reason="paragraph_revived",
                )
            logger.info(f"自动复活: {count} 项 (Soft Delete Revived)")
        
        return count

    def revive_entities_by_names(self, names: List[str]) -> int:
        """
        根据名称复活实体 (Convenience wrapper)
        """
        if not names: return 0
    
        # 使用内部方法计算哈希
        hashes = [compute_hash(self._canonicalize_name(n)) for n in names]
        return self.revive_if_deleted(entity_hashes=hashes)

    def get_entity_status_batch(self, hashes: List[str]) -> Dict[str, Dict[str, Any]]:
        """批量获取实体状态 (WebUI用)"""
        if not hashes: return {}
    
        result = {}
        batch_size = 900
        for i in range(0, len(hashes), batch_size):
            batch = hashes[i:i+batch_size]
            placeholders = ",".join(["?"] * len(batch))
        
            cursor = self._conn.cursor()
            cursor.execute(f"""
                SELECT hash, is_deleted, deleted_at 
                FROM entities 
                WHERE hash IN ({placeholders})
            """, batch)
        
            for row in cursor.fetchall():
                result[row[0]] = {
                    "is_deleted": bool(row[1]),
                    "deleted_at": row[2]
                }
        return result
