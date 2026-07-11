"""实体元数据与段落-实体关联。"""

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


class MetadataEntitiesMixin:
    """实体元数据与段落-实体关联。"""

    def _canonicalize_name(self, name: str) -> str:
        """
        规范化名称 (统一小写并去除首尾空格)
    
        Args:
            name: 原始名称
        
        Returns:
            规范化后的名称
        """
        if not name:
            return ""
        return name.strip().lower()

    def add_entity(
        self,
        name: str,
        vector_index: Optional[int] = None,
        source_paragraph: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        添加实体
    
        Args:
            name: 实体名称
            vector_index: 向量索引
            source_paragraph: 来源段落哈希 (如果提供，将建立关联)
            metadata: 额外元数据
        
        Returns:
            实体哈希值
        """
        # 1. 规范化名称
        name_normalized = self._canonicalize_name(name)
        if not name_normalized:
            raise ValueError("Entity name cannot be empty")
        
        hash_value = compute_hash(name_normalized)
        now = datetime.now().timestamp()

        cursor = self._conn.cursor()
    
        # 2. 插入实体 (INSERT OR IGNORE)
        # 注意：这里我们保留原有的 name 字段存储，可以是 display name，
        # 但 hash 必须由 canonical name 生成。
        # 如果实体已存在，我们其实不一定要更新 name (保留第一次的 display name 往往更好)
        # 或者我们也可以选择不作为唯一键冲突，而是逻辑判断。
        # 考虑到 entities.hash 是主键，entities.name 是 UNIQUE。
        # 如果 name 大小写不同但 hash 相同 (冲突)，或者 name 不同但 canonical name 相同?
        # 由于 hash 是由 canonical name 算出来的，所以 hash 相同意味着 canonical name 相同。
        # 如果 db 中已存在的 name 是 "Apple"，新来的 name 是 "apple"，它们 canonical name 都是 "apple"，hash 一样。
        # 此时 INSERT OR IGNORE 会忽略。
    
        try:
            cursor.execute("""
                INSERT INTO entities
                (hash, name, vector_index, appearance_count, created_at, metadata)
                VALUES (?, ?, ?, 1, ?, ?)
            """, (
                hash_value,
                name,
                vector_index,
                now,
                pickle.dumps(metadata or {}),
            ))
        
            logger.debug(f"添加实体: {name} ({hash_value[:8]})")
            self._conn.commit()
        
            # 3. 建立来源关联
            if source_paragraph:
                self.link_paragraph_entity(source_paragraph, hash_value)
            
            return hash_value
        
        except sqlite3.IntegrityError:
            # 实体已存在
            # 1. 尝试复活 (自动复活)
            self.revive_if_deleted(entity_hashes=[hash_value])
        
            # 2. 更新计数
            cursor.execute("""
                UPDATE entities
                SET appearance_count = appearance_count + 1
                WHERE hash = ?
            """, (hash_value,))
            self._conn.commit()
        
            logger.debug(f"实体已存在(复活/计数+1): {name}")
        
            # 3. 建立来源关联
            if source_paragraph:
                self.link_paragraph_entity(source_paragraph, hash_value)
            
            return hash_value

    def get_entity(self, hash_value: str) -> Optional[Dict[str, Any]]:
        """
        获取实体

        Args:
            hash_value: 实体哈希

        Returns:
            实体信息字典，不存在则返回None
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT * FROM entities WHERE hash = ?
        """, (hash_value,))
        row = cursor.fetchone()

        if row:
            return self._row_to_dict(row, "entity")
        return None

    def delete_entity(self, hash_or_name: str) -> bool:
        """
        删除实体（级联删除相关关联）
        支持通过哈希值或名称删除
    
        注意：会同时删除所有引用该实体（作为主语或宾语）的关系
        """
        cursor = self._conn.cursor()
    
        # 1. 解析实体信息 (获取 Name 和 Hash)
        entity_name = None
        entity_hash = None
    
        # 尝试作为 Hash 查询
        cursor.execute("SELECT name, hash FROM entities WHERE hash = ?", (hash_or_name,))
        row = cursor.fetchone()
        if row:
            entity_name = row[0]
            entity_hash = row[1]
        else:
            # 尝试作为 Name 查询 (原始匹配)
            cursor.execute("SELECT name, hash FROM entities WHERE name = ?", (hash_or_name,))
            row = cursor.fetchone()
            if row:
                entity_name = row[0]
                entity_hash = row[1]
            else:
                # 最后的最后：尝试规范化名称 (Canonical) 查询，解决大小写或 WebUI 手动输入导致的不匹配
                name_canon = self._canonicalize_name(hash_or_name)
                canon_hash = compute_hash(name_canon)
                cursor.execute("SELECT name, hash FROM entities WHERE hash = ?", (canon_hash,))
                row = cursor.fetchone()
                if row:
                    entity_name = row[0]
                    entity_hash = row[1]
            
        if not entity_name or not entity_hash:
            logger.debug(f"删除实体请求跳过：未在元数据记录中找到 {hash_or_name}")
            return False

        logger.info(f"开始删除实体: {entity_name} (Hash: {entity_hash[:8]}...)")

        try:
            # 2. 查找相关关系 (Subject 或 Object 为该实体)
            cursor.execute("""
                SELECT hash FROM relations 
                WHERE subject = ? OR object = ?
            """, (entity_name, entity_name))
        
            relation_hashes = [r[0] for r in cursor.fetchall()]
        
            if relation_hashes:
                logger.info(f"发现 {len(relation_hashes)} 个相关关系，准备级联删除")
            
                # 3. 删除这些关系与段落的关联
                # SQLite 不支持直接 DELETE ... WHERE ... IN (...) 的列表参数，需要拼接占位符
                placeholders = ','.join(['?'] * len(relation_hashes))
            
                cursor.execute(f"""
                    DELETE FROM paragraph_relations 
                    WHERE relation_hash IN ({placeholders})
                """, relation_hashes)
            
                # 4. 删除关系本体
                cursor.execute(f"""
                    DELETE FROM relations 
                    WHERE hash IN ({placeholders})
                """, relation_hashes)
            
                logger.info("相关关系已级联删除")
        
            # 5. 删除实体与段落的关联
            cursor.execute("DELETE FROM paragraph_entities WHERE entity_hash = ?", (entity_hash,))
        
            # 6. 删除实体本体
            cursor.execute("DELETE FROM entities WHERE hash = ?", (entity_hash,))
        
            self._conn.commit()
            logger.info("实体删除完成")
            return True
        
        except Exception as e:
            logger.error(f"删除实体时发生错误: {e}")
            self._conn.rollback()
            return False

    def get_paragraph_entities(self, paragraph_hash: str) -> List[Dict[str, Any]]:
        """
        获取段落的所有实体

        Args:
            paragraph_hash: 段落哈希

        Returns:
            实体列表
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT e.*, pe.mention_count
            FROM entities e
            JOIN paragraph_entities pe ON e.hash = pe.entity_hash
            WHERE pe.paragraph_hash = ?
        """, (paragraph_hash,))

        return [self._row_to_dict(row, "entity") for row in cursor.fetchall()]

    def get_paragraphs_by_entity(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        获取包含指定实体的所有段落 (自动处理规范化)
    
        Args:
            entity_name: 实体名称 (支持任意大小写)
        
        Returns:
            段落列表
        """
        # 1. 计算规范化 Hash
        name_canon = self._canonicalize_name(entity_name)
        if not name_canon:
            return []
        
        entity_hash = compute_hash(name_canon)
    
        cursor = self._conn.cursor()
        # 2. 直接使用 Hash 查询中间表，完全避开 Name 匹配
        cursor.execute("""
            SELECT p.*
            FROM paragraphs p
            JOIN paragraph_entities pe ON p.hash = pe.paragraph_hash
            WHERE pe.entity_hash = ?
        """, (entity_hash,))

        return [self._row_to_dict(row, "paragraph") for row in cursor.fetchall()]

    def link_paragraph_entity(
        self,
        paragraph_hash: str,
        entity_hash: str,
        mention_count: int = 1,
    ) -> bool:
        """
        关联段落和实体 (幂等)
        """
        cursor = self._conn.cursor()
        try:
            # 首先尝试插入
            cursor.execute("""
                INSERT OR IGNORE INTO paragraph_entities
                (paragraph_hash, entity_hash, mention_count)
                VALUES (?, ?, ?)
            """, (paragraph_hash, entity_hash, mention_count))
        
            if cursor.rowcount == 0:
                # 如果已存在 (IGNORE生效)，则更新计数
                cursor.execute("""
                    UPDATE paragraph_entities
                    SET mention_count = mention_count + ?
                    WHERE paragraph_hash = ? AND entity_hash = ?
                """, (mention_count, paragraph_hash, entity_hash))
        
            self._conn.commit()
            self._enqueue_episode_source_rebuilds(
                self._get_sources_for_paragraph_hashes([paragraph_hash], include_deleted=True),
                reason="paragraph_entity_linked",
            )
            return True
        except sqlite3.IntegrityError:
            return False

    def count_entities(self) -> int:
        """
        获取实体数量

        Returns:
            实体数量
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM entities")
        return cursor.fetchone()[0]

    def is_entity_still_referenced(self, entity_hash: str, entity_name: str = "") -> bool:
        """
        判断实体是否仍被引用：
        1) 被 paragraph_entities 引用
        2) 在 relations.subject/object 中出现
        """
        token_hash = str(entity_hash or "").strip()
        if token_hash:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT 1 FROM paragraph_entities WHERE entity_hash = ? LIMIT 1",
                (token_hash,),
            )
            if cursor.fetchone() is not None:
                return True

        canon_name = self._canonicalize_name(entity_name)
        if canon_name:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT 1
                FROM relations
                WHERE LOWER(TRIM(subject)) = ? OR LOWER(TRIM(object)) = ?
                LIMIT 1
                """,
                (canon_name, canon_name),
            )
            if cursor.fetchone() is not None:
                return True
        return False

    def restore_entity_by_hash(self, entity_hash: str) -> bool:
        """恢复软删除实体。"""
        cursor = self._conn.cursor()
        cursor.execute(
            "UPDATE entities SET is_deleted=0, deleted_at=NULL WHERE hash=?",
            (str(entity_hash),),
        )
        changed = cursor.rowcount > 0
        if changed:
            self._conn.commit()
        return changed

    def get_deleted_entities(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取已软删除的实体 (回收站用)"""
        if not self.has_table("entities"): return []
    
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT hash, name, deleted_at 
            FROM entities 
            WHERE is_deleted = 1 
            ORDER BY deleted_at DESC 
            LIMIT ?
        """, (limit,))
    
        items = []
        for row in cursor.fetchall():
            items.append({
                "hash": row[0],
                "name": row[1],
                "type": "entity", # 标记为实体
                "deleted_at": row[2]
            })
        return items
