"""段落/关系 FTS 与 ngram 检索相关方法。"""

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


class MetadataFtsMixin:
    """段落/关系 FTS 与 ngram 检索相关方法。"""

    def ensure_fts_schema(self, conn: Optional[sqlite3.Connection] = None) -> bool:
        """
        确保 FTS5 schema 存在（幂等）。

        采用 external-content 方式，不在 FTS 表重复存储正文。
        """
        c = self._resolve_conn(conn)
        cur = c.cursor()
        try:
            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS paragraphs_fts
                USING fts5(
                    content,
                    content='paragraphs',
                    content_rowid='rowid',
                    tokenize='unicode61'
                )
            """)

            # insert trigger
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS paragraphs_ai
                AFTER INSERT ON paragraphs
                BEGIN
                    INSERT INTO paragraphs_fts(rowid, content)
                    VALUES (new.rowid, new.content);
                END
            """)

            # delete trigger
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS paragraphs_ad
                AFTER DELETE ON paragraphs
                BEGIN
                    INSERT INTO paragraphs_fts(paragraphs_fts, rowid, content)
                    VALUES ('delete', old.rowid, old.content);
                END
            """)

            # update trigger
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS paragraphs_au
                AFTER UPDATE OF content ON paragraphs
                BEGIN
                    INSERT INTO paragraphs_fts(paragraphs_fts, rowid, content)
                    VALUES ('delete', old.rowid, old.content);
                    INSERT INTO paragraphs_fts(rowid, content)
                    VALUES (new.rowid, new.content);
                END
            """)
            c.commit()
            return True
        except sqlite3.OperationalError as e:
            logger.warning(f"FTS5 schema 创建失败（可能不支持 FTS5）: {e}")
            c.rollback()
            return False

    def ensure_fts_backfilled(self, conn: Optional[sqlite3.Connection] = None) -> bool:
        """
        确保 FTS 索引已回填。

        当历史数据存在但 FTS 表为空/不一致时执行 rebuild。
        """
        c = self._resolve_conn(conn)
        cur = c.cursor()
        try:
            cur.execute("SELECT COUNT(1) AS n FROM paragraphs")
            para_count = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(1) AS n FROM paragraphs_fts")
            fts_count = int(cur.fetchone()[0])

            if para_count > 0 and fts_count != para_count:
                cur.execute("INSERT INTO paragraphs_fts(paragraphs_fts) VALUES ('rebuild')")
                c.commit()
                logger.info(f"FTS 回填完成: paragraphs={para_count}, fts={para_count}")
            return True
        except sqlite3.OperationalError as e:
            logger.warning(f"FTS 回填失败: {e}")
            c.rollback()
            return False

    def ensure_relations_fts_schema(self, conn: Optional[sqlite3.Connection] = None) -> bool:
        """
        确保关系 FTS5 schema 存在（幂等）。

        注意：relations 表没有 content 列，因此使用独立 FTS 表并通过触发器同步。
        """
        c = self._resolve_conn(conn)
        cur = c.cursor()
        try:
            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS relations_fts
                USING fts5(
                    relation_hash UNINDEXED,
                    content,
                    tokenize='unicode61'
                )
            """)

            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS relations_ai
                AFTER INSERT ON relations
                BEGIN
                    INSERT INTO relations_fts(relation_hash, content)
                    VALUES (
                        new.hash,
                        COALESCE(new.subject, '') || ' ' || COALESCE(new.predicate, '') || ' ' || COALESCE(new.object, '')
                    );
                END
            """)

            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS relations_ad
                AFTER DELETE ON relations
                BEGIN
                    DELETE FROM relations_fts WHERE relation_hash = old.hash;
                END
            """)

            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS relations_au
                AFTER UPDATE OF subject, predicate, object ON relations
                BEGIN
                    DELETE FROM relations_fts WHERE relation_hash = new.hash;
                    INSERT INTO relations_fts(relation_hash, content)
                    VALUES (
                        new.hash,
                        COALESCE(new.subject, '') || ' ' || COALESCE(new.predicate, '') || ' ' || COALESCE(new.object, '')
                    );
                END
            """)
            c.commit()
            return True
        except sqlite3.OperationalError as e:
            logger.warning(f"relations FTS5 schema 创建失败（可能不支持 FTS5）: {e}")
            c.rollback()
            return False

    def ensure_relations_fts_backfilled(self, conn: Optional[sqlite3.Connection] = None) -> bool:
        """确保关系 FTS 索引已回填。"""
        c = self._resolve_conn(conn)
        cur = c.cursor()
        try:
            cur.execute("SELECT COUNT(1) AS n FROM relations")
            rel_count = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(1) AS n FROM relations_fts")
            fts_count = int(cur.fetchone()[0])

            if rel_count != fts_count:
                cur.execute("DELETE FROM relations_fts")
                cur.execute("""
                    INSERT INTO relations_fts(relation_hash, content)
                    SELECT
                        r.hash,
                        COALESCE(r.subject, '') || ' ' || COALESCE(r.predicate, '') || ' ' || COALESCE(r.object, '')
                    FROM relations r
                """)
                c.commit()
                logger.info(f"relations FTS 回填完成: relations={rel_count}, fts={rel_count}")
            return True
        except sqlite3.OperationalError as e:
            logger.warning(f"relations FTS 回填失败: {e}")
            c.rollback()
            return False

    def ensure_paragraph_ngram_schema(self, conn: Optional[sqlite3.Connection] = None) -> bool:
        """确保段落 ngram 倒排表存在。"""
        c = self._resolve_conn(conn)
        cur = c.cursor()
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS paragraph_ngrams (
                    term TEXT NOT NULL,
                    paragraph_hash TEXT NOT NULL,
                    PRIMARY KEY (term, paragraph_hash),
                    FOREIGN KEY (paragraph_hash) REFERENCES paragraphs(hash) ON DELETE CASCADE
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_paragraph_ngrams_hash
                ON paragraph_ngrams(paragraph_hash)
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS paragraph_ngram_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            c.commit()
            return True
        except sqlite3.OperationalError as e:
            logger.warning(f"paragraph ngram schema 创建失败: {e}")
            c.rollback()
            return False

    @staticmethod
    def _char_ngrams(text: str, n: int) -> List[str]:
        compact = "".join(str(text or "").lower().split())
        if not compact:
            return []
        if len(compact) < n:
            return [compact]
        return [compact[i : i + n] for i in range(0, len(compact) - n + 1)]

    def ensure_paragraph_ngram_backfilled(
        self,
        n: int = 2,
        conn: Optional[sqlite3.Connection] = None,
    ) -> bool:
        """
        确保段落 ngram 倒排索引已回填。

        仅在 n 变化或文档数量变化时重建，避免每次加载都全量重建。
        """
        c = self._resolve_conn(conn)
        cur = c.cursor()
        n = max(1, int(n))
        try:
            cur.execute("SELECT value FROM paragraph_ngram_meta WHERE key='ngram_n'")
            row = cur.fetchone()
            current_n = int(row[0]) if row and row[0] is not None else None

            cur.execute("SELECT COUNT(1) FROM paragraphs WHERE is_deleted IS NULL OR is_deleted = 0")
            para_count = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(DISTINCT paragraph_hash) FROM paragraph_ngrams")
            indexed_docs = int(cur.fetchone()[0])

            need_rebuild = (current_n != n) or (para_count != indexed_docs)
            if not need_rebuild:
                return True

            cur.execute("DELETE FROM paragraph_ngrams")
            cur.execute("""
                SELECT hash, content
                FROM paragraphs
                WHERE is_deleted IS NULL OR is_deleted = 0
            """)
            rows = cur.fetchall()

            batch: List[Tuple[str, str]] = []
            batch_size = 2000
            for row in rows:
                p_hash = str(row["hash"])
                terms = list(dict.fromkeys(self._char_ngrams(str(row["content"] or ""), n)))
                for term in terms:
                    batch.append((term, p_hash))
                if len(batch) >= batch_size:
                    cur.executemany(
                        "INSERT OR IGNORE INTO paragraph_ngrams(term, paragraph_hash) VALUES (?, ?)",
                        batch,
                    )
                    batch.clear()
            if batch:
                cur.executemany(
                    "INSERT OR IGNORE INTO paragraph_ngrams(term, paragraph_hash) VALUES (?, ?)",
                    batch,
                )

            cur.execute("""
                INSERT INTO paragraph_ngram_meta(key, value) VALUES('ngram_n', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """, (str(n),))
            cur.execute("""
                INSERT INTO paragraph_ngram_meta(key, value) VALUES('paragraph_count', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """, (str(para_count),))
            c.commit()
            logger.info(f"paragraph ngram 回填完成: n={n}, paragraphs={para_count}")
            return True
        except Exception as e:
            logger.warning(f"paragraph ngram 回填失败: {e}")
            c.rollback()
            return False

    def fts_upsert_paragraph(
        self,
        paragraph_hash: str,
        conn: Optional[sqlite3.Connection] = None,
    ) -> bool:
        """
        将段落写入（或覆盖）到 FTS 索引。
        """
        c = self._resolve_conn(conn)
        cur = c.cursor()
        try:
            cur.execute(
                "SELECT rowid, content FROM paragraphs WHERE hash = ?",
                (paragraph_hash,),
            )
            row = cur.fetchone()
            if not row:
                return False
            rowid = int(row[0])
            content = str(row[1] or "")
            cur.execute(
                "INSERT OR REPLACE INTO paragraphs_fts(rowid, content) VALUES (?, ?)",
                (rowid, content),
            )
            c.commit()
            return True
        except sqlite3.OperationalError as e:
            logger.warning(f"FTS upsert 失败: {e}")
            c.rollback()
            return False

    def fts_delete_paragraph(
        self,
        paragraph_hash: str,
        conn: Optional[sqlite3.Connection] = None,
    ) -> bool:
        """
        从 FTS 索引删除段落。
        """
        c = self._resolve_conn(conn)
        cur = c.cursor()
        try:
            cur.execute(
                "SELECT rowid, content FROM paragraphs WHERE hash = ?",
                (paragraph_hash,),
            )
            row = cur.fetchone()
            if not row:
                return False
            rowid = int(row[0])
            content = str(row[1] or "")
            cur.execute(
                "INSERT INTO paragraphs_fts(paragraphs_fts, rowid, content) VALUES ('delete', ?, ?)",
                (rowid, content),
            )
            c.commit()
            return True
        except sqlite3.OperationalError as e:
            logger.warning(f"FTS delete 失败: {e}")
            c.rollback()
            return False

    def fts_search_bm25(
        self,
        match_query: str,
        limit: int = 20,
        max_doc_len: int = 2000,
        conn: Optional[sqlite3.Connection] = None,
    ) -> List[Dict[str, Any]]:
        """
        使用 FTS5 + bm25 执行全文检索。
        """
        if not match_query.strip():
            return []

        c = self._resolve_conn(conn)
        cur = c.cursor()
        try:
            cur.execute(
                """
                SELECT p.hash, p.content, bm25(paragraphs_fts) AS bm25_score
                FROM paragraphs_fts
                JOIN paragraphs p ON p.rowid = paragraphs_fts.rowid
                WHERE paragraphs_fts MATCH ?
                  AND (p.is_deleted IS NULL OR p.is_deleted = 0)
                ORDER BY bm25_score ASC
                LIMIT ?
                """,
                (match_query, max(1, int(limit))),
            )
            rows = cur.fetchall()
            results: List[Dict[str, Any]] = []
            for row in rows:
                content = str(row["content"] or "")
                if max_doc_len > 0:
                    content = content[:max_doc_len]
                results.append(
                    {
                        "hash": row["hash"],
                        "content": content,
                        "bm25_score": float(row["bm25_score"]),
                    }
                )
            return results
        except sqlite3.OperationalError as e:
            logger.warning(f"FTS 查询失败: {e}")
            return []

    def fts_search_relations_bm25(
        self,
        match_query: str,
        limit: int = 20,
        max_doc_len: int = 512,
        include_inactive: bool = True,
        conn: Optional[sqlite3.Connection] = None,
    ) -> List[Dict[str, Any]]:
        """使用 FTS5 + bm25 执行关系全文检索。"""
        if not match_query.strip():
            return []

        c = self._resolve_conn(conn)
        cur = c.cursor()
        active_clause = "" if include_inactive else " AND (r.is_inactive IS NULL OR r.is_inactive = 0)"
        try:
            cur.execute(
                f"""
                SELECT
                    r.hash,
                    r.subject,
                    r.predicate,
                    r.object,
                    bm25(relations_fts) AS bm25_score
                FROM relations_fts
                JOIN relations r ON r.hash = relations_fts.relation_hash
                WHERE relations_fts MATCH ?
                {active_clause}
                ORDER BY bm25_score ASC
                LIMIT ?
                """,
                (match_query, max(1, int(limit))),
            )
            rows = cur.fetchall()
            out: List[Dict[str, Any]] = []
            for row in rows:
                content = f"{row['subject']} {row['predicate']} {row['object']}"
                if max_doc_len > 0:
                    content = content[:max_doc_len]
                out.append(
                    {
                        "hash": row["hash"],
                        "subject": row["subject"],
                        "predicate": row["predicate"],
                        "object": row["object"],
                        "content": content,
                        "bm25_score": float(row["bm25_score"]),
                    }
                )
            return out
        except sqlite3.OperationalError as e:
            logger.warning(f"relations FTS 查询失败: {e}")
            return []

    def ngram_search_paragraphs(
        self,
        tokens: List[str],
        limit: int = 20,
        max_doc_len: int = 2000,
        conn: Optional[sqlite3.Connection] = None,
    ) -> List[Dict[str, Any]]:
        """按 ngram 倒排索引检索段落，避免 LIKE 全表扫描。"""
        uniq = [t for t in dict.fromkeys([str(x).strip().lower() for x in tokens]) if t]
        if not uniq:
            return []

        c = self._resolve_conn(conn)
        cur = c.cursor()
        placeholders = ",".join(["?"] * len(uniq))
        try:
            cur.execute(
                f"""
                SELECT
                    p.hash,
                    p.content,
                    COUNT(*) AS hit_terms
                FROM paragraph_ngrams ng
                JOIN paragraphs p ON p.hash = ng.paragraph_hash
                WHERE ng.term IN ({placeholders})
                  AND (p.is_deleted IS NULL OR p.is_deleted = 0)
                GROUP BY p.hash, p.content
                ORDER BY hit_terms DESC
                LIMIT ?
                """,
                tuple(uniq + [max(1, int(limit))]),
            )
            rows = cur.fetchall()
            out: List[Dict[str, Any]] = []
            token_count = max(1, len(uniq))
            for row in rows:
                hit_terms = int(row["hit_terms"])
                score = float(hit_terms / token_count)
                content = str(row["content"] or "")
                if max_doc_len > 0:
                    content = content[:max_doc_len]
                out.append(
                    {
                        "hash": row["hash"],
                        "content": content,
                        "bm25_score": -score,
                        "fallback_score": score,
                    }
                )
            return out
        except sqlite3.OperationalError as e:
            logger.warning(f"ngram 倒排查询失败: {e}")
            return []

    def fts_doc_count(self, conn: Optional[sqlite3.Connection] = None) -> int:
        """获取 FTS 文档数量。"""
        c = self._resolve_conn(conn)
        cur = c.cursor()
        try:
            cur.execute("SELECT COUNT(1) FROM paragraphs_fts")
            return int(cur.fetchone()[0])
        except sqlite3.OperationalError:
            return 0
