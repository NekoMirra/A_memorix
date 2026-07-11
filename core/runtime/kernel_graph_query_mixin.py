from __future__ import annotations

from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Sequence
from src.common.logger import get_logger
from ..utils.hash import compute_hash, normalize_text

logger = get_logger("A_Memorix.SDKMemoryKernel")


class KernelGraphQueryMixin:
    def _serialize_graph(self, *, limit: int = 200) -> Dict[str, Any]:
        assert self.graph_store is not None
        assert self.metadata_store is not None
        nodes = self.graph_store.get_nodes()
        if limit > 0:
            nodes = nodes[:limit]
        node_set = set(nodes)
        node_payload = []
        for name in nodes:
            attrs = self.graph_store.get_node_attributes(name) or {}
            node_payload.append({"id": name, "name": name, "attributes": attrs})

        edge_payload = []
        for source, target, relation_hashes in self.graph_store.iter_edge_hash_entries():
            if source not in node_set or target not in node_set:
                continue
            relation_hash_tokens = sorted(str(item) for item in relation_hashes if str(item).strip())
            relation_rows = self._query_relation_rows_by_hashes(relation_hash_tokens)
            predicates = self._dedupe_strings(row.get("predicate", "") for row in relation_rows)
            evidence_hashes = self._query_distinct_paragraph_hashes_for_relations(relation_hash_tokens)
            edge_payload.append(
                {
                    "source": source,
                    "target": target,
                    "weight": float(self.graph_store.get_edge_weight(source, target)),
                    "relation_hashes": relation_hash_tokens,
                    "predicates": predicates,
                    "relation_count": len(relation_hash_tokens),
                    "evidence_count": len(evidence_hashes),
                    "label": self._build_graph_edge_label(predicates),
                }
            )
        return {
            "nodes": node_payload,
            "edges": edge_payload,
            "total_nodes": int(self.graph_store.num_nodes),
            "total_edges": int(self.graph_store.num_edges),
        }

    def _dedupe_strings(values: Iterable[Any]) -> List[str]:
        deduped: List[str] = []
        for value in values:
            token = str(value or "").strip()
            if token and token not in deduped:
                deduped.append(token)
        return deduped

    def _build_graph_edge_label(predicates: Sequence[str]) -> str:
        labels = [str(item or "").strip() for item in predicates if str(item or "").strip()]
        if not labels:
            return ""
        if len(labels) == 1:
            return labels[0]
        return f"{labels[0]} +{len(labels) - 1}"

    def _trim_text(value: str, limit: int = 220) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return f"{text[:limit]}..."

    def _format_relation_text(subject: Any, predicate: Any, obj: Any) -> str:
        return " ".join(
            [
                str(subject or "").strip(),
                str(predicate or "").strip(),
                str(obj or "").strip(),
            ]
        ).strip()

    def _query_relation_rows_by_hashes(
        self,
        relation_hashes: Sequence[str],
        *,
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        assert self.metadata_store is not None
        hashes = [str(item or "").strip() for item in relation_hashes if str(item or "").strip()]
        if not hashes:
            return []
        placeholders = ",".join(["?"] * len(hashes))
        inactive_clause = "" if include_inactive else "AND (is_inactive IS NULL OR is_inactive = 0)"
        rows = self.metadata_store.query(
            f"""
            SELECT hash, subject, predicate, object, confidence, created_at, source_paragraph
            FROM relations
            WHERE hash IN ({placeholders})
              {inactive_clause}
            """,
            tuple(hashes),
        )
        order = {hash_value: index for index, hash_value in enumerate(hashes)}
        rows.sort(key=lambda row: order.get(str(row.get("hash", "") or ""), len(order)))
        return rows

    def _query_distinct_paragraph_hashes_for_relations(
        self,
        relation_hashes: Sequence[str],
        *,
        limit: Optional[int] = None,
    ) -> List[str]:
        assert self.metadata_store is not None
        hashes = [str(item or "").strip() for item in relation_hashes if str(item or "").strip()]
        if not hashes:
            return []
        placeholders = ",".join(["?"] * len(hashes))
        sql = f"""
            SELECT DISTINCT p.hash, p.updated_at, p.created_at
            FROM paragraphs p
            JOIN paragraph_relations pr ON p.hash = pr.paragraph_hash
            WHERE pr.relation_hash IN ({placeholders})
              AND (p.is_deleted IS NULL OR p.is_deleted = 0)
            ORDER BY p.updated_at DESC, p.created_at DESC, p.hash ASC
        """
        params: List[Any] = list(hashes)
        if limit is not None and limit > 0:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self.metadata_store.query(sql, tuple(params))
        return [str(row.get("hash", "") or "").strip() for row in rows if str(row.get("hash", "") or "").strip()]

    def _load_paragraph_rows(self, paragraph_hashes: Sequence[str]) -> List[Dict[str, Any]]:
        assert self.metadata_store is not None
        hashes = [str(item or "").strip() for item in paragraph_hashes if str(item or "").strip()]
        if not hashes:
            return []
        rows: List[Dict[str, Any]] = []
        for hash_value in hashes:
            row = self.metadata_store.get_paragraph(hash_value)
            if row is None:
                continue
            if bool(row.get("is_deleted", 0)):
                continue
            rows.append(row)
        return rows

    def _resolve_graph_node_name(self, node_id: str) -> str:
        assert self.metadata_store is not None
        assert self.graph_store is not None
        token = str(node_id or "").strip()
        if not token:
            return ""
        graph_nodes = self.graph_store.get_nodes()
        for candidate in graph_nodes:
            if str(candidate or "").strip().lower() == token.lower():
                return str(candidate)
        entity_rows = self.metadata_store.query(
            """
            SELECT name
            FROM entities
            WHERE LOWER(TRIM(name)) = LOWER(TRIM(?))
               OR hash = ?
            ORDER BY appearance_count DESC, created_at ASC
            LIMIT 1
            """,
            (token, token),
        )
        if entity_rows:
            return str(entity_rows[0].get("name", "") or token)
        relation_rows = self.metadata_store.query(
            """
            SELECT subject, object
            FROM relations
            WHERE (LOWER(TRIM(subject)) = LOWER(TRIM(?)) OR LOWER(TRIM(object)) = LOWER(TRIM(?)))
              AND (is_inactive IS NULL OR is_inactive = 0)
            LIMIT 1
            """,
            (token, token),
        )
        if relation_rows:
            subject = str(relation_rows[0].get("subject", "") or "").strip()
            obj = str(relation_rows[0].get("object", "") or "").strip()
            if subject.lower() == token.lower():
                return subject
            if obj.lower() == token.lower():
                return obj
        return token

    def _get_related_relation_rows_for_entity(self, entity_name: str, *, limit: int) -> List[Dict[str, Any]]:
        assert self.metadata_store is not None
        rows = self.metadata_store.query(
            """
            SELECT hash, subject, predicate, object, confidence, created_at, source_paragraph
            FROM relations
            WHERE (LOWER(TRIM(subject)) = LOWER(TRIM(?)) OR LOWER(TRIM(object)) = LOWER(TRIM(?)))
              AND (is_inactive IS NULL OR is_inactive = 0)
            ORDER BY confidence DESC, created_at DESC
            LIMIT ?
            """,
            (entity_name, entity_name, limit),
        )
        return rows

    def _build_relation_summary(self, row: Dict[str, Any], paragraph_hashes: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        relation_hash = str(row.get("hash", "") or "").strip()
        hashes = [str(item or "").strip() for item in (paragraph_hashes or []) if str(item or "").strip()]
        if not hashes and relation_hash:
            hashes = self._query_distinct_paragraph_hashes_for_relations([relation_hash])
        return {
            "hash": relation_hash,
            "subject": str(row.get("subject", "") or "").strip(),
            "predicate": str(row.get("predicate", "") or "").strip(),
            "object": str(row.get("object", "") or "").strip(),
            "text": self._format_relation_text(row.get("subject"), row.get("predicate"), row.get("object")),
            "confidence": float(row.get("confidence", 0.0) or 0.0),
            "paragraph_count": len(hashes),
            "paragraph_hashes": hashes,
            "source_paragraph": str(row.get("source_paragraph", "") or "").strip(),
        }

    def _build_paragraph_summary(self, row: Dict[str, Any]) -> Dict[str, Any]:
        assert self.metadata_store is not None
        paragraph_hash = str(row.get("hash", "") or "").strip()
        entities = self.metadata_store.get_paragraph_entities(paragraph_hash)
        relations = self.metadata_store.get_paragraph_relations(paragraph_hash)
        stale_marks_map, stale_status_map = self._load_paragraph_stale_marks([paragraph_hash])
        stale_marks = [
            {
                **mark,
                "relation_inactive": self._relation_status_is_inactive(
                    stale_status_map.get(str(mark.get("relation_hash", "") or "").strip())
                ),
            }
            for mark in stale_marks_map.get(paragraph_hash, [])
        ]
        return {
            "hash": paragraph_hash,
            "content": str(row.get("content", "") or ""),
            "preview": self._trim_text(str(row.get("content", "") or "")),
            "source": str(row.get("source", "") or ""),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
            "entity_count": len(entities),
            "relation_count": len(relations),
            "entities": self._dedupe_strings(entity.get("name", "") for entity in entities),
            "relations": [
                self._format_relation_text(
                    relation.get("subject", ""),
                    relation.get("predicate", ""),
                    relation.get("object", ""),
                )
                for relation in relations
            ],
            "is_stale": bool(stale_marks),
            "stale_relation_marks": stale_marks,
        }

    def _evidence_entity_node_id(name: str) -> str:
        return f"entity:{name}"

    def _evidence_relation_node_id(hash_value: str) -> str:
        return f"relation:{hash_value}"

    def _evidence_paragraph_node_id(hash_value: str) -> str:
        return f"paragraph:{hash_value}"
