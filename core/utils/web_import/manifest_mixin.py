from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
import uuid

from src.common.logger import get_logger
from src.services import llm_service as llm_api

from ....paths import default_data_dir, repo_root, resolve_repo_path, scripts_root
from ...storage import (
    KnowledgeType,
    MetadataStore,
    parse_import_strategy,
    resolve_stored_knowledge_type,
    select_import_strategy,
)
from ...storage.knowledge_types import ImportStrategy
from ...storage.type_detection import looks_like_quote_text
from ...strategies.base import KnowledgeType as StrategyKnowledgeType, ProcessedChunk
from ...strategies.factual import FactualStrategy
from ...strategies.narrative import NarrativeStrategy
from ...strategies.quote import QuoteStrategy
from ..import_payloads import (
    ImportPayloadValidationError,
    is_probable_hash_token,
    normalize_entity_import_item,
    normalize_paragraph_import_item,
    normalize_relation_import_item,
)
from ..runtime_self_check import ensure_runtime_self_check
from ..time_parser import normalize_time_meta
from .helpers import (
    CHUNK_STATUS,
    FILE_STATUS,
    FILE_WARNING_KEEP_LIMIT,
    ImportChunkRecord,
    ImportFileRecord,
    ImportTaskRecord,
    TASK_STATUS,
    _clamp,
    _coerce_bool,
    _coerce_import_data_dict,
    _coerce_int,
    _coerce_list,
    _normalize_import_entity_list,
    _normalize_import_relation_list,
    _now,
    _parse_optional_positive_int,
    _safe_filename,
    _storage_type_from_strategy,
)

logger = get_logger("A_Memorix.WebImportManager")


class ImportManifestMixin:
    """Import manifest cache helpers."""
    def _load_manifest(self) -> Dict[str, Any]:
        if self._manifest_cache is not None:
            return self._manifest_cache
        path = self._manifest_path
        if not path.exists():
            self._manifest_cache = {}
            return self._manifest_cache
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._manifest_cache = payload
            else:
                self._manifest_cache = {}
        except Exception:
            self._manifest_cache = {}
        return self._manifest_cache

    def _save_manifest(self, payload: Dict[str, Any]) -> None:
        path = self._manifest_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._manifest_cache = payload

    def _clear_manifest(self) -> None:
        self._save_manifest({})

    def _normalize_manifest_path(self, raw_path: str) -> str:
        text = str(raw_path or "").strip()
        if not text:
            return ""
        return text.replace("\\", "/").strip().lower()

    def _match_manifest_item_for_source(self, source: str, item: Dict[str, Any]) -> bool:
        source_text = str(source or "").strip()
        if not source_text or ":" not in source_text:
            return False
        prefix, tail = source_text.split(":", 1)
        source_kind = prefix.strip().lower()
        source_value = tail.strip()
        if not source_value:
            return False

        item_kind = str(item.get("source_kind") or "").strip().lower()
        item_name = str(item.get("name") or "").strip()
        item_path_norm = self._normalize_manifest_path(item.get("source_path") or "")

        if source_kind in {"raw_scan", "lpmm_openie"}:
            source_path_norm = self._normalize_manifest_path(source_value)
            if source_path_norm and item_path_norm and source_path_norm == item_path_norm and item_kind == source_kind:
                return True

        if source_kind == "web_import":
            return item_kind in {"upload", "paste"} and item_name == source_value

        if source_kind == "lpmm_openie":
            source_name = Path(source_value).name
            return item_kind == "lpmm_openie" and item_name == source_name

        return False

    async def invalidate_manifest_for_sources(self, sources: List[str]) -> Dict[str, Any]:
        requested_sources: List[str] = []
        seen_sources = set()
        for raw in sources or []:
            source = str(raw or "").strip()
            if not source:
                continue
            key = source.lower()
            if key in seen_sources:
                continue
            seen_sources.add(key)
            requested_sources.append(source)

        result: Dict[str, Any] = {
            "requested_sources": requested_sources,
            "removed_count": 0,
            "removed_keys": [],
            "remaining_count": 0,
            "unmatched_sources": [],
            "warnings": [],
        }

        async with self._lock:
            manifest = self._load_manifest()
            if not isinstance(manifest, dict):
                manifest = {}

            valid_items: List[Tuple[str, Dict[str, Any]]] = []
            malformed_keys: List[str] = []
            for key, item in manifest.items():
                if isinstance(item, dict):
                    valid_items.append((str(key), item))
                else:
                    malformed_keys.append(str(key))

            keys_to_remove = set()
            for source in requested_sources:
                matched = False
                for key, item in valid_items:
                    if self._match_manifest_item_for_source(source, item):
                        keys_to_remove.add(key)
                        matched = True
                if not matched:
                    result["unmatched_sources"].append(source)

            if keys_to_remove:
                for key in keys_to_remove:
                    manifest.pop(key, None)
                self._save_manifest(manifest)

            result["removed_keys"] = sorted(keys_to_remove)
            result["removed_count"] = len(keys_to_remove)
            result["remaining_count"] = len(manifest)

            if malformed_keys:
                preview = ", ".join(malformed_keys[:5])
                extra = "" if len(malformed_keys) <= 5 else f" ... (+{len(malformed_keys) - 5})"
                result["warnings"].append(
                    f"manifest 条目结构异常，已跳过 {len(malformed_keys)} 项: {preview}{extra}"
                )

        return result

    def _manifest_key_for_file(self, file_record: ImportFileRecord, content_hash: str, dedupe_policy: str) -> str:
        if dedupe_policy == "content_hash":
            return f"hash:{content_hash}"
        if file_record.source_path:
            return f"path:{Path(file_record.source_path).as_posix().lower()}"
        return f"hash:{content_hash}"

    def _is_manifest_hit(
        self,
        file_record: ImportFileRecord,
        content_hash: str,
        dedupe_policy: str,
    ) -> bool:
        key = self._manifest_key_for_file(file_record, content_hash, dedupe_policy)
        manifest = self._load_manifest()
        item = manifest.get(key)
        if not isinstance(item, dict):
            return False
        return str(item.get("hash") or "") == content_hash and bool(item.get("imported"))

    def _record_manifest_import(
        self,
        file_record: ImportFileRecord,
        content_hash: str,
        dedupe_policy: str,
        task_id: str,
    ) -> None:
        key = self._manifest_key_for_file(file_record, content_hash, dedupe_policy)
        manifest = self._load_manifest()
        manifest[key] = {
            "hash": content_hash,
            "imported": True,
            "timestamp": _now(),
            "task_id": task_id,
            "name": file_record.name,
            "source_path": file_record.source_path or "",
            "source_kind": file_record.source_kind,
        }
        self._save_manifest(manifest)

