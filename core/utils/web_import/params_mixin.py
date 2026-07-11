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


class ImportParamsMixin:
    """Task parameter normalization."""
    def _normalize_common_import_params(self, payload: Dict[str, Any], *, default_dedupe: str) -> Dict[str, Any]:
        input_mode = str(payload.get("input_mode", "text") or "text").strip().lower()
        if input_mode not in {"text", "json"}:
            raise ValueError("input_mode 必须为 text 或 json")

        file_concurrency = _coerce_int(
            payload.get("file_concurrency", self._default_file_concurrency()),
            self._default_file_concurrency(),
        )
        chunk_concurrency = _coerce_int(
            payload.get("chunk_concurrency", self._default_chunk_concurrency()),
            self._default_chunk_concurrency(),
        )
        file_concurrency = _clamp(file_concurrency, 1, self._max_file_concurrency())
        chunk_concurrency = _clamp(chunk_concurrency, 1, self._max_chunk_concurrency())

        llm_enabled = _coerce_bool(payload.get("llm_enabled", True), True)
        strategy_override = parse_import_strategy(
            payload.get("strategy_override", "auto"),
            default=ImportStrategy.AUTO,
        ).value

        dedupe_policy = str(payload.get("dedupe_policy", default_dedupe) or default_dedupe).strip().lower()
        if dedupe_policy not in {"content_hash", "manifest", "none"}:
            raise ValueError("dedupe_policy 必须为 content_hash/manifest/none")

        chat_log = _coerce_bool(payload.get("chat_log"), False)
        chat_reference_time = str(payload.get("chat_reference_time") or "").strip() or None
        force = _coerce_bool(payload.get("force"), False)
        clear_manifest = _coerce_bool(payload.get("clear_manifest"), False)

        return {
            "input_mode": input_mode,
            "file_concurrency": file_concurrency,
            "chunk_concurrency": chunk_concurrency,
            "llm_enabled": llm_enabled,
            "strategy_override": strategy_override,
            "chat_log": chat_log,
            "chat_reference_time": chat_reference_time,
            "force": force,
            "clear_manifest": clear_manifest,
            "dedupe_policy": dedupe_policy,
        }

    def _normalize_params(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        params = self._normalize_common_import_params(payload, default_dedupe="content_hash")
        params["task_kind"] = "upload"
        return params

    def _normalize_raw_scan_params(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        params = self._normalize_common_import_params(payload, default_dedupe="manifest")
        alias = str(payload.get("alias") or "raw").strip()
        relative_path = str(payload.get("relative_path") or "").strip()
        glob_pattern = str(payload.get("glob") or "*").strip() or "*"
        recursive = _coerce_bool(payload.get("recursive"), True)
        if ".." in relative_path.replace("\\", "/").split("/"):
            raise ValueError("relative_path 不允许包含 ..")
        params.update(
            {
                "task_kind": "raw_scan",
                "alias": alias,
                "relative_path": relative_path,
                "glob": glob_pattern,
                "recursive": recursive,
            }
        )
        return params

    def _normalize_lpmm_openie_params(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        params = self._normalize_common_import_params(payload, default_dedupe="manifest")
        alias = str(payload.get("alias") or "lpmm").strip()
        relative_path = str(payload.get("relative_path") or "").strip()
        include_all_json = _coerce_bool(payload.get("include_all_json"), False)
        params.update(
            {
                "task_kind": "lpmm_openie",
                "alias": alias,
                "relative_path": relative_path,
                "include_all_json": include_all_json,
                "input_mode": "json",
            }
        )
        return params

    def _normalize_temporal_backfill_params(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        alias = str(payload.get("alias") or "plugin_data").strip()
        relative_path = str(payload.get("relative_path") or "").strip()
        dry_run = _coerce_bool(payload.get("dry_run"), False)
        no_created_fallback = _coerce_bool(payload.get("no_created_fallback"), False)
        limit = _parse_optional_positive_int(payload.get("limit"), "limit") or 100000
        return {
            "task_kind": "temporal_backfill",
            "alias": alias,
            "relative_path": relative_path,
            "dry_run": dry_run,
            "no_created_fallback": no_created_fallback,
            "limit": limit,
        }

    def _normalize_lpmm_convert_params(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        alias = str(payload.get("alias") or "lpmm").strip()
        relative_path = str(payload.get("relative_path") or "").strip()
        target_alias = str(payload.get("target_alias") or "plugin_data").strip()
        target_relative_path = str(payload.get("target_relative_path") or "").strip()
        dimension = _parse_optional_positive_int(payload.get("dimension"), "dimension") or _coerce_int(
            self._cfg("embedding.dimension", 384),
            384,
        )
        batch_size = _parse_optional_positive_int(payload.get("batch_size"), "batch_size") or 1024
        return {
            "task_kind": "lpmm_convert",
            "alias": alias,
            "relative_path": relative_path,
            "target_alias": target_alias,
            "target_relative_path": target_relative_path,
            "dimension": dimension,
            "batch_size": batch_size,
        }

    def _normalize_by_task_kind(self, task_kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        kind = str(task_kind or "").strip().lower()
        if kind in {"upload", "paste"}:
            params = self._normalize_params(payload)
            params["task_kind"] = kind
            return params
        if kind == "maibot_migration":
            return self._normalize_migration_params(payload)
        if kind == "raw_scan":
            return self._normalize_raw_scan_params(payload)
        if kind == "lpmm_openie":
            return self._normalize_lpmm_openie_params(payload)
        if kind == "temporal_backfill":
            return self._normalize_temporal_backfill_params(payload)
        if kind == "lpmm_convert":
            return self._normalize_lpmm_convert_params(payload)
        # upload/paste 默认走通用文本导入参数
        return self._normalize_params(payload)

    def _normalize_migration_params(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        source_db = str(payload.get("source_db") or "").strip()
        if not source_db:
            source_db = str(self._default_maibot_source_db())

        time_from = str(payload.get("time_from") or "").strip() or None
        time_to = str(payload.get("time_to") or "").strip() or None

        stream_ids = _coerce_list(payload.get("stream_ids"))
        group_ids = _coerce_list(payload.get("group_ids"))
        user_ids = _coerce_list(payload.get("user_ids"))

        start_id = _parse_optional_positive_int(payload.get("start_id"), "start_id")
        end_id = _parse_optional_positive_int(payload.get("end_id"), "end_id")
        if start_id is not None and end_id is not None and start_id > end_id:
            raise ValueError("start_id 不能大于 end_id")

        read_batch_size = _parse_optional_positive_int(payload.get("read_batch_size"), "read_batch_size") or 2000
        commit_window_rows = _parse_optional_positive_int(payload.get("commit_window_rows"), "commit_window_rows") or 20000
        embed_batch_size = _parse_optional_positive_int(payload.get("embed_batch_size"), "embed_batch_size") or 256
        entity_embed_batch_size = (
            _parse_optional_positive_int(payload.get("entity_embed_batch_size"), "entity_embed_batch_size") or 512
        )
        embed_workers = _parse_optional_positive_int(payload.get("embed_workers"), "embed_workers")
        max_errors = _parse_optional_positive_int(payload.get("max_errors"), "max_errors") or 500
        log_every = _parse_optional_positive_int(payload.get("log_every"), "log_every") or 5000
        preview_limit = _parse_optional_positive_int(payload.get("preview_limit"), "preview_limit") or 20

        no_resume = _coerce_bool(payload.get("no_resume"), False)
        reset_state = _coerce_bool(payload.get("reset_state"), False)
        dry_run = _coerce_bool(payload.get("dry_run"), False)
        verify_only = _coerce_bool(payload.get("verify_only"), False)

        return {
            "task_kind": "maibot_migration",
            "source_db": source_db,
            "target_data_dir": str(self._resolve_data_dir()),
            "time_from": time_from,
            "time_to": time_to,
            "stream_ids": stream_ids,
            "group_ids": group_ids,
            "user_ids": user_ids,
            "start_id": start_id,
            "end_id": end_id,
            "read_batch_size": read_batch_size,
            "commit_window_rows": commit_window_rows,
            "embed_batch_size": embed_batch_size,
            "entity_embed_batch_size": entity_embed_batch_size,
            "embed_workers": embed_workers,
            "max_errors": max_errors,
            "log_every": log_every,
            "preview_limit": preview_limit,
            "no_resume": no_resume,
            "reset_state": reset_state,
            "dry_run": dry_run,
            "verify_only": verify_only,
        }

