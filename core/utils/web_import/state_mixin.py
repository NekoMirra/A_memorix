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


class ImportStateMixin:
    """Task/file/chunk state transitions and progress."""
    async def _set_file_state(self, task_id: str, file_id: str, status: str, step: str) -> None:
        if status not in FILE_STATUS:
            return
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            f = self._find_file(task, file_id)
            if not f:
                return
            f.status = status
            f.current_step = step
            f.updated_at = _now()
            task.updated_at = _now()
            if step in {"preparing", "splitting", "extracting", "writing", "saving"} and task.status in {"queued", "preparing"}:
                task.status = "running"
                task.current_step = "running"

    async def _append_file_warning(self, task_id: str, file_id: str, warning: str) -> None:
        warning_text = str(warning or "").strip()
        if not warning_text:
            return
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            file_record = self._find_file(task, file_id)
            if not file_record:
                return
            file_record.warning_count += 1
            file_record.warnings.append(warning_text)
            if len(file_record.warnings) > FILE_WARNING_KEEP_LIMIT:
                file_record.warnings = file_record.warnings[-FILE_WARNING_KEEP_LIMIT:]
            file_record.updated_at = _now()
            task.updated_at = _now()

    async def _append_file_warnings(self, task_id: str, file_id: str, warnings: List[str]) -> None:
        for warning in warnings:
            await self._append_file_warning(task_id, file_id, warning)

    async def _set_file_failed(self, task_id: str, file_id: str, error: str) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            f = self._find_file(task, file_id)
            if not f:
                return
            f.status = "failed"
            f.current_step = "failed"
            f.error = str(error)
            f.updated_at = _now()
            task.updated_at = _now()
            self._recompute_task_progress(task)

    async def _set_file_cancelled(self, task_id: str, file_id: str, reason: str) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            f = self._find_file(task, file_id)
            if not f:
                return
            f.status = "cancelled"
            f.current_step = "cancelled"
            f.error = reason
            additional_cancelled = 0
            for chunk in f.chunks:
                if chunk.status in {"completed", "failed", "cancelled"}:
                    continue
                chunk.status = "cancelled"
                chunk.step = "cancelled"
                chunk.retryable = False
                chunk.error = reason
                chunk.progress = 1.0
                chunk.updated_at = _now()
                additional_cancelled += 1
            if additional_cancelled > 0:
                f.cancelled_chunks += additional_cancelled
            self._recompute_file_progress(f)
            f.updated_at = _now()
            task.updated_at = _now()
            self._recompute_task_progress(task)

    async def _set_chunk_state(
        self,
        task_id: str,
        file_id: str,
        chunk_id: str,
        status: str,
        step: str,
        progress: float,
    ) -> None:
        if status not in CHUNK_STATUS:
            return
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            f = self._find_file(task, file_id)
            if not f:
                return
            c = self._find_chunk(f, chunk_id)
            if not c:
                return
            c.status = status
            c.step = step
            if status in {"queued", "extracting", "writing"}:
                c.error = ""
                c.failed_at = ""
                c.retryable = False
            c.progress = max(0.0, min(1.0, float(progress)))
            c.updated_at = _now()
            if f.status not in {"failed", "cancelled"}:
                f.status = "extracting" if status == "extracting" else "writing"
                f.current_step = step
            f.updated_at = _now()
            task.updated_at = _now()

    async def _set_chunk_completed(self, task_id: str, file_id: str, chunk_id: str) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            f = self._find_file(task, file_id)
            if not f:
                return
            c = self._find_chunk(f, chunk_id)
            if not c or c.status == "completed":
                return
            c.status = "completed"
            c.step = "completed"
            c.failed_at = ""
            c.retryable = False
            c.progress = 1.0
            c.updated_at = _now()
            f.done_chunks += 1
            self._recompute_file_progress(f)
            f.updated_at = _now()
            self._recompute_task_progress(task)

    async def _set_chunk_failed(self, task_id: str, file_id: str, chunk_id: str, error: str) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            f = self._find_file(task, file_id)
            if not f:
                return
            c = self._find_chunk(f, chunk_id)
            if not c or c.status == "failed":
                return
            failed_stage = str(c.step or "").strip().lower()
            if failed_stage in {"", "queued", "failed", "completed", "cancelled"}:
                failed_stage = str(f.current_step or "").strip().lower()
            if failed_stage in {"", "queued", "failed", "completed", "cancelled"}:
                failed_stage = "unknown"
            c.status = "failed"
            c.step = "failed"
            c.failed_at = failed_stage
            c.retryable = bool(f.input_mode == "text" and failed_stage == "extracting")
            c.error = str(error)
            c.progress = 1.0
            c.updated_at = _now()
            f.failed_chunks += 1
            self._recompute_file_progress(f)
            if not f.error:
                f.error = str(error)
            f.updated_at = _now()
            self._recompute_task_progress(task)

    async def _set_chunk_cancelled(self, task_id: str, file_id: str, chunk_id: str, reason: str) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            f = self._find_file(task, file_id)
            if not f:
                return
            c = self._find_chunk(f, chunk_id)
            if not c or c.status == "cancelled":
                return
            c.status = "cancelled"
            c.step = "cancelled"
            c.retryable = False
            c.error = reason
            c.progress = 1.0
            c.updated_at = _now()
            f.cancelled_chunks += 1
            self._recompute_file_progress(f)
            f.updated_at = _now()
            self._recompute_task_progress(task)

    async def _is_cancel_requested(self, task_id: str) -> bool:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return True
            return task.status == "cancel_requested"

    def _find_file(self, task: ImportTaskRecord, file_id: str) -> Optional[ImportFileRecord]:
        for f in task.files:
            if f.file_id == file_id:
                return f
        return None

    def _find_chunk(self, file_record: ImportFileRecord, chunk_id: str) -> Optional[ImportChunkRecord]:
        for c in file_record.chunks:
            if c.chunk_id == chunk_id:
                return c
        return None

    def _compute_ratio(self, done: int, total: int) -> float:
        if total <= 0:
            return 1.0
        return max(0.0, min(1.0, float(done) / float(total)))

    def _recompute_file_progress(self, file_record: ImportFileRecord) -> None:
        file_record.progress = self._compute_ratio(file_record.done_chunks, file_record.total_chunks)

    def _recompute_task_progress(self, task: ImportTaskRecord) -> None:
        total = 0
        done = 0
        failed = 0
        cancelled = 0
        for f in task.files:
            total += f.total_chunks
            done += f.done_chunks
            failed += f.failed_chunks
            cancelled += f.cancelled_chunks
        task.total_chunks = total
        task.done_chunks = done
        task.failed_chunks = failed
        task.cancelled_chunks = cancelled
        task.progress = self._compute_ratio(done, total)
        task.updated_at = _now()

