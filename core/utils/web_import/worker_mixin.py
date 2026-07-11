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


class ImportWorkerMixin:
    """Worker loop and task runner."""
    async def _worker_loop(self) -> None:
        logger.info("Web 导入任务 worker 已启动")
        while True:
            if self._stopping:
                break

            task_id: Optional[str] = None
            async with self._lock:
                while self._queue:
                    candidate = self._queue.popleft()
                    t = self._tasks.get(candidate)
                    if not t:
                        continue
                    if t.status == "cancelled":
                        continue
                    task_id = candidate
                    self._active_task_id = candidate
                    break

            if not task_id:
                await asyncio.sleep(0.2)
                continue

            try:
                await self._run_task(task_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"导入任务执行失败 task={task_id}: {e}\n{traceback.format_exc()}")
                async with self._lock:
                    task = self._tasks.get(task_id)
                    if task and task.status not in {"cancelled", "completed", "completed_with_errors"}:
                        task.status = "failed"
                        task.current_step = "failed"
                        task.error = str(e)
                        task.finished_at = _now()
                        task.updated_at = _now()
            finally:
                should_cleanup = await self._should_cleanup_task_temp(task_id)
                async with self._lock:
                    if self._active_task_id == task_id:
                        self._active_task_id = None
                if should_cleanup:
                    await self._cleanup_task_temp_files(task_id)

        logger.info("Web 导入任务 worker 已停止")

    async def _cleanup_task_temp_files(self, task_id: str) -> None:
        task_dir = self._temp_root / task_id
        if not task_dir.exists():
            return
        try:
            for child in task_dir.rglob("*"):
                if child.is_file():
                    child.unlink(missing_ok=True)
            for child in sorted(task_dir.rglob("*"), reverse=True):
                if child.is_dir():
                    child.rmdir()
            task_dir.rmdir()
        except Exception as e:
            logger.warning(f"清理任务临时文件失败 task={task_id}: {e}")

    def _task_report_path(self, task_id: str) -> Path:
        self._reports_root.mkdir(parents=True, exist_ok=True)
        return self._reports_root / f"{task_id}_summary.json"

    def _write_task_report(self, task: ImportTaskRecord) -> None:
        path = self._task_report_path(task.task_id)
        payload = task.to_detail(include_chunks=False)
        payload["generated_at"] = _now()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        task.artifact_paths["summary"] = str(path)

    async def _run_task(self, task_id: str) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task.status = "preparing"
            task.current_step = "preparing"
            task.started_at = _now()
            task.updated_at = _now()
            if task.params.get("clear_manifest"):
                self._clear_manifest()

        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            if task.status == "cancel_requested":
                task.status = "cancelled"
                task.current_step = "cancelled"
                task.finished_at = _now()
                task.updated_at = _now()
                return
            task.status = "running"
            task.current_step = "running"
            task.updated_at = _now()

        task_kind = str(task.params.get("task_kind") or task.source).strip().lower()
        if task_kind == "maibot_migration":
            if not task.files:
                raise RuntimeError("迁移任务缺少文件记录")
            await self._process_maibot_migration(task_id, task.files[0])
        elif task_kind == "temporal_backfill":
            if not task.files:
                raise RuntimeError("回填任务缺少文件记录")
            await self._process_temporal_backfill(task_id, task.files[0])
        elif task_kind == "lpmm_convert":
            if not task.files:
                raise RuntimeError("转换任务缺少文件记录")
            await self._process_lpmm_convert(task_id, task.files[0])
        else:
            file_semaphore = asyncio.Semaphore(task.params["file_concurrency"])
            chunk_semaphore = asyncio.Semaphore(task.params["chunk_concurrency"])
            jobs = [
                asyncio.create_task(self._process_file(task_id, f, file_semaphore, chunk_semaphore))
                for f in task.files
            ]
            await asyncio.gather(*jobs, return_exceptions=True)

        write_changed_payload: Optional[Dict[str, Any]] = None
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            self._recompute_task_progress(task)
            has_failed = any(
                (f.status == "failed")
                or (f.failed_chunks > 0)
                or bool(str(f.error or "").strip())
                for f in task.files
            )
            has_cancelled = any(f.status == "cancelled" for f in task.files)
            has_completed = any(f.status == "completed" for f in task.files)

            # 统一按文件真实终态收敛任务状态，避免出现“任务已取消但文件已完成”的矛盾结果。
            if has_failed and not has_cancelled:
                task.status = "completed_with_errors"
                task.current_step = "completed_with_errors"
            elif has_cancelled and not has_completed:
                task.status = "cancelled"
                task.current_step = "cancelled"
            elif has_cancelled and has_completed:
                task.status = "cancelled"
                task.current_step = "cancelled"
            else:
                task.status = "completed"
                task.current_step = "completed"
            task.finished_at = _now()
            task.updated_at = _now()
            try:
                self._write_task_report(task)
            except Exception as report_err:
                logger.warning(f"写入任务报告失败 task={task_id}: {report_err}")
            task_kind = str(task.params.get("task_kind") or task.source).strip().lower()
            write_task_kinds = {"upload", "paste", "raw_scan", "lpmm_openie", "maibot_migration", "lpmm_convert"}
            has_written_chunks = (task.done_chunks > 0) or any(f.done_chunks > 0 for f in task.files)
            if task_kind in write_task_kinds and has_written_chunks:
                write_changed_payload = {
                    "task_id": task.task_id,
                    "task_kind": task_kind,
                    "status": task.status,
                    "done_chunks": task.done_chunks,
                    "finished_at": task.finished_at,
                }

        if write_changed_payload:
            await self._notify_write_changed(write_changed_payload)

    async def _process_file(
        self,
        task_id: str,
        file_record: ImportFileRecord,
        file_semaphore: asyncio.Semaphore,
        chunk_semaphore: asyncio.Semaphore,
    ) -> None:
        async with file_semaphore:
            await self._set_file_state(task_id, file_record.file_id, "preparing", "preparing")
            if await self._is_cancel_requested(task_id):
                await self._set_file_cancelled(task_id, file_record.file_id, "任务已取消")
                return

            try:
                content = await self._read_file_content(file_record)
                content_hash = hashlib.md5(content.encode("utf-8", errors="ignore")).hexdigest()
                file_record.content_hash = content_hash
                task = self._tasks.get(task_id)
                if task:
                    dedupe_policy = str(task.params.get("dedupe_policy") or "none")
                    force = bool(task.params.get("force"))
                    if dedupe_policy != "none" and not force:
                        async with self._lock:
                            if self._is_manifest_hit(file_record, content_hash, dedupe_policy):
                                task2 = self._tasks.get(task_id)
                                if task2:
                                    f = self._find_file(task2, file_record.file_id)
                                    if f:
                                        f.status = "completed"
                                        f.current_step = "skipped"
                                        f.progress = 1.0
                                        f.total_chunks = 0
                                        f.done_chunks = 0
                                        f.failed_chunks = 0
                                        f.cancelled_chunks = 0
                                        f.detected_strategy_type = "skipped"
                                        f.error = ""
                                        f.updated_at = _now()
                                        self._recompute_task_progress(task2)
                                return
                if file_record.input_mode == "json":
                    await self._process_json_file(task_id, file_record, content, chunk_semaphore)
                else:
                    await self._process_text_file(task_id, file_record, content, chunk_semaphore)
                task3 = self._tasks.get(task_id)
                if task3:
                    dedupe_policy = str(task3.params.get("dedupe_policy") or "none")
                    f3 = self._find_file(task3, file_record.file_id)
                    if dedupe_policy != "none" and f3 and f3.status == "completed":
                        async with self._lock:
                            self._record_manifest_import(file_record, content_hash, dedupe_policy, task_id)
            except Exception as e:
                await self._set_file_failed(task_id, file_record.file_id, str(e))

    async def _should_cleanup_task_temp(self, task_id: str) -> bool:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return True
            for f in task.files:
                if f.status == "failed":
                    return False
            return True

    def _mark_task_cancelled_locked(self, task: ImportTaskRecord, reason: str) -> None:
        for f in task.files:
            if f.status in {"completed", "failed", "cancelled"}:
                continue
            f.status = "cancelled"
            f.current_step = "cancelled"
            f.error = reason
            additional_cancelled = 0
            for c in f.chunks:
                if c.status in {"completed", "failed", "cancelled"}:
                    continue
                c.status = "cancelled"
                c.step = "cancelled"
                c.retryable = False
                c.error = reason
                c.progress = 1.0
                c.updated_at = _now()
                additional_cancelled += 1
            if additional_cancelled > 0:
                f.cancelled_chunks += additional_cancelled
            self._recompute_file_progress(f)
            f.updated_at = _now()
        task.status = "cancelled"
        task.current_step = "cancelled"
        task.finished_at = _now()
        task.updated_at = _now()
        self._recompute_task_progress(task)

