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


class ImportTaskApiMixin:
    """Public task queue API and worker lifecycle."""
    def _pending_task_count(self) -> int:
        pending = 0
        for task in self._tasks.values():
            if task.status in {"queued", "preparing", "running", "cancel_requested"}:
                pending += 1
        return pending

    async def _ensure_worker(self) -> None:
        async with self._lock:
            if self._worker_task and not self._worker_task.done():
                return
            self._stopping = False
            self._worker_task = asyncio.create_task(self._worker_loop())

    def _scan_files(
        self,
        base_path: Path,
        *,
        recursive: bool,
        glob_pattern: str,
        allowed_exts: Optional[set[str]] = None,
    ) -> List[Path]:
        if base_path.is_file():
            candidates = [base_path]
        else:
            if recursive:
                candidates = list(base_path.rglob(glob_pattern))
            else:
                candidates = list(base_path.glob(glob_pattern))
        out: List[Path] = []
        for p in candidates:
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if allowed_exts and ext not in allowed_exts:
                continue
            out.append(p.resolve())
        out.sort(key=lambda x: x.as_posix().lower())
        return out

    async def create_upload_task(self, files: List[Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._is_enabled():
            raise ValueError("导入功能已禁用")
        self._ensure_ready()
        if not files:
            raise ValueError("至少需要上传一个文件")

        params = self._normalize_params(payload)
        max_files = self._max_files_per_task()
        if len(files) > max_files:
            raise ValueError(f"单任务文件数超过上限: {max_files}")

        async with self._lock:
            if self._pending_task_count() >= self._queue_limit():
                raise ValueError("任务队列已满，请稍后重试")

            task = ImportTaskRecord(
                task_id=uuid.uuid4().hex,
                source="upload",
                params=params,
                status="queued",
                current_step="queued",
            )
            task_dir = self._temp_root / task.task_id
            task_dir.mkdir(parents=True, exist_ok=True)

            max_size = self._max_file_size_bytes()
            for idx, uploaded in enumerate(files):
                file_id = uuid.uuid4().hex
                if isinstance(uploaded, dict):
                    staged_path_raw = uploaded.get("staged_path") or uploaded.get("path") or ""
                    staged_path = Path(str(staged_path_raw or "")).expanduser().resolve()
                    if not staged_path.is_file():
                        raise ValueError(f"上传暂存文件不存在: {staged_path}")
                    name = _safe_filename(uploaded.get("filename") or uploaded.get("name") or staged_path.name)
                    ext = Path(name).suffix.lower()
                    if ext not in {".txt", ".md", ".json"}:
                        raise ValueError(f"不支持的文件类型: {name}")
                    if staged_path.stat().st_size > max_size:
                        raise ValueError(f"文件超过大小限制: {name}")
                    temp_path = task_dir / f"{file_id}_{name}"
                    shutil.copy2(staged_path, temp_path)
                else:
                    name = _safe_filename(getattr(uploaded, "filename", f"file_{idx}.txt"))
                    ext = Path(name).suffix.lower()
                    if ext not in {".txt", ".md", ".json"}:
                        raise ValueError(f"不支持的文件类型: {name}")
                    content = await uploaded.read()
                    if len(content) > max_size:
                        raise ValueError(f"文件超过大小限制: {name}")
                    temp_path = task_dir / f"{file_id}_{name}"
                    temp_path.write_bytes(content)
                file_mode = "json" if ext == ".json" else params["input_mode"]
                task.files.append(
                    ImportFileRecord(
                        file_id=file_id,
                        name=name,
                        source_kind="upload",
                        input_mode=file_mode,
                        temp_path=str(temp_path),
                    )
                )

            self._tasks[task.task_id] = task
            self._task_order.appendleft(task.task_id)
            self._queue.append(task.task_id)

        await self._ensure_worker()
        return task.to_summary()

    async def create_paste_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._is_enabled():
            raise ValueError("导入功能已禁用")
        self._ensure_ready()

        params = self._normalize_params(payload)
        params["task_kind"] = "paste"
        content = str(payload.get("content", "") or "")
        if not content.strip():
            raise ValueError("content 不能为空")
        if len(content) > self._max_paste_chars():
            raise ValueError(f"粘贴内容超过限制: {self._max_paste_chars()} 字符")

        name = _safe_filename(payload.get("name") or f"paste_{int(_now())}.txt")
        if params["input_mode"] == "json" and Path(name).suffix.lower() != ".json":
            name = f"{Path(name).stem}.json"

        async with self._lock:
            if self._pending_task_count() >= self._queue_limit():
                raise ValueError("任务队列已满，请稍后重试")

            task = ImportTaskRecord(
                task_id=uuid.uuid4().hex,
                source="paste",
                params=params,
                status="queued",
                current_step="queued",
            )
            task.files.append(
                ImportFileRecord(
                    file_id=uuid.uuid4().hex,
                    name=name,
                    source_kind="paste",
                    input_mode=params["input_mode"],
                    inline_content=content,
                )
            )
            self._tasks[task.task_id] = task
            self._task_order.appendleft(task.task_id)
            self._queue.append(task.task_id)

        await self._ensure_worker()
        return task.to_summary()

    async def create_raw_scan_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._is_enabled():
            raise ValueError("导入功能已禁用")
        self._ensure_ready()
        params = self._normalize_raw_scan_params(payload)
        source_path = self.resolve_path_alias(
            params["alias"],
            params["relative_path"],
            must_exist=True,
        )
        files = self._scan_files(
            source_path,
            recursive=bool(params["recursive"]),
            glob_pattern=str(params["glob"] or "*"),
            allowed_exts={".txt", ".md", ".json"},
        )
        if not files:
            raise ValueError("未找到可导入文件")
        if len(files) > self._max_files_per_task():
            raise ValueError(f"单任务文件数超过上限: {self._max_files_per_task()}")

        async with self._lock:
            if self._pending_task_count() >= self._queue_limit():
                raise ValueError("任务队列已满，请稍后重试")

            task = ImportTaskRecord(
                task_id=uuid.uuid4().hex,
                source="raw_scan",
                params=params,
                status="queued",
                current_step="queued",
            )
            for path in files:
                mode = "json" if path.suffix.lower() == ".json" else params["input_mode"]
                task.files.append(
                    ImportFileRecord(
                        file_id=uuid.uuid4().hex,
                        name=path.name,
                        source_kind="raw_scan",
                        input_mode=mode,
                        source_path=str(path),
                    )
                )
            self._tasks[task.task_id] = task
            self._task_order.appendleft(task.task_id)
            self._queue.append(task.task_id)

        await self._ensure_worker()
        return task.to_summary()

    async def create_lpmm_openie_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._is_enabled():
            raise ValueError("导入功能已禁用")
        self._ensure_ready()
        params = self._normalize_lpmm_openie_params(payload)
        source_path = self.resolve_path_alias(
            params["alias"],
            params["relative_path"],
            must_exist=True,
        )
        files: List[Path] = []
        if source_path.is_file():
            files = [source_path]
        else:
            files = self._scan_files(
                source_path,
                recursive=True,
                glob_pattern="*-openie.json",
                allowed_exts={".json"},
            )
            if not files and params.get("include_all_json"):
                files = self._scan_files(
                    source_path,
                    recursive=True,
                    glob_pattern="*.json",
                    allowed_exts={".json"},
                )
        if not files:
            raise ValueError("未找到 LPMM OpenIE JSON 文件")
        if len(files) > self._max_files_per_task():
            raise ValueError(f"单任务文件数超过上限: {self._max_files_per_task()}")

        async with self._lock:
            if self._pending_task_count() >= self._queue_limit():
                raise ValueError("任务队列已满，请稍后重试")
            task = ImportTaskRecord(
                task_id=uuid.uuid4().hex,
                source="lpmm_openie",
                params=params,
                status="queued",
                current_step="queued",
                schema_detected="lpmm_openie",
            )
            for path in files:
                task.files.append(
                    ImportFileRecord(
                        file_id=uuid.uuid4().hex,
                        name=path.name,
                        source_kind="lpmm_openie",
                        input_mode="json",
                        source_path=str(path),
                    )
                )
            self._tasks[task.task_id] = task
            self._task_order.appendleft(task.task_id)
            self._queue.append(task.task_id)

        await self._ensure_worker()
        return task.to_summary()

    async def create_temporal_backfill_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._is_enabled():
            raise ValueError("导入功能已禁用")
        params = self._normalize_temporal_backfill_params(payload)
        target_path = self.resolve_path_alias(
            params["alias"],
            params["relative_path"],
            must_exist=True,
        )
        if not target_path.is_dir():
            raise ValueError("temporal_backfill 目标路径必须为目录")

        async with self._lock:
            if self._pending_task_count() >= self._queue_limit():
                raise ValueError("任务队列已满，请稍后重试")
            task = ImportTaskRecord(
                task_id=uuid.uuid4().hex,
                source="temporal_backfill",
                params=params,
                status="queued",
                current_step="queued",
            )
            task.files.append(
                ImportFileRecord(
                    file_id=uuid.uuid4().hex,
                    name=f"temporal_backfill_{int(_now())}",
                    source_kind="temporal_backfill",
                    input_mode="json",
                    source_path=str(target_path),
                )
            )
            self._tasks[task.task_id] = task
            self._task_order.appendleft(task.task_id)
            self._queue.append(task.task_id)

        await self._ensure_worker()
        return task.to_summary()

    async def create_lpmm_convert_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._is_enabled():
            raise ValueError("导入功能已禁用")
        params = self._normalize_lpmm_convert_params(payload)
        source_path = self.resolve_path_alias(
            params["alias"],
            params["relative_path"],
            must_exist=True,
        )
        if not source_path.is_dir():
            raise ValueError("lpmm_convert 输入路径必须为目录")
        target_path = self.resolve_path_alias(
            params["target_alias"],
            params["target_relative_path"],
            must_exist=False,
        )
        target_path.mkdir(parents=True, exist_ok=True)
        if not target_path.is_dir():
            raise ValueError("lpmm_convert 目标路径必须为目录")

        async with self._lock:
            if self._pending_task_count() >= self._queue_limit():
                raise ValueError("任务队列已满，请稍后重试")
            task = ImportTaskRecord(
                task_id=uuid.uuid4().hex,
                source="lpmm_convert",
                params={**params, "source_path": str(source_path), "target_path": str(target_path)},
                status="queued",
                current_step="queued",
            )
            task.files.append(
                ImportFileRecord(
                    file_id=uuid.uuid4().hex,
                    name=f"lpmm_convert_{int(_now())}",
                    source_kind="lpmm_convert",
                    input_mode="json",
                    source_path=str(source_path),
                )
            )
            self._tasks[task.task_id] = task
            self._task_order.appendleft(task.task_id)
            self._queue.append(task.task_id)

        await self._ensure_worker()
        return task.to_summary()

    async def create_maibot_migration_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._is_enabled():
            raise ValueError("导入功能已禁用")
        self._ensure_ready()

        params = self._normalize_migration_params(payload)
        script_path = self._resolve_migration_script()
        if not script_path.exists():
            raise ValueError(f"迁移脚本不存在: {script_path}")

        async with self._lock:
            if self._pending_task_count() >= self._queue_limit():
                raise ValueError("任务队列已满，请稍后重试")

            task = ImportTaskRecord(
                task_id=uuid.uuid4().hex,
                source="maibot_migration",
                params=params,
                status="queued",
                current_step="queued",
            )
            task.files.append(
                ImportFileRecord(
                    file_id=uuid.uuid4().hex,
                    name=f"maibot_migration_{int(_now())}",
                    source_kind="maibot_migration",
                    input_mode="text",
                    inline_content=json.dumps(params, ensure_ascii=False),
                )
            )
            self._tasks[task.task_id] = task
            self._task_order.appendleft(task.task_id)
            self._queue.append(task.task_id)

        await self._ensure_worker()
        return task.to_summary()

    async def list_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        async with self._lock:
            task_ids = list(self._task_order)[: max(1, int(limit))]
            return [self._tasks[task_id].to_summary() for task_id in task_ids if task_id in self._tasks]

    async def get_task(self, task_id: str, include_chunks: bool = False) -> Optional[Dict[str, Any]]:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            return task.to_detail(include_chunks=include_chunks)

    async def get_chunks(self, task_id: str, file_id: str, offset: int = 0, limit: int = 50) -> Optional[Dict[str, Any]]:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            file_obj = self._find_file(task, file_id)
            if not file_obj:
                return None
            start = max(0, int(offset))
            size = max(1, min(500, int(limit)))
            items = file_obj.chunks[start : start + size]
            return {
                "task_id": task_id,
                "file_id": file_id,
                "offset": start,
                "limit": size,
                "total": len(file_obj.chunks),
                "file": file_obj.to_dict(include_chunks=False),
                "items": [x.to_dict() for x in items],
            }

    async def cancel_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            if task.status == "queued":
                self._mark_task_cancelled_locked(task, "任务已取消")
                self._queue = deque([x for x in self._queue if x != task_id])
            elif task.status in {"preparing", "running"}:
                task.status = "cancel_requested"
                task.current_step = "cancel_requested"
                task.updated_at = _now()
            return task.to_summary()

    def _build_retry_plan(self, task: ImportTaskRecord) -> Dict[str, Any]:
        chunk_retry_candidates: List[Tuple[ImportFileRecord, List[int]]] = []
        file_fallback_candidates: List[ImportFileRecord] = []
        skipped: List[Dict[str, str]] = []

        for file_obj in task.files:
            if file_obj.status == "cancelled":
                continue

            failed_chunks = [c for c in file_obj.chunks if c.status == "failed"]
            has_file_level_failure = file_obj.status == "failed" and not failed_chunks
            if has_file_level_failure:
                file_fallback_candidates.append(file_obj)
                continue

            if not failed_chunks:
                continue

            retry_indexes: List[int] = []
            has_non_retryable = False
            for chunk in failed_chunks:
                failed_at = str(chunk.failed_at or "").strip().lower()
                retryable = bool(chunk.retryable) or (
                    file_obj.input_mode == "text" and failed_at == "extracting"
                )
                if retryable:
                    try:
                        retry_indexes.append(int(chunk.index))
                    except Exception:
                        has_non_retryable = True
                else:
                    has_non_retryable = True

            if has_non_retryable:
                file_fallback_candidates.append(file_obj)
                continue

            retry_indexes = sorted(set(retry_indexes))
            if retry_indexes:
                chunk_retry_candidates.append((file_obj, retry_indexes))
            else:
                skipped.append(
                    {
                        "file_name": file_obj.name,
                        "source_kind": file_obj.source_kind,
                        "reason": "no_retryable_failed_chunks",
                    }
                )

        unique_fallback: List[ImportFileRecord] = []
        fallback_seen = set()
        for file_obj in file_fallback_candidates:
            if file_obj.file_id in fallback_seen:
                continue
            fallback_seen.add(file_obj.file_id)
            unique_fallback.append(file_obj)

        return {
            "chunk_retry_candidates": chunk_retry_candidates,
            "file_fallback_candidates": unique_fallback,
            "skipped": skipped,
        }

    def _clone_failed_file_for_retry(
        self,
        retry_task: ImportTaskRecord,
        failed_file: ImportFileRecord,
        task_dir: Path,
        *,
        retry_mode: str,
        retry_chunk_indexes: Optional[List[int]] = None,
    ) -> Tuple[bool, str]:
        source_kind = str(failed_file.source_kind or "").strip().lower()
        retry_chunk_indexes = list(retry_chunk_indexes or [])

        if source_kind == "upload":
            candidate_paths: List[Path] = []
            if failed_file.temp_path:
                candidate_paths.append(Path(failed_file.temp_path))
            if failed_file.source_path:
                candidate_paths.append(Path(failed_file.source_path))
            src_path = next((p for p in candidate_paths if p.exists() and p.is_file()), None)
            if src_path is None:
                return False, "upload_source_missing"
            data = src_path.read_bytes()
            file_id = uuid.uuid4().hex
            name = _safe_filename(failed_file.name)
            dst = task_dir / f"{file_id}_{name}"
            dst.write_bytes(data)
            retry_task.files.append(
                ImportFileRecord(
                    file_id=file_id,
                    name=name,
                    source_kind="upload",
                    input_mode=failed_file.input_mode,
                    temp_path=str(dst),
                    retry_mode=retry_mode,
                    retry_chunk_indexes=retry_chunk_indexes,
                )
            )
            return True, ""

        if source_kind == "paste":
            if failed_file.inline_content is None:
                return False, "paste_content_missing"
            retry_task.files.append(
                ImportFileRecord(
                    file_id=uuid.uuid4().hex,
                    name=_safe_filename(failed_file.name),
                    source_kind="paste",
                    input_mode=failed_file.input_mode,
                    inline_content=failed_file.inline_content,
                    retry_mode=retry_mode,
                    retry_chunk_indexes=retry_chunk_indexes,
                )
            )
            return True, ""

        if source_kind == "maibot_migration":
            retry_task.files.append(
                ImportFileRecord(
                    file_id=uuid.uuid4().hex,
                    name=_safe_filename(failed_file.name),
                    source_kind="maibot_migration",
                    input_mode="text",
                    inline_content=failed_file.inline_content,
                    retry_mode="file_fallback",
                    retry_chunk_indexes=[],
                )
            )
            return True, ""

        if source_kind in {"raw_scan", "lpmm_openie", "lpmm_convert", "temporal_backfill"}:
            retry_task.files.append(
                ImportFileRecord(
                    file_id=uuid.uuid4().hex,
                    name=_safe_filename(failed_file.name),
                    source_kind=source_kind,
                    input_mode=failed_file.input_mode,
                    source_path=failed_file.source_path,
                    inline_content=failed_file.inline_content,
                    retry_mode=retry_mode,
                    retry_chunk_indexes=retry_chunk_indexes,
                )
            )
            return True, ""

        return False, f"unsupported_source_kind:{source_kind or 'unknown'}"

    async def retry_failed(self, task_id: str, overrides: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            retry_plan = self._build_retry_plan(task)
            chunk_retry_candidates = list(retry_plan["chunk_retry_candidates"])
            file_fallback_candidates = list(retry_plan["file_fallback_candidates"])
            skipped_candidates = list(retry_plan["skipped"])
            if not chunk_retry_candidates and not file_fallback_candidates:
                raise ValueError("当前任务没有可重试失败项")
            base_params = dict(task.params)
            task_kind = str(task.params.get("task_kind") or "").strip().lower()

        if overrides:
            base_params.update(overrides)
        params = self._normalize_by_task_kind(task_kind, base_params)
        params["retry_parent_task_id"] = task_id
        params["retry_strategy"] = "chunk_first_auto_file_fallback"

        async with self._lock:
            if self._pending_task_count() >= self._queue_limit():
                raise ValueError("任务队列已满，请稍后重试")
            retry_task = ImportTaskRecord(
                task_id=uuid.uuid4().hex,
                source=task.source,
                params=params,
                status="queued",
                current_step="queued",
                schema_detected=task.schema_detected,
                retry_parent_task_id=task_id,
            )

            task_dir = self._temp_root / retry_task.task_id
            task_dir.mkdir(parents=True, exist_ok=True)

            retry_summary = {
                "chunk_retry_files": 0,
                "chunk_retry_chunks": 0,
                "file_fallback_files": 0,
                "skipped_files": 0,
                "parent_task_id": task_id,
            }
            skipped_details = list(skipped_candidates)

            for file_obj, chunk_indexes in chunk_retry_candidates:
                ok, reason = self._clone_failed_file_for_retry(
                    retry_task,
                    file_obj,
                    task_dir,
                    retry_mode="chunk",
                    retry_chunk_indexes=chunk_indexes,
                )
                if ok:
                    retry_summary["chunk_retry_files"] += 1
                    retry_summary["chunk_retry_chunks"] += len(chunk_indexes)
                else:
                    skipped_details.append(
                        {
                            "file_name": file_obj.name,
                            "source_kind": file_obj.source_kind,
                            "reason": reason,
                        }
                    )

            for file_obj in file_fallback_candidates:
                ok, reason = self._clone_failed_file_for_retry(
                    retry_task,
                    file_obj,
                    task_dir,
                    retry_mode="file_fallback",
                    retry_chunk_indexes=[],
                )
                if ok:
                    retry_summary["file_fallback_files"] += 1
                else:
                    skipped_details.append(
                        {
                            "file_name": file_obj.name,
                            "source_kind": file_obj.source_kind,
                            "reason": reason,
                        }
                    )

            retry_summary["skipped_files"] = len(skipped_details)
            if skipped_details:
                retry_summary["skipped_details"] = skipped_details
            retry_task.retry_summary = retry_summary

            if not retry_task.files:
                raise ValueError("无可执行的重试输入：失败项均无法构建重试任务")

            self._tasks[retry_task.task_id] = retry_task
            self._task_order.appendleft(retry_task.task_id)
            self._queue.append(retry_task.task_id)
            logger.info(
                "重试任务已创建 "
                f"parent={task_id} retry={retry_task.task_id} "
                f"chunk_files={retry_summary['chunk_retry_files']} "
                f"chunk_chunks={retry_summary['chunk_retry_chunks']} "
                f"file_fallback={retry_summary['file_fallback_files']} "
                f"skipped={retry_summary['skipped_files']}"
            )

        await self._ensure_worker()
        return retry_task.to_summary()

    async def shutdown(self) -> None:
        async with self._lock:
            self._stopping = True
            for task in self._tasks.values():
                if task.status in {"queued", "preparing", "running", "cancel_requested"}:
                    self._mark_task_cancelled_locked(task, "服务关闭")
            self._queue.clear()
            worker = self._worker_task
            self._worker_task = None

        if worker:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        self._cleanup_temp_root()

    def _cleanup_temp_root(self) -> None:
        try:
            if not self._temp_root.exists():
                return
            for child in self._temp_root.rglob("*"):
                if child.is_file():
                    child.unlink(missing_ok=True)
            for child in sorted(self._temp_root.rglob("*"), reverse=True):
                if child.is_dir():
                    child.rmdir()
            self._temp_root.rmdir()
        except Exception as e:
            logger.warning(f"清理临时导入目录失败: {e}")

