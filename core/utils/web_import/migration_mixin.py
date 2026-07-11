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


class ImportMigrationMixin:
    """External migration / convert / temporal backfill processors."""
    def _build_maibot_migration_command(self, params: Dict[str, Any]) -> List[str]:
        script_path = self._resolve_migration_script()
        if not script_path.exists():
            raise RuntimeError(f"迁移脚本不存在: {script_path}")

        cmd = [
            sys.executable,
            str(script_path),
            "--source-db",
            str(params["source_db"]),
            "--target-data-dir",
            str(params["target_data_dir"]),
            "--read-batch-size",
            str(params["read_batch_size"]),
            "--commit-window-rows",
            str(params["commit_window_rows"]),
            "--embed-batch-size",
            str(params["embed_batch_size"]),
            "--entity-embed-batch-size",
            str(params["entity_embed_batch_size"]),
            "--max-errors",
            str(params["max_errors"]),
            "--log-every",
            str(params["log_every"]),
            "--preview-limit",
            str(params["preview_limit"]),
            "--yes",
        ]

        if params.get("embed_workers") is not None:
            cmd.extend(["--embed-workers", str(params["embed_workers"])])
        if params.get("start_id") is not None:
            cmd.extend(["--start-id", str(params["start_id"])])
        if params.get("end_id") is not None:
            cmd.extend(["--end-id", str(params["end_id"])])
        if params.get("time_from"):
            cmd.extend(["--time-from", str(params["time_from"])])
        if params.get("time_to"):
            cmd.extend(["--time-to", str(params["time_to"])])

        for sid in params.get("stream_ids") or []:
            cmd.extend(["--stream-id", str(sid)])
        for gid in params.get("group_ids") or []:
            cmd.extend(["--group-id", str(gid)])
        for uid in params.get("user_ids") or []:
            cmd.extend(["--user-id", str(uid)])

        if params.get("reset_state"):
            cmd.append("--reset-state")
        if params.get("no_resume"):
            cmd.append("--no-resume")
        if params.get("dry_run"):
            cmd.append("--dry-run")
        if params.get("verify_only"):
            cmd.append("--verify-only")

        return cmd

    async def _ensure_maibot_migration_chunk(
        self,
        task_id: str,
        file_id: str,
        *,
        chunk_type: str = "maibot_migration",
        preview: str = "MaiBot chat_history 迁移任务",
    ) -> str:
        chunk_id = f"{file_id}_{chunk_type}"
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return chunk_id
            f = self._find_file(task, file_id)
            if not f:
                return chunk_id
            if not f.chunks:
                f.chunks = [
                    ImportChunkRecord(
                        chunk_id=chunk_id,
                        index=0,
                        chunk_type=chunk_type,
                        status="queued",
                        step="queued",
                        progress=0.0,
                        content_preview=preview,
                    )
                ]
                f.total_chunks = 1
                f.done_chunks = 0
                f.failed_chunks = 0
                f.cancelled_chunks = 0
                f.progress = 0.0
                f.updated_at = _now()
                self._recompute_task_progress(task)
            else:
                chunk_id = f.chunks[0].chunk_id
        return chunk_id

    async def _refresh_maibot_progress_from_state(
        self,
        task_id: str,
        file_id: str,
        chunk_id: str,
        state_path: Path,
    ) -> None:
        if not state_path.exists():
            return
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return

        stats = payload.get("stats", {}) if isinstance(payload, dict) else {}
        if not isinstance(stats, dict):
            stats = {}

        total = max(0, _coerce_int(stats.get("source_matched_total", 0), 0))
        scanned = max(0, _coerce_int(stats.get("scanned_rows", 0), 0))
        bad = max(0, _coerce_int(stats.get("bad_rows", 0), 0))
        done = max(0, scanned - bad)
        migrated = max(0, _coerce_int(stats.get("migrated_rows", 0), 0))
        last_id = max(0, _coerce_int(stats.get("last_committed_id", 0), 0))

        if total <= 0:
            total = max(1, scanned)

        chunk_progress = max(0.0, min(1.0, float(scanned) / float(total))) if total > 0 else 0.0
        preview = f"scanned={scanned}/{total}, migrated={migrated}, bad={bad}, last_id={last_id}"

        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            f = self._find_file(task, file_id)
            if not f:
                return
            c = self._find_chunk(f, chunk_id)
            if c:
                if c.status not in {"completed", "failed", "cancelled"}:
                    c.status = "writing"
                    c.step = "migrating"
                c.progress = chunk_progress
                c.content_preview = preview
                c.updated_at = _now()
            f.total_chunks = total
            f.done_chunks = done
            f.failed_chunks = bad
            f.cancelled_chunks = 0
            self._recompute_file_progress(f)
            if f.status not in {"failed", "cancelled"}:
                f.status = "writing"
                f.current_step = "migrating"
            f.updated_at = _now()
            self._recompute_task_progress(task)

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except Exception:
            try:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=3.0)
            except Exception:
                pass

    async def _reload_stores_after_external_migration(self) -> None:
        async with self._storage_lock:
            try:
                if self.plugin.vector_store and self.plugin.vector_store.has_data():
                    self.plugin.vector_store.load()
            except Exception as e:
                logger.warning(f"迁移后重载 VectorStore 失败: {e}")
            try:
                if self.plugin.graph_store and self.plugin.graph_store.has_data():
                    self.plugin.graph_store.load()
            except Exception as e:
                logger.warning(f"迁移后重载 GraphStore 失败: {e}")

    async def _process_maibot_migration(self, task_id: str, file_record: ImportFileRecord) -> None:
        await self._set_file_strategy(task_id, file_record.file_id, "maibot_migration")
        await self._set_file_state(task_id, file_record.file_id, "preparing", "preparing")
        chunk_id = await self._ensure_maibot_migration_chunk(
            task_id,
            file_record.file_id,
            chunk_type="maibot_migration",
            preview="MaiBot chat_history 迁移任务",
        )
        await self._set_chunk_state(task_id, file_record.file_id, chunk_id, "writing", "migrating", 0.0)

        task = self._tasks.get(task_id)
        if not task:
            await self._set_file_failed(task_id, file_record.file_id, "任务不存在")
            return
        params = dict(task.params)

        command = self._build_maibot_migration_command(params)
        project_root = self._resolve_repo_root()
        state_path = Path(params["target_data_dir"]) / "migration_state" / "chat_history_resume.json"
        report_path = Path(params["target_data_dir"]) / "migration_state" / "chat_history_report.json"

        logger.info(f"开始执行 MaiBot 迁移任务: {' '.join(command)}")
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(project_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_lines: List[str] = []
        stderr_lines: List[str] = []

        async def _drain(stream: Optional[asyncio.StreamReader], target: List[str]) -> None:
            if stream is None:
                return
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                target.append(text)
                if len(target) > 120:
                    del target[:-120]

        drain_tasks = [
            asyncio.create_task(_drain(process.stdout, stdout_lines)),
            asyncio.create_task(_drain(process.stderr, stderr_lines)),
        ]

        cancelled = False
        return_code: Optional[int] = None
        try:
            while True:
                if await self._is_cancel_requested(task_id):
                    cancelled = True
                    await self._terminate_process(process)
                    break

                await self._refresh_maibot_progress_from_state(task_id, file_record.file_id, chunk_id, state_path)
                try:
                    return_code = await asyncio.wait_for(process.wait(), timeout=1.0)
                    break
                except asyncio.TimeoutError:
                    continue
        finally:
            await asyncio.gather(*drain_tasks, return_exceptions=True)

        if cancelled:
            await self._set_chunk_cancelled(task_id, file_record.file_id, chunk_id, "任务已取消")
            await self._set_file_cancelled(task_id, file_record.file_id, "任务已取消")
            return

        await self._refresh_maibot_progress_from_state(task_id, file_record.file_id, chunk_id, state_path)

        report: Dict[str, Any] = {}
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                report = {}

        stats = report.get("stats", {}) if isinstance(report, dict) else {}
        if not isinstance(stats, dict):
            stats = {}
        bad_rows = max(0, _coerce_int(stats.get("bad_rows", 0), 0))

        if return_code in {0, 2}:
            await self._set_file_state(task_id, file_record.file_id, "saving", "saving")
            await self._reload_stores_after_external_migration()

            async with self._lock:
                task2 = self._tasks.get(task_id)
                if not task2:
                    return
                f = self._find_file(task2, file_record.file_id)
                if not f:
                    return
                c = self._find_chunk(f, chunk_id)
                if c and c.status not in {"cancelled", "failed"}:
                    c.status = "completed"
                    c.step = "completed"
                    c.progress = 1.0
                    c.updated_at = _now()
                if f.total_chunks <= 0:
                    f.total_chunks = 1
                if f.done_chunks + f.failed_chunks <= 0:
                    f.done_chunks = f.total_chunks - bad_rows
                    f.failed_chunks = bad_rows
                f.done_chunks = max(0, min(f.done_chunks, f.total_chunks))
                f.failed_chunks = max(0, min(f.failed_chunks, f.total_chunks))
                f.cancelled_chunks = 0
                self._recompute_file_progress(f)
                f.status = "completed"
                f.current_step = "completed"
                if bad_rows > 0 and not f.error:
                    f.error = f"迁移完成，但存在坏行: {bad_rows}"
                f.updated_at = _now()
                self._recompute_task_progress(task2)
            return

        fail_reason = ""
        if isinstance(report, dict):
            fail_reason = str(report.get("fail_reason") or "").strip()
        tail = (stderr_lines[-1] if stderr_lines else "") or (stdout_lines[-1] if stdout_lines else "")
        detail = fail_reason or tail or f"迁移进程退出码: {return_code}"
        await self._set_chunk_failed(task_id, file_record.file_id, chunk_id, detail)
        await self._set_file_failed(task_id, file_record.file_id, detail)

    def _resolve_convert_script(self) -> Path:
        return Path(__file__).resolve().parents[2] / "scripts" / "convert_lpmm.py"

    def _cleanup_old_backups(self) -> None:
        keep = max(0, self._cfg_int("web.import.convert.keep_backup_count", 3))
        backup_root = self._resolve_backup_root()
        if not backup_root.exists() or keep <= 0:
            return
        dirs = [p for p in backup_root.iterdir() if p.is_dir() and p.name.startswith("lpmm_convert_")]
        dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for old in dirs[keep:]:
            try:
                shutil.rmtree(old, ignore_errors=True)
            except Exception:
                pass

    def _verify_convert_output(self, output_dir: Path) -> Dict[str, Any]:
        vectors = output_dir / "vectors"
        graph = output_dir / "graph"
        metadata = output_dir / "metadata"
        checks = {
            "vectors_exists": vectors.exists(),
            "graph_exists": graph.exists(),
            "metadata_exists": metadata.exists(),
            "vectors_nonempty": vectors.exists() and any(vectors.iterdir()),
            "graph_nonempty": graph.exists() and any(graph.iterdir()),
            "metadata_nonempty": metadata.exists() and any(metadata.iterdir()),
        }
        checks["ok"] = checks["vectors_exists"] and checks["graph_exists"] and checks["metadata_exists"]
        return checks

    async def _preflight_convert_runtime(self) -> Tuple[bool, str]:
        """使用当前服务解释器做 convert 依赖预检，避免子进程报错信息不透明。"""
        probe_code = (
            "import importlib\n"
            "mods=['networkx','scipy','pyarrow']\n"
            "failed=[]\n"
            "for m in mods:\n"
            "    try:\n"
            "        importlib.import_module(m)\n"
            "    except Exception as e:\n"
            "        failed.append(f'{m}:{e.__class__.__name__}:{e}')\n"
            "print('OK' if not failed else ';'.join(failed))\n"
        )
        try:
            probe = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                probe_code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(probe.communicate(), timeout=20.0)
        except Exception as e:
            return False, f"依赖预检执行失败: {e}"

        out = (stdout or b"").decode("utf-8", errors="replace").strip()
        err = (stderr or b"").decode("utf-8", errors="replace").strip()
        if probe.returncode != 0:
            detail = err or out or f"return_code={probe.returncode}"
            return False, f"依赖预检失败 (python={sys.executable}): {detail}"
        if out != "OK":
            return False, f"依赖预检失败 (python={sys.executable}): {out}"
        return True, ""

    async def _process_lpmm_convert(self, task_id: str, file_record: ImportFileRecord) -> None:
        await self._set_file_strategy(task_id, file_record.file_id, "lpmm_convert")
        await self._set_file_state(task_id, file_record.file_id, "preparing", "preflight")
        chunk_id = await self._ensure_maibot_migration_chunk(
            task_id,
            file_record.file_id,
            chunk_type="lpmm_convert",
            preview="LPMM 二进制转换任务",
        )
        await self._set_chunk_state(task_id, file_record.file_id, chunk_id, "writing", "converting", 0.05)

        task = self._tasks.get(task_id)
        if not task:
            await self._set_file_failed(task_id, file_record.file_id, "任务不存在")
            return
        params = dict(task.params)
        source_dir = Path(params.get("source_path") or "")
        target_dir = Path(params.get("target_path") or "")
        if not source_dir.exists() or not source_dir.is_dir():
            await self._set_file_failed(task_id, file_record.file_id, f"输入目录无效: {source_dir}")
            return
        if not target_dir.exists() or not target_dir.is_dir():
            await self._set_file_failed(task_id, file_record.file_id, f"目标目录无效: {target_dir}")
            return

        script_path = self._resolve_convert_script()
        if not script_path.exists():
            await self._set_file_failed(task_id, file_record.file_id, f"转换脚本不存在: {script_path}")
            return

        runtime_ok, runtime_detail = await self._preflight_convert_runtime()
        if not runtime_ok:
            await self._set_file_failed(task_id, file_record.file_id, runtime_detail)
            await self._set_chunk_failed(task_id, file_record.file_id, chunk_id, runtime_detail)
            return

        required_inputs = ["paragraph.parquet", "entity.parquet"]
        if not any((source_dir / name).exists() for name in required_inputs):
            await self._set_file_failed(
                task_id,
                file_record.file_id,
                f"输入目录缺少必要文件，至少需要其一: {', '.join(required_inputs)}",
            )
            return

        staging_root = self._resolve_staging_root()
        staging_root.mkdir(parents=True, exist_ok=True)
        staging_dir = staging_root / f"lpmm_convert_{task_id}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)

        # 简单空间预检：至少保留 512MB
        usage = shutil.disk_usage(str(target_dir))
        if usage.free < 512 * 1024 * 1024:
            await self._set_file_failed(task_id, file_record.file_id, "磁盘剩余空间不足（<512MB）")
            return

        cmd = [
            sys.executable,
            str(script_path),
            "--input",
            str(source_dir),
            "--output",
            str(staging_dir),
            "--dim",
            str(params.get("dimension", 384)),
            "--batch-size",
            str(params.get("batch_size", 1024)),
        ]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self._resolve_repo_root()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []

        async def _drain(stream: Optional[asyncio.StreamReader], target: List[str]) -> None:
            if stream is None:
                return
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    target.append(text)
                    if len(target) > 120:
                        del target[:-120]

        drain_tasks = [
            asyncio.create_task(_drain(process.stdout, stdout_lines)),
            asyncio.create_task(_drain(process.stderr, stderr_lines)),
        ]

        cancelled = False
        return_code: Optional[int] = None
        try:
            while True:
                if await self._is_cancel_requested(task_id):
                    cancelled = True
                    await self._terminate_process(process)
                    break
                try:
                    return_code = await asyncio.wait_for(process.wait(), timeout=1.0)
                    break
                except asyncio.TimeoutError:
                    continue
        finally:
            await asyncio.gather(*drain_tasks, return_exceptions=True)

        if cancelled:
            await self._set_chunk_cancelled(task_id, file_record.file_id, chunk_id, "任务已取消")
            await self._set_file_cancelled(task_id, file_record.file_id, "任务已取消")
            return
        if return_code != 0:
            detail = (stderr_lines[-1] if stderr_lines else "") or (stdout_lines[-1] if stdout_lines else "")
            await self._set_file_failed(task_id, file_record.file_id, detail or f"转换失败，退出码: {return_code}")
            await self._set_chunk_failed(task_id, file_record.file_id, chunk_id, detail or f"退出码: {return_code}")
            return

        await self._set_chunk_state(task_id, file_record.file_id, chunk_id, "writing", "verifying", 0.65)
        verify = self._verify_convert_output(staging_dir)
        async with self._lock:
            t = self._tasks.get(task_id)
            if t:
                t.artifact_paths["staging_dir"] = str(staging_dir)
                t.artifact_paths["verify"] = json.dumps(verify, ensure_ascii=False)
        if not verify.get("ok"):
            await self._set_file_failed(task_id, file_record.file_id, f"校验失败: {verify}")
            await self._set_chunk_failed(task_id, file_record.file_id, chunk_id, f"校验失败: {verify}")
            return

        enable_switch = _coerce_bool(self._cfg("web.import.convert.enable_staging_switch", True), True)
        if not enable_switch:
            await self._set_file_failed(task_id, file_record.file_id, "未启用 staging 切换")
            await self._set_chunk_failed(task_id, file_record.file_id, chunk_id, "未启用 staging 切换")
            return

        await self._set_chunk_state(task_id, file_record.file_id, chunk_id, "writing", "switching", 0.85)
        backup_root = self._resolve_backup_root()
        backup_root.mkdir(parents=True, exist_ok=True)
        backup_dir = backup_root / f"lpmm_convert_{task_id}_{int(_now())}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        switched = False
        rollback_info: Dict[str, Any] = {"attempted": True, "restored": False, "error": ""}
        moved_items: List[Tuple[Path, Path]] = []
        try:
            for name in ("vectors", "graph", "metadata"):
                src_current = target_dir / name
                src_new = staging_dir / name
                if not src_new.exists():
                    raise RuntimeError(f"staging 缺少目录: {src_new}")
                if src_current.exists():
                    dst_backup = backup_dir / name
                    shutil.move(str(src_current), str(dst_backup))
                    moved_items.append((dst_backup, src_current))
                shutil.move(str(src_new), str(src_current))
            switched = True
        except Exception as switch_err:
            rollback_info["error"] = str(switch_err)
            # 尝试回滚
            for src_backup, dst_original in moved_items:
                if src_backup.exists() and not dst_original.exists():
                    try:
                        shutil.move(str(src_backup), str(dst_original))
                    except Exception:
                        pass
            rollback_info["restored"] = True
            async with self._lock:
                t = self._tasks.get(task_id)
                if t:
                    t.rollback_info = rollback_info
            await self._set_file_failed(task_id, file_record.file_id, f"切换失败并回滚: {switch_err}")
            await self._set_chunk_failed(task_id, file_record.file_id, chunk_id, f"switch failed: {switch_err}")
            return

        if switched:
            async with self._lock:
                t = self._tasks.get(task_id)
                if t:
                    t.rollback_info = rollback_info
                    t.artifact_paths["backup_dir"] = str(backup_dir)
            self._cleanup_old_backups()
            try:
                await self._reload_stores_after_external_migration()
            except Exception as reload_err:
                logger.warning(f"转换后重载存储失败: {reload_err}")

        await self._set_chunk_completed(task_id, file_record.file_id, chunk_id)
        async with self._lock:
            t = self._tasks.get(task_id)
            if not t:
                return
            f = self._find_file(t, file_record.file_id)
            if not f:
                return
            f.total_chunks = 1
            f.done_chunks = 1
            f.failed_chunks = 0
            f.cancelled_chunks = 0
            f.progress = 1.0
            f.status = "completed"
            f.current_step = "completed"
            f.updated_at = _now()
            self._recompute_task_progress(t)

    async def _process_temporal_backfill(self, task_id: str, file_record: ImportFileRecord) -> None:
        await self._set_file_strategy(task_id, file_record.file_id, "temporal_backfill")
        await self._set_file_state(task_id, file_record.file_id, "preparing", "backfilling")
        chunk_id = await self._ensure_maibot_migration_chunk(
            task_id,
            file_record.file_id,
            chunk_type="temporal_backfill",
            preview="时序字段回填任务",
        )
        await self._set_chunk_state(task_id, file_record.file_id, chunk_id, "writing", "backfilling", 0.2)

        task = self._tasks.get(task_id)
        if not task:
            await self._set_file_failed(task_id, file_record.file_id, "任务不存在")
            return
        params = dict(task.params)
        target_dir = Path(file_record.source_path or "")
        metadata_dir = target_dir / "metadata"
        if not metadata_dir.exists():
            await self._set_file_failed(task_id, file_record.file_id, f"metadata 目录不存在: {metadata_dir}")
            return

        dry_run = bool(params.get("dry_run"))
        no_created_fallback = bool(params.get("no_created_fallback"))
        limit = max(1, _coerce_int(params.get("limit"), 100000))

        store = MetadataStore(data_dir=metadata_dir)
        updated = 0
        candidates = 0
        try:
            store.connect()
            summary = store.backfill_temporal_metadata_from_created_at(
                limit=limit,
                dry_run=dry_run,
                no_created_fallback=no_created_fallback,
            )
            candidates = int(summary.get("candidates", 0))
            updated = int(summary.get("updated", 0))
        finally:
            try:
                store.close()
            except Exception:
                pass

        async with self._lock:
            t = self._tasks.get(task_id)
            if t:
                t.artifact_paths["temporal_backfill"] = json.dumps(
                    {
                        "target_dir": str(target_dir),
                        "dry_run": dry_run,
                        "no_created_fallback": no_created_fallback,
                        "limit": limit,
                        "candidates": candidates,
                        "updated": updated,
                    },
                    ensure_ascii=False,
                )
        await self._set_chunk_completed(task_id, file_record.file_id, chunk_id)
        async with self._lock:
            t = self._tasks.get(task_id)
            if not t:
                return
            f = self._find_file(t, file_record.file_id)
            if not f:
                return
            f.total_chunks = 1
            f.done_chunks = 1
            f.failed_chunks = 0
            f.cancelled_chunks = 0
            f.progress = 1.0
            f.status = "completed"
            f.current_step = "completed"
            f.updated_at = _now()
            self._recompute_task_progress(t)

