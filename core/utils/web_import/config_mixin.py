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


class ImportConfigMixin:
    """Config, path resolution, and runtime readiness."""
    def set_write_changed_callback(self, callback: Optional[Callable[[Dict[str, Any]], Any]]) -> None:
        self._write_changed_callback = callback

    async def _notify_write_changed(self, payload: Dict[str, Any]) -> None:
        callback = self._write_changed_callback
        if callback is None:
            return
        try:
            maybe_awaitable = callback(payload)
            if asyncio.iscoroutine(maybe_awaitable):
                await maybe_awaitable
        except Exception as e:
            logger.warning(f"写入变更回调执行失败: {e}")

    def _resolve_temp_root(self) -> Path:
        data_dir = resolve_repo_path(self.plugin.get_config("storage.data_dir", "./data"), fallback=default_data_dir())
        return data_dir / "web_import_tmp"

    def _resolve_reports_root(self) -> Path:
        return self._resolve_data_dir() / "web_import_reports"

    def _resolve_manifest_path(self) -> Path:
        return self._resolve_data_dir() / "import_manifest.json"

    def _resolve_staging_root(self) -> Path:
        return self._resolve_data_dir() / "import_staging"

    def _resolve_backup_root(self) -> Path:
        return self._resolve_data_dir() / "import_backup"

    def _resolve_repo_root(self) -> Path:
        return repo_root()

    def _resolve_data_dir(self) -> Path:
        return resolve_repo_path(self.plugin.get_config("storage.data_dir", "./data"), fallback=default_data_dir())

    def _resolve_migration_script(self) -> Path:
        return scripts_root() / "migrate_maibot_memory.py"

    def _default_maibot_source_db(self) -> Path:
        # A_memorix/core/utils -> workspace root
        return self._resolve_repo_root() / "MaiBot" / "data" / "MaiBot.db"

    def _cfg(self, key: str, default: Any) -> Any:
        return self.plugin.get_config(key, default)

    def _cfg_int(self, key: str, default: int) -> int:
        return _coerce_int(self._cfg(key, default), default)

    def _allow_metadata_only_write(self) -> bool:
        return bool(self._cfg("embedding.fallback.allow_metadata_only_write", True))

    def _is_embedding_degraded(self) -> bool:
        checker = getattr(self.plugin, "is_embedding_degraded", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return False
        return False

    def _enqueue_paragraph_backfill(self, paragraph_hash: str, *, error: str = "") -> None:
        if not paragraph_hash:
            return
        enqueue = getattr(self.plugin, "enqueue_paragraph_vector_backfill", None)
        if callable(enqueue):
            try:
                enqueue(paragraph_hash, error=error)
                return
            except Exception as exc:
                logger.warning(f"回填入队失败（runtime facade）: {exc}")
        try:
            self.plugin.metadata_store.enqueue_paragraph_vector_backfill(paragraph_hash, error=error)
        except Exception as exc:
            logger.warning(f"回填入队失败（metadata_store）: {exc}")

    async def _write_paragraph_vector_or_enqueue(
        self,
        *,
        paragraph_hash: str,
        content: str,
        context: str,
    ) -> Dict[str, Any]:
        writer = getattr(self.plugin, "write_paragraph_vector_or_enqueue", None)
        if callable(writer):
            return await writer(paragraph_hash=paragraph_hash, content=content, context=context)

        if self._is_embedding_degraded():
            if not self._allow_metadata_only_write():
                raise RuntimeError("embedding 处于降级态且 metadata-only 写入被禁用")
            self._enqueue_paragraph_backfill(paragraph_hash, error="embedding_degraded")
            return {
                "success": True,
                "vector_written": False,
                "queued": True,
                "warning": "vector_degraded_write",
                "detail": "embedding_degraded",
            }

        try:
            emb = await self.plugin.embedding_manager.encode(content)
            self.plugin.vector_store.add(emb.reshape(1, -1), [paragraph_hash])
            return {
                "success": True,
                "vector_written": True,
                "queued": False,
                "warning": "",
                "detail": "",
            }
        except Exception as exc:
            if not self._allow_metadata_only_write():
                raise
            self._enqueue_paragraph_backfill(paragraph_hash, error=str(exc))
            return {
                "success": True,
                "vector_written": False,
                "queued": True,
                "warning": "vector_degraded_write",
                "detail": str(exc),
            }

    def _is_enabled(self) -> bool:
        return bool(self._cfg("web.import.enabled", True))

    def _queue_limit(self) -> int:
        return max(1, self._cfg_int("web.import.max_queue_size", 20))

    def _max_files_per_task(self) -> int:
        return max(1, self._cfg_int("web.import.max_files_per_task", 200))

    def _max_file_size_bytes(self) -> int:
        mb = max(1, self._cfg_int("web.import.max_file_size_mb", 20))
        return mb * 1024 * 1024

    def _max_paste_chars(self) -> int:
        return max(1000, self._cfg_int("web.import.max_paste_chars", 200000))

    def _default_file_concurrency(self) -> int:
        return max(1, self._cfg_int("web.import.default_file_concurrency", 2))

    def _default_chunk_concurrency(self) -> int:
        return max(1, self._cfg_int("web.import.default_chunk_concurrency", 4))

    def _max_file_concurrency(self) -> int:
        return max(1, self._cfg_int("web.import.max_file_concurrency", 6))

    def _max_chunk_concurrency(self) -> int:
        return max(1, self._cfg_int("web.import.max_chunk_concurrency", 12))

    def _llm_retry_config(self) -> Dict[str, float]:
        retries = max(0, self._cfg_int("web.import.llm_retry.max_attempts", 4))
        min_wait = max(0.1, float(self._cfg("web.import.llm_retry.min_wait_seconds", 3) or 3))
        max_wait = max(min_wait, float(self._cfg("web.import.llm_retry.max_wait_seconds", 40) or 40))
        mult = max(1.0, float(self._cfg("web.import.llm_retry.backoff_multiplier", 3) or 3))
        return {
            "retries": retries,
            "min_wait": min_wait,
            "max_wait": max_wait,
            "multiplier": mult,
        }

    def _default_path_aliases(self) -> Dict[str, str]:
        plugin_dir = Path(__file__).resolve().parents[2]
        repo_root = self._resolve_repo_root()
        return {
            "raw": str((plugin_dir / "data" / "raw").resolve()),
            "lpmm": str((repo_root / "data" / "lpmm_storage").resolve()),
            "plugin_data": str((plugin_dir / "data").resolve()),
        }

    def get_path_aliases(self) -> Dict[str, str]:
        configured = self._cfg("web.import.path_aliases", self._default_path_aliases())
        if not isinstance(configured, dict):
            configured = self._default_path_aliases()

        repo_root = self._resolve_repo_root()
        result: Dict[str, str] = {}
        for alias, raw_path in configured.items():
            key = str(alias or "").strip()
            if not key:
                continue
            text = str(raw_path or "").strip()
            if not text:
                continue
            if text.startswith("\\\\"):
                continue
            p = Path(text)
            if not p.is_absolute():
                p = (repo_root / p).resolve()
            else:
                p = p.resolve()
            result[key] = str(p)

        defaults = self._default_path_aliases()
        for key, path in defaults.items():
            result.setdefault(key, path)
        return result

    def resolve_path_alias(
        self,
        alias: str,
        relative_path: str = "",
        *,
        must_exist: bool = False,
    ) -> Path:
        alias_key = str(alias or "").strip()
        aliases = self.get_path_aliases()
        if alias_key not in aliases:
            raise ValueError(f"未知路径别名: {alias_key}")

        root = Path(aliases[alias_key]).resolve()
        rel = str(relative_path or "").strip().replace("\\", "/")
        if rel.startswith("/") or rel.startswith("\\") or rel.startswith("//"):
            raise ValueError("relative_path 不能为绝对路径")
        if ":" in rel:
            raise ValueError("relative_path 不允许包含盘符")

        candidate = (root / rel).resolve() if rel else root
        try:
            candidate.relative_to(root)
        except ValueError:
            raise ValueError("路径越界：relative_path 超出白名单目录")
        if must_exist and not candidate.exists():
            raise ValueError(f"路径不存在: {candidate}")
        return candidate

    async def resolve_path_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        alias = str(payload.get("alias") or "").strip()
        relative_path = str(payload.get("relative_path") or "").strip()
        must_exist = _coerce_bool(payload.get("must_exist"), True)
        resolved = self.resolve_path_alias(alias, relative_path, must_exist=must_exist)
        return {
            "alias": alias,
            "relative_path": relative_path,
            "resolved_path": str(resolved),
            "exists": resolved.exists(),
            "is_file": resolved.is_file(),
            "is_dir": resolved.is_dir(),
        }

    async def get_runtime_settings(self) -> Dict[str, Any]:
        llm_retry = self._llm_retry_config()
        return {
            "max_queue_size": self._queue_limit(),
            "max_files_per_task": self._max_files_per_task(),
            "max_file_size_mb": self._cfg_int("web.import.max_file_size_mb", 20),
            "max_paste_chars": self._max_paste_chars(),
            "default_file_concurrency": self._default_file_concurrency(),
            "default_chunk_concurrency": self._default_chunk_concurrency(),
            "max_file_concurrency": self._max_file_concurrency(),
            "max_chunk_concurrency": self._max_chunk_concurrency(),
            "poll_interval_ms": max(200, self._cfg_int("web.import.poll_interval_ms", 1000)),
            "maibot_source_db_default": str(self._default_maibot_source_db()),
            "maibot_target_data_dir": str(self._resolve_data_dir()),
            "path_aliases": self.get_path_aliases(),
            "llm_retry": llm_retry,
            "convert_enable_staging_switch": _coerce_bool(
                self._cfg("web.import.convert.enable_staging_switch", True), True
            ),
            "convert_keep_backup_count": max(0, self._cfg_int("web.import.convert.keep_backup_count", 3)),
        }

    def is_write_blocked(self) -> bool:
        task_id = self._active_task_id
        if not task_id:
            return False
        task = self._tasks.get(task_id)
        if not task:
            return False
        return task.status in {"preparing", "running", "cancel_requested"}

    def _ensure_ready(self) -> None:
        required_attrs = ("metadata_store", "vector_store", "graph_store", "embedding_manager")

        def _collect_missing() -> List[str]:
            missing_local: List[str] = []
            for attr in required_attrs:
                if getattr(self.plugin, attr, None) is None:
                    missing_local.append(attr)
            return missing_local

        missing = _collect_missing()
        if missing:
            raise ValueError(f"导入依赖未初始化: {', '.join(missing)}")
        ready_checker = getattr(self.plugin, "is_runtime_ready", None)
        if callable(ready_checker) and not ready_checker():
            raise ValueError("插件运行时未就绪，请先完成 on_enable 初始化")

