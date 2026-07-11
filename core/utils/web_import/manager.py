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


from .config_mixin import ImportConfigMixin
from .file_process_mixin import ImportFileProcessMixin
from .manifest_mixin import ImportManifestMixin
from .migration_mixin import ImportMigrationMixin
from .params_mixin import ImportParamsMixin
from .state_mixin import ImportStateMixin
from .task_api_mixin import ImportTaskApiMixin
from .worker_mixin import ImportWorkerMixin


class ImportTaskManager(
    ImportTaskApiMixin,
    ImportWorkerMixin,
    ImportMigrationMixin,
    ImportFileProcessMixin,
    ImportParamsMixin,
    ImportManifestMixin,
    ImportStateMixin,
    ImportConfigMixin,
):
    """Web 导入任务管理器（拆分后的聚合类）。"""

    def __init__(self, plugin: Any):
        self.plugin = plugin
        self._lock = asyncio.Lock()
        self._storage_lock = asyncio.Lock()

        self._tasks: Dict[str, ImportTaskRecord] = {}
        self._task_order: deque[str] = deque()
        self._queue: deque[str] = deque()
        self._active_task_id: Optional[str] = None

        self._worker_task: Optional[asyncio.Task] = None
        self._stopping = False

        self._temp_root = self._resolve_temp_root()
        self._temp_root.mkdir(parents=True, exist_ok=True)
        self._reports_root = self._resolve_reports_root()
        self._reports_root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._resolve_manifest_path()
        self._manifest_cache: Optional[Dict[str, Any]] = None
        self._write_changed_callback: Optional[Callable[[Dict[str, Any]], Any]] = None
