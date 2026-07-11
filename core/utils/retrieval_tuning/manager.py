from __future__ import annotations

import asyncio
import copy
import json
import random
import re
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.common.logger import get_logger

from ....paths import artifacts_root
from ...runtime.search_runtime_initializer import build_search_runtime
from ..search_execution_service import SearchExecutionRequest, SearchExecutionService
from .helpers import (
    CATEGORIES,
    INTENSITIES,
    OBJECTIVES,
    RetrievalQueryCase,
    RetrievalTuningRoundRecord,
    RetrievalTuningTaskRecord,
    _RUNTIME_CONFIG_INSTANCE_KEYS,
    _clamp_float,
    _clamp_int,
    _coerce_bool,
    _deep_merge,
    _nested_get,
    _nested_set,
    _now,
    _safe_json_loads,
)

try:
    from src.services import llm_service as llm_api
except Exception:  # pragma: no cover
    llm_api = None

logger = get_logger("A_Memorix.RetrievalTuningManager")


from .config_mixin import RetrievalTuningConfigMixin
from .eval_mixin import RetrievalTuningEvalMixin
from .query_mixin import RetrievalTuningQueryMixin
from .task_api_mixin import RetrievalTuningTaskApiMixin


class RetrievalTuningManager(
    RetrievalTuningTaskApiMixin,
    RetrievalTuningQueryMixin,
    RetrievalTuningEvalMixin,
    RetrievalTuningConfigMixin,
):
    """Retrieval tuning manager for WebUI（拆分后的聚合类）。"""

    def __init__(
        self,
        plugin: Any,
        *,
        import_write_blocked_provider: Optional[Callable[[], bool]] = None,
    ):
        self.plugin = plugin
        self._import_write_blocked_provider = import_write_blocked_provider

        self._lock = asyncio.Lock()
        self._tasks: Dict[str, RetrievalTuningTaskRecord] = {}
        self._task_order: deque[str] = deque()
        self._queue: deque[str] = deque()
        self._active_task_id: Optional[str] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._stopping = False

        self._rollback_snapshot: Optional[Dict[str, Any]] = None

        self._artifacts_root = artifacts_root() / "retrieval_tuning"
        self._artifacts_root.mkdir(parents=True, exist_ok=True)
