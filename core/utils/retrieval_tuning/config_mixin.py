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


class RetrievalTuningConfigMixin:
    """Config access, profile apply/export."""
    def _cfg(self, key: str, default: Any = None) -> Any:
        getter = getattr(self.plugin, "get_config", None)
        if callable(getter):
            return getter(key, default)
        return default

    def _is_enabled(self) -> bool:
        return bool(self._cfg("web.tuning.enabled", True))

    def _queue_limit(self) -> int:
        return _clamp_int(self._cfg("web.tuning.max_queue_size", 8), 8, 1, 100)

    def _poll_interval_s(self) -> float:
        ms = _clamp_int(self._cfg("web.tuning.poll_interval_ms", 1200), 1200, 200, 60000)
        return max(0.2, ms / 1000.0)

    def _llm_retry_cfg(self) -> Dict[str, Any]:
        return {
            "max_attempts": _clamp_int(self._cfg("web.tuning.llm_retry.max_attempts", 3), 3, 1, 10),
            "min_wait_seconds": _clamp_float(self._cfg("web.tuning.llm_retry.min_wait_seconds", 2), 2.0, 0.1, 60.0),
            "max_wait_seconds": _clamp_float(self._cfg("web.tuning.llm_retry.max_wait_seconds", 20), 20.0, 0.2, 120.0),
            "backoff_multiplier": _clamp_float(self._cfg("web.tuning.llm_retry.backoff_multiplier", 2), 2.0, 1.0, 10.0),
        }

    def _eval_query_timeout_s(self) -> float:
        return _clamp_float(
            self._cfg("web.tuning.eval_query_timeout_seconds", 10.0),
            10.0,
            0.01,
            120.0,
        )

    def get_runtime_settings(self) -> Dict[str, Any]:
        intensity = str(self._cfg("web.tuning.default_intensity", "standard") or "standard")
        if intensity not in INTENSITIES:
            intensity = "standard"
        objective = str(self._cfg("web.tuning.default_objective", "precision_priority") or "precision_priority")
        if objective not in OBJECTIVES:
            objective = "precision_priority"
        return {
            "enabled": self._is_enabled(),
            "poll_interval_ms": _clamp_int(self._cfg("web.tuning.poll_interval_ms", 1200), 1200, 200, 60000),
            "max_queue_size": self._queue_limit(),
            "default_objective": objective,
            "default_intensity": intensity,
            "default_rounds": INTENSITIES[intensity],
            "default_top_k_eval": _clamp_int(self._cfg("web.tuning.default_top_k_eval", 20), 20, 5, 100),
            "default_sample_size": _clamp_int(self._cfg("web.tuning.default_sample_size", 24), 24, 4, 200),
            "eval_query_timeout_seconds": self._eval_query_timeout_s(),
            "llm_retry": self._llm_retry_cfg(),
        }

    def _ensure_ready(self) -> None:
        required = ("metadata_store", "vector_store", "graph_store", "embedding_manager")
        missing = [x for x in required if getattr(self.plugin, x, None) is None]
        if missing:
            raise ValueError(f"调优依赖未初始化: {', '.join(missing)}")
        checker = getattr(self.plugin, "is_runtime_ready", None)
        if callable(checker) and not checker():
            raise ValueError("插件运行时未就绪")
        provider = self._import_write_blocked_provider
        if provider is not None and bool(provider()):
            raise ValueError("导入任务运行中，当前禁止启动检索调优")

    def get_profile_snapshot(self) -> Dict[str, Any]:
        cfg = getattr(self.plugin, "config", {}) or {}
        profile = {
            "retrieval": {
                "top_k_paragraphs": _nested_get(cfg, "retrieval.top_k_paragraphs", 20),
                "top_k_relations": _nested_get(cfg, "retrieval.top_k_relations", 10),
                "top_k_final": _nested_get(cfg, "retrieval.top_k_final", 10),
                "alpha": _nested_get(cfg, "retrieval.alpha", 0.5),
                "enable_ppr": _nested_get(cfg, "retrieval.enable_ppr", True),
                "search": {"smart_fallback": {"enabled": _nested_get(cfg, "retrieval.search.smart_fallback.enabled", True)}},
                "sparse": {
                    "enabled": _nested_get(cfg, "retrieval.sparse.enabled", True),
                    "mode": _nested_get(cfg, "retrieval.sparse.mode", "auto"),
                    "candidate_k": _nested_get(cfg, "retrieval.sparse.candidate_k", 80),
                    "relation_candidate_k": _nested_get(cfg, "retrieval.sparse.relation_candidate_k", 60),
                },
                "fusion": {
                    "method": _nested_get(cfg, "retrieval.fusion.method", "weighted_rrf"),
                    "rrf_k": _nested_get(cfg, "retrieval.fusion.rrf_k", 60),
                    "vector_weight": _nested_get(cfg, "retrieval.fusion.vector_weight", 0.7),
                    "bm25_weight": _nested_get(cfg, "retrieval.fusion.bm25_weight", 0.3),
                },
            },
            "threshold": {
                "percentile": _nested_get(cfg, "threshold.percentile", 75.0),
                "min_results": _nested_get(cfg, "threshold.min_results", 3),
            },
        }
        return self._normalize_profile(profile, fallback=profile)

    def _normalize_profile(self, profile: Optional[Dict[str, Any]], *, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raw = copy.deepcopy(profile or {})
        base = copy.deepcopy(fallback or self.get_profile_snapshot())

        def pick(path: str, default: Any) -> Any:
            if _nested_get(raw, path, None) is not None:
                return _nested_get(raw, path, default)
            if path in raw:
                return raw.get(path, default)
            return _nested_get(base, path, default)

        fusion_method = str(pick("retrieval.fusion.method", "weighted_rrf") or "weighted_rrf").strip().lower()
        if fusion_method not in {"weighted_rrf", "alpha_legacy"}:
            fusion_method = "weighted_rrf"

        sparse_mode = str(pick("retrieval.sparse.mode", "auto") or "auto").strip().lower()
        if sparse_mode not in {"auto", "hybrid", "fallback_only"}:
            sparse_mode = "auto"

        vec_w = _clamp_float(pick("retrieval.fusion.vector_weight", 0.7), 0.7, 0.0, 1.0)
        bm_w = _clamp_float(pick("retrieval.fusion.bm25_weight", 0.3), 0.3, 0.0, 1.0)
        s = vec_w + bm_w
        if s <= 1e-9:
            vec_w, bm_w = 0.7, 0.3
        else:
            vec_w, bm_w = vec_w / s, bm_w / s

        return {
            "retrieval": {
                "top_k_paragraphs": _clamp_int(pick("retrieval.top_k_paragraphs", 20), 20, 10, 1200),
                "top_k_relations": _clamp_int(pick("retrieval.top_k_relations", 10), 10, 4, 512),
                "top_k_final": _clamp_int(pick("retrieval.top_k_final", 10), 10, 4, 512),
                "alpha": _clamp_float(pick("retrieval.alpha", 0.5), 0.5, 0.0, 1.0),
                "enable_ppr": _coerce_bool(pick("retrieval.enable_ppr", True), True),
                "search": {"smart_fallback": {"enabled": _coerce_bool(pick("retrieval.search.smart_fallback.enabled", True), True)}},
                "sparse": {
                    "enabled": _coerce_bool(pick("retrieval.sparse.enabled", True), True),
                    "mode": sparse_mode,
                    "candidate_k": _clamp_int(pick("retrieval.sparse.candidate_k", 80), 80, 20, 2000),
                    "relation_candidate_k": _clamp_int(pick("retrieval.sparse.relation_candidate_k", 60), 60, 20, 2000),
                },
                "fusion": {
                    "method": fusion_method,
                    "rrf_k": _clamp_int(pick("retrieval.fusion.rrf_k", 60), 60, 1, 500),
                    "vector_weight": float(vec_w),
                    "bm25_weight": float(bm_w),
                },
            },
            "threshold": {
                "percentile": _clamp_float(pick("threshold.percentile", 75.0), 75.0, 1.0, 99.0),
                "min_results": _clamp_int(pick("threshold.min_results", 3), 3, 1, 100),
            },
        }

    def _apply_profile_to_runtime(self, normalized: Dict[str, Any]) -> None:
        if not isinstance(getattr(self.plugin, "config", None), dict):
            raise RuntimeError("插件 config 不可写")
        for key, value in normalized.items():
            _nested_set(self.plugin.config, key, value)
        plugin_cfg = getattr(self.plugin, "_plugin_config", None)
        if isinstance(plugin_cfg, dict):
            for key, value in normalized.items():
                _nested_set(plugin_cfg, key, value)

    async def apply_profile(self, profile: Dict[str, Any], *, reason: str = "manual") -> Dict[str, Any]:
        normalized = self._normalize_profile(profile)
        current = self.get_profile_snapshot()
        self._rollback_snapshot = current
        self._apply_profile_to_runtime(normalized)
        return {
            "applied": normalized,
            "rollback_snapshot": current,
            "reason": reason,
            "applied_at": _now(),
        }

    async def rollback_profile(self) -> Dict[str, Any]:
        if not self._rollback_snapshot:
            raise ValueError("暂无可回滚的参数快照")
        target = self._normalize_profile(self._rollback_snapshot, fallback=self._rollback_snapshot)
        self._apply_profile_to_runtime(target)
        return {"rolled_back_to": target, "rolled_back_at": _now()}

    def export_toml_snippet(self, profile: Optional[Dict[str, Any]] = None) -> str:
        p = self._normalize_profile(profile or self.get_profile_snapshot())
        r = p["retrieval"]
        t = p["threshold"]
        lines = [
            "[retrieval]",
            f"top_k_paragraphs = {int(r['top_k_paragraphs'])}",
            f"top_k_relations = {int(r['top_k_relations'])}",
            f"top_k_final = {int(r['top_k_final'])}",
            f"alpha = {float(r['alpha']):.4f}",
            f"enable_ppr = {str(bool(r['enable_ppr'])).lower()}",
            "",
            "[retrieval.search.smart_fallback]",
            f"enabled = {str(bool(r['search']['smart_fallback']['enabled'])).lower()}",
            "",
            "[retrieval.sparse]",
            f"enabled = {str(bool(r['sparse']['enabled'])).lower()}",
            f"mode = \"{r['sparse']['mode']}\"",
            f"candidate_k = {int(r['sparse']['candidate_k'])}",
            f"relation_candidate_k = {int(r['sparse']['relation_candidate_k'])}",
            "",
            "[retrieval.fusion]",
            f"method = \"{r['fusion']['method']}\"",
            f"rrf_k = {int(r['fusion']['rrf_k'])}",
            f"vector_weight = {float(r['fusion']['vector_weight']):.4f}",
            f"bm25_weight = {float(r['fusion']['bm25_weight']):.4f}",
            "",
            "[threshold]",
            f"percentile = {float(t['percentile']):.4f}",
            f"min_results = {int(t['min_results'])}",
        ]
        return "\n".join(lines).strip() + "\n"

