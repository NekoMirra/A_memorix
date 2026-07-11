"""Retrieval tuning shared constants, helpers, and record types."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


OBJECTIVES = {"precision_priority", "balanced", "recall_priority"}
INTENSITIES = {"quick": 8, "standard": 20, "deep": 32}
CATEGORIES = {"query_nl", "query_kw", "spo_relation", "spo_search"}
_RUNTIME_CONFIG_INSTANCE_KEYS = {
    "vector_store",
    "graph_store",
    "metadata_store",
    "embedding_manager",
    "sparse_index",
    "relation_write_service",
    "plugin_instance",
}


def _now() -> float:
    return time.time()


def _clamp_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    return max(min_value, min(max_value, parsed))


def _clamp_float(value: Any, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    return max(min_value, min(max_value, parsed))


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _nested_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = data
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _nested_set(data: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = data
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _safe_json_loads(text: str) -> Optional[Any]:
    raw = str(text or "").strip()
    if not raw:
        return None
    if "```" in raw:
        raw = raw.replace("```json", "```")
        for seg in raw.split("```"):
            seg = seg.strip()
            if seg.startswith("{") or seg.startswith("["):
                raw = seg
                break
    try:
        return json.loads(raw)
    except Exception:
        pass
    s = raw.find("{")
    e = raw.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(raw[s : e + 1])
        except Exception:
            return None
    return None


@dataclass
class RetrievalQueryCase:
    case_id: str
    category: str
    query: str
    expected_hashes: List[str] = field(default_factory=list)
    expected_spo: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "query": self.query,
            "expected_hashes": list(self.expected_hashes),
            "expected_spo": dict(self.expected_spo),
            "metadata": dict(self.metadata),
        }


@dataclass
class RetrievalTuningRoundRecord:
    round_index: int
    candidate_profile: Dict[str, Any]
    metrics: Dict[str, Any]
    score: float
    latency_ms: float
    failure_summary: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_index": self.round_index,
            "candidate_profile": copy.deepcopy(self.candidate_profile),
            "metrics": copy.deepcopy(self.metrics),
            "score": float(self.score),
            "latency_ms": float(self.latency_ms),
            "failure_summary": copy.deepcopy(self.failure_summary),
            "created_at": float(self.created_at),
        }


@dataclass
class RetrievalTuningTaskRecord:
    task_id: str
    status: str
    progress: float
    objective: str
    intensity: str
    rounds_total: int
    rounds_done: int = 0
    best_profile: Dict[str, Any] = field(default_factory=dict)
    best_metrics: Dict[str, Any] = field(default_factory=dict)
    best_score: float = -1.0
    baseline_profile: Dict[str, Any] = field(default_factory=dict)
    baseline_metrics: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    query_set_stats: Dict[str, Any] = field(default_factory=dict)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    rounds: List[RetrievalTuningRoundRecord] = field(default_factory=list)
    cancel_requested: bool = False
    created_at: float = field(default_factory=_now)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    updated_at: float = field(default_factory=_now)
    apply_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "objective": self.objective,
            "intensity": self.intensity,
            "rounds_total": self.rounds_total,
            "rounds_done": self.rounds_done,
            "best_score": self.best_score,
            "error": self.error,
            "query_set_stats": dict(self.query_set_stats),
            "artifact_paths": dict(self.artifact_paths),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "updated_at": self.updated_at,
        }

    def to_detail(self, include_rounds: bool = False) -> Dict[str, Any]:
        payload = self.to_summary()
        payload.update(
            {
                "params": copy.deepcopy(self.params),
                "best_profile": copy.deepcopy(self.best_profile),
                "best_metrics": copy.deepcopy(self.best_metrics),
                "baseline_profile": copy.deepcopy(self.baseline_profile),
                "baseline_metrics": copy.deepcopy(self.baseline_metrics),
                "apply_log": copy.deepcopy(self.apply_log),
            }
        )
        if include_rounds:
            payload["rounds"] = [x.to_dict() for x in self.rounds]
        return payload
