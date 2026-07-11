"""Web import shared constants, helpers, and record types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os
import time
import uuid

from ..import_payloads import (
    normalize_entity_import_item,
    normalize_relation_import_item,
)
from ...storage import KnowledgeType
from ...strategies.base import KnowledgeType as StrategyKnowledgeType


TASK_STATUS = {
    "queued",
    "preparing",
    "running",
    "cancel_requested",
    "cancelled",
    "completed",
    "completed_with_errors",
    "failed",
}

FILE_STATUS = {
    "queued",
    "preparing",
    "splitting",
    "extracting",
    "writing",
    "saving",
    "completed",
    "failed",
    "cancelled",
}

CHUNK_STATUS = {
    "queued",
    "extracting",
    "writing",
    "completed",
    "failed",
    "cancelled",
}

FILE_WARNING_KEEP_LIMIT = 50


def _now() -> float:
    return time.time()


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


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
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value or "").replace("\r", "\n")
        raw_items = []
        for seg in text.split("\n"):
            raw_items.extend(seg.split(","))

    out: List[str] = []
    seen = set()
    for item in raw_items:
        v = str(item or "").strip()
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _coerce_import_data_dict(value: Any, *, context: str) -> Dict[str, Any]:
    """确保 LLM 抽取结果是对象，避免写入阶段出现部分提交。"""

    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise ValueError(f"{context} 必须返回 JSON 对象，当前类型: {type(value).__name__}")


def _normalize_import_relation_list(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    relations: List[Dict[str, str]] = []
    for item in value:
        relation = normalize_relation_import_item(item)
        if relation is not None:
            relations.append(relation)
    return relations


def _normalize_import_entity_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    entities: List[str] = []
    seen = set()
    for item in value:
        name = normalize_entity_import_item(item)
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(name)
    return entities


def _parse_optional_positive_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        parsed = int(text)
    except Exception:
        raise ValueError(f"{field_name} 必须为整数")
    if parsed <= 0:
        raise ValueError(f"{field_name} 必须 > 0")
    return parsed


def _safe_filename(name: str) -> str:
    base = os.path.basename(str(name or "").strip())
    if not base:
        return f"unnamed_{uuid.uuid4().hex[:8]}.txt"
    return base


def _storage_type_from_strategy(strategy_type: StrategyKnowledgeType) -> str:
    if strategy_type == StrategyKnowledgeType.NARRATIVE:
        return KnowledgeType.NARRATIVE.value
    if strategy_type == StrategyKnowledgeType.FACTUAL:
        return KnowledgeType.FACTUAL.value
    if strategy_type == StrategyKnowledgeType.QUOTE:
        return KnowledgeType.QUOTE.value
    return KnowledgeType.MIXED.value


@dataclass
class ImportChunkRecord:
    chunk_id: str
    index: int
    chunk_type: str
    status: str = "queued"
    step: str = "queued"
    failed_at: str = ""
    retryable: bool = False
    error: str = ""
    progress: float = 0.0
    content_preview: str = ""
    updated_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "index": self.index,
            "chunk_type": self.chunk_type,
            "status": self.status,
            "step": self.step,
            "failed_at": self.failed_at,
            "retryable": self.retryable,
            "error": self.error,
            "progress": self.progress,
            "content_preview": self.content_preview,
            "updated_at": self.updated_at,
        }


@dataclass
class ImportFileRecord:
    file_id: str
    name: str
    source_kind: str
    input_mode: str
    status: str = "queued"
    current_step: str = "queued"
    detected_strategy_type: str = "unknown"
    total_chunks: int = 0
    done_chunks: int = 0
    failed_chunks: int = 0
    cancelled_chunks: int = 0
    progress: float = 0.0
    error: str = ""
    chunks: List[ImportChunkRecord] = field(default_factory=list)
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    temp_path: Optional[str] = None
    source_path: Optional[str] = None
    inline_content: Optional[str] = None
    content_hash: str = ""
    retry_chunk_indexes: List[int] = field(default_factory=list)
    retry_mode: str = ""
    warning_count: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self, include_chunks: bool = False) -> Dict[str, Any]:
        payload = {
            "file_id": self.file_id,
            "name": self.name,
            "source_kind": self.source_kind,
            "input_mode": self.input_mode,
            "status": self.status,
            "current_step": self.current_step,
            "detected_strategy_type": self.detected_strategy_type,
            "total_chunks": self.total_chunks,
            "done_chunks": self.done_chunks,
            "failed_chunks": self.failed_chunks,
            "cancelled_chunks": self.cancelled_chunks,
            "progress": self.progress,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_path": self.source_path or "",
            "content_hash": self.content_hash or "",
            "retry_chunk_indexes": list(self.retry_chunk_indexes or []),
            "retry_mode": self.retry_mode or "",
            "warning_count": int(self.warning_count),
            "warnings": list(self.warnings),
        }
        if include_chunks:
            payload["chunks"] = [chunk.to_dict() for chunk in self.chunks]
        return payload


@dataclass
class ImportTaskRecord:
    task_id: str
    source: str
    params: Dict[str, Any]
    status: str = "queued"
    current_step: str = "queued"
    total_chunks: int = 0
    done_chunks: int = 0
    failed_chunks: int = 0
    cancelled_chunks: int = 0
    progress: float = 0.0
    error: str = ""
    files: List[ImportFileRecord] = field(default_factory=list)
    created_at: float = field(default_factory=_now)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    updated_at: float = field(default_factory=_now)
    schema_detected: str = ""
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    rollback_info: Dict[str, Any] = field(default_factory=dict)
    retry_parent_task_id: str = ""
    retry_summary: Dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source": self.source,
            "status": self.status,
            "current_step": self.current_step,
            "total_chunks": self.total_chunks,
            "done_chunks": self.done_chunks,
            "failed_chunks": self.failed_chunks,
            "cancelled_chunks": self.cancelled_chunks,
            "progress": self.progress,
            "error": self.error,
            "file_count": len(self.files),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "updated_at": self.updated_at,
            "task_kind": str(self.params.get("task_kind") or self.source),
            "schema_detected": self.schema_detected,
            "artifact_paths": dict(self.artifact_paths),
            "rollback_info": dict(self.rollback_info),
            "retry_parent_task_id": self.retry_parent_task_id or "",
            "retry_summary": dict(self.retry_summary),
        }

    def to_detail(self, include_chunks: bool = False) -> Dict[str, Any]:
        payload = self.to_summary()
        payload["params"] = self.params
        payload["files"] = [f.to_dict(include_chunks=include_chunks) for f in self.files]
        return payload
