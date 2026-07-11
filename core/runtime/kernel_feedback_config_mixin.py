from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Sequence
from json_repair import repair_json
from src.common.logger import get_logger
from src.config.config import global_config
from src.services import message_service as message_api
from src.services.llm_service import LLMServiceClient
from ..utils.hash import compute_hash, normalize_text
from ..utils.person_profile_service import PersonProfileService

logger = get_logger("A_Memorix.SDKMemoryKernel")


class KernelFeedbackConfigMixin:
    def _coerce_datetime(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value))
            except Exception:
                return None
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text)
        except Exception:
            return None

    def _feedback_signal_tokens() -> tuple[str, ...]:
        return (
            "不对",
            "错了",
            "你记错",
            "记错了",
            "不是",
            "并不是",
            "纠正",
            "更正",
            "改成",
            "应该是",
            "实际是",
            "说反了",
        )

    def _feedback_contains_signal(cls, text: str) -> bool:
        content = str(text or "").strip().lower()
        if not content:
            return False
        return any(token in content for token in cls._feedback_signal_tokens())

    def _feedback_noise(text: str) -> bool:
        content = str(text or "").strip()
        if not content:
            return True
        if KernelFeedbackConfigMixin._feedback_contains_signal(content):
            return False
        if len(content) <= 2:
            return True
        markers = (
            "哈哈",
            "好的",
            "收到",
            "谢谢",
            "嗯嗯",
            "晚安",
            "早安",
            "拜拜",
            "在吗",
        )
        return len(content) <= 8 and any(marker in content for marker in markers)

    def _safe_json_loads(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        text = str(raw or "").strip()
        if not text:
            return {}
        try:
            repaired = repair_json(text)
            payload = json.loads(repaired) if isinstance(repaired, str) else repaired
        except Exception:
            payload = None
        return payload if isinstance(payload, dict) else {}

    def _feedback_cfg_enabled() -> bool:
        memory_cfg = global_config.a_memorix.integration
        return bool(getattr(memory_cfg, "feedback_correction_enabled", False))

    def _feedback_cfg_window_hours() -> float:
        memory_cfg = global_config.a_memorix.integration
        return max(0.1, float(getattr(memory_cfg, "feedback_correction_window_hours", 12.0) or 12.0))

    def _feedback_cfg_check_interval_seconds() -> float:
        memory_cfg = global_config.a_memorix.integration
        minutes = max(1, int(getattr(memory_cfg, "feedback_correction_check_interval_minutes", 30) or 30))
        return float(minutes) * 60.0

    def _feedback_cfg_batch_size() -> int:
        memory_cfg = global_config.a_memorix.integration
        return max(1, int(getattr(memory_cfg, "feedback_correction_batch_size", 20) or 20))

    def _feedback_cfg_auto_apply_threshold() -> float:
        memory_cfg = global_config.a_memorix.integration
        value = float(getattr(memory_cfg, "feedback_correction_auto_apply_threshold", 0.85) or 0.85)
        return min(1.0, max(0.0, value))

    def _feedback_cfg_max_messages() -> int:
        memory_cfg = global_config.a_memorix.integration
        return max(1, int(getattr(memory_cfg, "feedback_correction_max_feedback_messages", 30) or 30))

    def _feedback_cfg_prefilter_enabled() -> bool:
        memory_cfg = global_config.a_memorix.integration
        return bool(getattr(memory_cfg, "feedback_correction_prefilter_enabled", True))

    def _feedback_cfg_paragraph_mark_enabled() -> bool:
        memory_cfg = global_config.a_memorix.integration
        return bool(getattr(memory_cfg, "feedback_correction_paragraph_mark_enabled", True))

    def _feedback_cfg_paragraph_hard_filter_enabled() -> bool:
        memory_cfg = global_config.a_memorix.integration
        return bool(getattr(memory_cfg, "feedback_correction_paragraph_hard_filter_enabled", True))

    def _feedback_cfg_profile_refresh_enabled() -> bool:
        memory_cfg = global_config.a_memorix.integration
        return bool(getattr(memory_cfg, "feedback_correction_profile_refresh_enabled", True))

    def _feedback_cfg_profile_force_refresh_on_read() -> bool:
        memory_cfg = global_config.a_memorix.integration
        return bool(getattr(memory_cfg, "feedback_correction_profile_force_refresh_on_read", True))

    def _feedback_cfg_episode_rebuild_enabled() -> bool:
        memory_cfg = global_config.a_memorix.integration
        return bool(getattr(memory_cfg, "feedback_correction_episode_rebuild_enabled", True))

    def _feedback_cfg_episode_query_block_enabled() -> bool:
        memory_cfg = global_config.a_memorix.integration
        return bool(getattr(memory_cfg, "feedback_correction_episode_query_block_enabled", True))

    def _feedback_cfg_reconcile_interval_seconds() -> float:
        memory_cfg = global_config.a_memorix.integration
        minutes = max(1, int(getattr(memory_cfg, "feedback_correction_reconcile_interval_minutes", 5) or 5))
        return float(minutes) * 60.0

    def _feedback_cfg_reconcile_batch_size() -> int:
        memory_cfg = global_config.a_memorix.integration
        return max(1, int(getattr(memory_cfg, "feedback_correction_reconcile_batch_size", 20) or 20))

    def _feedback_cfg_window_label(cls) -> str:
        hours = cls._feedback_cfg_window_hours()
        if abs(hours - round(hours)) < 1e-9:
            return f"{int(round(hours))}h"
        return f"{hours:.2f}h"
