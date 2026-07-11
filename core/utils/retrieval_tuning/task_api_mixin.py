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


class RetrievalTuningTaskApiMixin:
    """Task queue public API and worker lifecycle."""
    def _pending_task_count(self) -> int:
        return sum(1 for t in self._tasks.values() if t.status in {"queued", "running", "cancel_requested"})

    async def _ensure_worker(self) -> None:
        async with self._lock:
            if self._worker_task and not self._worker_task.done():
                return
            self._stopping = False
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def shutdown(self) -> None:
        self._stopping = True
        worker = self._worker_task
        if worker is None or worker.done():
            return
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Retrieval tuning worker shutdown failed: {e}")

    async def create_task(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self._is_enabled():
            raise ValueError("检索调优中心已禁用")
        self._ensure_ready()

        data = payload or {}
        objective = str(data.get("objective") or self._cfg("web.tuning.default_objective", "precision_priority"))
        if objective not in OBJECTIVES:
            raise ValueError(f"objective 非法: {objective}")

        intensity = str(data.get("intensity") or self._cfg("web.tuning.default_intensity", "standard"))
        if intensity not in INTENSITIES:
            raise ValueError(f"intensity 非法: {intensity}")

        rounds_total = _clamp_int(data.get("rounds", INTENSITIES[intensity]), INTENSITIES[intensity], 1, 200)
        sample_size = _clamp_int(data.get("sample_size", self._cfg("web.tuning.default_sample_size", 24)), 24, 4, 500)
        top_k_eval = _clamp_int(data.get("top_k_eval", self._cfg("web.tuning.default_top_k_eval", 20)), 20, 5, 100)
        eval_query_timeout_seconds = _clamp_float(
            data.get("eval_query_timeout_seconds", self._eval_query_timeout_s()),
            self._eval_query_timeout_s(),
            0.01,
            120.0,
        )
        llm_enabled = _coerce_bool(data.get("llm_enabled", True), True)
        seed = data.get("seed")
        try:
            seed = int(seed)
        except Exception:
            seed = int(time.time()) % 1000003

        async with self._lock:
            if self._pending_task_count() >= self._queue_limit():
                raise ValueError("调优任务队列已满，请稍后重试")
            task = RetrievalTuningTaskRecord(
                task_id=uuid.uuid4().hex,
                status="queued",
                progress=0.0,
                objective=objective,
                intensity=intensity,
                rounds_total=rounds_total,
                params={
                    "sample_size": sample_size,
                    "top_k_eval": top_k_eval,
                    "eval_query_timeout_seconds": float(eval_query_timeout_seconds),
                    "llm_enabled": llm_enabled,
                    "seed": seed,
                },
            )
            self._tasks[task.task_id] = task
            self._task_order.appendleft(task.task_id)
            self._queue.append(task.task_id)
            task.updated_at = _now()

        await self._ensure_worker()
        return task.to_summary()

    async def list_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        limit = _clamp_int(limit, 50, 1, 500)
        async with self._lock:
            items: List[Dict[str, Any]] = []
            for task_id in list(self._task_order)[:limit]:
                task = self._tasks.get(task_id)
                if task:
                    items.append(task.to_summary())
            return items

    async def get_task(self, task_id: str, include_rounds: bool = False) -> Optional[Dict[str, Any]]:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            return task.to_detail(include_rounds=include_rounds)

    async def get_rounds(self, task_id: str, offset: int = 0, limit: int = 50) -> Optional[Dict[str, Any]]:
        offset = max(0, int(offset))
        limit = _clamp_int(limit, 50, 1, 500)
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            total = len(task.rounds)
            sliced = task.rounds[offset : offset + limit]
            return {
                "total": total,
                "offset": offset,
                "limit": limit,
                "items": [item.to_dict() for item in sliced],
            }

    async def cancel_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            if task.status in {"completed", "failed", "cancelled"}:
                return task.to_summary()
            if task.status == "queued":
                task.status = "cancelled"
                task.cancel_requested = True
                task.finished_at = _now()
                task.updated_at = task.finished_at
                self._queue = deque([x for x in self._queue if x != task_id])
                return task.to_summary()
            task.status = "cancel_requested"
            task.cancel_requested = True
            task.updated_at = _now()
            return task.to_summary()

    async def apply_best(self, task_id: str) -> Dict[str, Any]:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise ValueError("任务不存在")
            if task.status != "completed":
                raise ValueError("任务未完成，无法应用最优参数")
            if not task.best_profile:
                raise ValueError("任务没有可应用的最优参数")
            best = copy.deepcopy(task.best_profile)
        applied = await self.apply_profile(best, reason=f"task:{task_id}:apply_best")
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                task.apply_log.append({"applied_at": _now(), "reason": "apply_best", "profile": best})
                task.updated_at = _now()
        return applied

    async def get_report(self, task_id: str, fmt: str = "md") -> Optional[Dict[str, Any]]:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            artifacts = dict(task.artifact_paths)
        fmt = str(fmt or "md").strip().lower()
        if fmt not in {"md", "json"}:
            fmt = "md"
        path_key = "report_md" if fmt == "md" else "report_json"
        path = artifacts.get(path_key)
        if not path:
            return {"format": fmt, "content": "", "path": ""}
        p = Path(path)
        if not p.exists():
            return {"format": fmt, "content": "", "path": str(p)}
        try:
            content = p.read_text(encoding="utf-8")
        except Exception:
            content = ""
        return {"format": fmt, "content": content, "path": str(p)}

    async def _worker_loop(self) -> None:
        while not self._stopping:
            task_id: Optional[str] = None
            async with self._lock:
                while self._queue:
                    candidate = self._queue.popleft()
                    task = self._tasks.get(candidate)
                    if task is None:
                        continue
                    if task.status != "queued":
                        continue
                    task_id = candidate
                    self._active_task_id = candidate
                    break

            if not task_id:
                await asyncio.sleep(self._poll_interval_s())
                continue

            try:
                await self._run_task(task_id)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Retrieval tuning task crashed: task_id={task_id}, err={e}")
                async with self._lock:
                    task = self._tasks.get(task_id)
                    if task is not None:
                        task.status = "failed"
                        task.error = str(e)
                        task.finished_at = _now()
                        task.updated_at = task.finished_at
            finally:
                async with self._lock:
                    if self._active_task_id == task_id:
                        self._active_task_id = None

    async def _run_task(self, task_id: str) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = "running"
            task.started_at = _now()
            task.updated_at = task.started_at

        artifacts_dir = self._artifacts_root / task_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        query_set_path = artifacts_dir / "query_set.json"
        rounds_path = artifacts_dir / "round_metrics.jsonl"
        best_profile_path = artifacts_dir / "best_profile.json"
        report_json_path = artifacts_dir / "report.json"
        report_md_path = artifacts_dir / "report.md"

        try:
            params = dict(task.params)
            cases, stats = await self._build_query_set(
                sample_size=int(params["sample_size"]),
                seed=int(params["seed"]),
                llm_enabled=bool(params.get("llm_enabled", True)),
            )
            if not cases:
                raise ValueError("当前知识库样本不足，无法构建调优测试集")

            query_set_path.write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "created_at": _now(),
                        "stats": stats,
                        "items": [c.to_dict() for c in cases],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            baseline_profile = self.get_profile_snapshot()
            top_k_eval = int(params["top_k_eval"])
            baseline_eval = await self._evaluate_profile(
                profile=baseline_profile,
                cases=cases,
                objective=task.objective,
                top_k_eval=top_k_eval,
                query_timeout_s=float(params.get("eval_query_timeout_seconds") or self._eval_query_timeout_s()),
            )
            baseline_round = RetrievalTuningRoundRecord(
                round_index=0,
                candidate_profile=baseline_profile,
                metrics=baseline_eval["metrics"],
                score=float(baseline_eval["score"]),
                latency_ms=float(baseline_eval["avg_elapsed_ms"]),
                failure_summary=baseline_eval["failure_summary"],
            )
            rounds_path.write_text(json.dumps(baseline_round.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")

            async with self._lock:
                task = self._tasks.get(task_id)
                if task is None:
                    return
                task.query_set_stats = stats
                task.baseline_profile = copy.deepcopy(baseline_profile)
                task.baseline_metrics = copy.deepcopy(baseline_eval["metrics"])
                task.rounds.append(baseline_round)
                task.best_profile = copy.deepcopy(baseline_profile)
                task.best_metrics = copy.deepcopy(baseline_eval["metrics"])
                task.best_score = float(baseline_eval["score"])
                task.progress = 0.0
                task.updated_at = _now()

            best_profile = copy.deepcopy(baseline_profile)
            best_metrics = copy.deepcopy(baseline_eval["metrics"])
            best_failure_summary = copy.deepcopy(baseline_eval["failure_summary"])
            best_score = float(baseline_eval["score"])
            llm_suggestions: List[Dict[str, Any]] = []
            task_cancelled = False

            for round_idx in range(1, int(task.rounds_total) + 1):
                async with self._lock:
                    task = self._tasks.get(task_id)
                    if task is None:
                        return
                    if task.cancel_requested or task.status == "cancel_requested":
                        task.status = "cancelled"
                        task.finished_at = _now()
                        task.updated_at = task.finished_at
                        task_cancelled = True
                        break

                if round_idx == 1 or (round_idx % 5 == 0 and not llm_suggestions):
                    llm_suggestions = await self._suggest_profiles_with_llm(
                        base_profile=best_profile,
                        failure_summary=best_failure_summary,
                        objective=task.objective,
                        max_count=3,
                        enabled=bool(params.get("llm_enabled", True)),
                    )

                candidate_profile = self._generate_candidate_profile(
                    task_id=task_id,
                    round_index=round_idx,
                    objective=task.objective,
                    baseline_profile=baseline_profile,
                    best_profile=best_profile,
                    llm_suggestions=llm_suggestions,
                )
                eval_cases = self._select_round_eval_cases(
                    cases=cases,
                    intensity=task.intensity,
                    round_index=round_idx,
                    seed=int(params.get("seed", 0)),
                )
                eval_result = await self._evaluate_profile(
                    profile=candidate_profile,
                    cases=eval_cases,
                    objective=task.objective,
                    top_k_eval=top_k_eval,
                    query_timeout_s=float(params.get("eval_query_timeout_seconds") or self._eval_query_timeout_s()),
                )
                round_record = RetrievalTuningRoundRecord(
                    round_index=round_idx,
                    candidate_profile=candidate_profile,
                    metrics=eval_result["metrics"],
                    score=float(eval_result["score"]),
                    latency_ms=float(eval_result["avg_elapsed_ms"]),
                    failure_summary=eval_result["failure_summary"],
                )
                with rounds_path.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(round_record.to_dict(), ensure_ascii=False) + "\n")

                if float(eval_result["score"]) > float(best_score):
                    best_score = float(eval_result["score"])
                    best_profile = copy.deepcopy(candidate_profile)
                    best_metrics = copy.deepcopy(eval_result["metrics"])
                    best_failure_summary = copy.deepcopy(eval_result["failure_summary"])

                async with self._lock:
                    task = self._tasks.get(task_id)
                    if task is None:
                        return
                    task.rounds_done = round_idx
                    task.rounds.append(round_record)
                    task.best_profile = copy.deepcopy(best_profile)
                    task.best_metrics = copy.deepcopy(best_metrics)
                    task.best_score = float(best_score)
                    task.progress = min(1.0, float(round_idx) / float(task.rounds_total))
                    task.updated_at = _now()

            if best_profile and (not task_cancelled):
                # 候选轮可能基于子样本评估，收官时用全量样本复核，确保最终指标可解释。
                best_full = await self._evaluate_profile(
                    profile=best_profile,
                    cases=cases,
                    objective=task.objective,
                    top_k_eval=top_k_eval,
                    query_timeout_s=float(params.get("eval_query_timeout_seconds") or self._eval_query_timeout_s()),
                )
                best_profile = copy.deepcopy(best_profile)
                best_metrics = copy.deepcopy(best_full["metrics"])
                best_failure_summary = copy.deepcopy(best_full["failure_summary"])
                best_score = float(best_full["score"])
                if best_score < float(baseline_eval["score"]):
                    best_profile = copy.deepcopy(baseline_profile)
                    best_metrics = copy.deepcopy(baseline_eval["metrics"])
                    best_failure_summary = copy.deepcopy(baseline_eval["failure_summary"])
                    best_score = float(baseline_eval["score"])

                async with self._lock:
                    task = self._tasks.get(task_id)
                    if task is not None:
                        task.best_profile = copy.deepcopy(best_profile)
                        task.best_metrics = copy.deepcopy(best_metrics)
                        task.best_score = float(best_score)
                        task.updated_at = _now()

            async with self._lock:
                task = self._tasks.get(task_id)
                if task is None:
                    return
                if task.status not in {"cancelled", "failed"}:
                    task.status = "completed"
                    task.progress = 1.0
                    task.finished_at = _now()
                    task.updated_at = task.finished_at
                final_task = copy.deepcopy(task)

            if final_task.status == "completed":
                best_profile_path.write_text(json.dumps(final_task.best_profile, ensure_ascii=False, indent=2), encoding="utf-8")
                report_payload = self._build_report_payload(final_task)
                report_json_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                report_md_path.write_text(self._build_report_markdown(final_task, report_payload), encoding="utf-8")

            async with self._lock:
                task = self._tasks.get(task_id)
                if task is not None:
                    task.artifact_paths = {
                        "query_set": str(query_set_path),
                        "round_metrics_jsonl": str(rounds_path),
                        "best_profile": str(best_profile_path),
                        "report_json": str(report_json_path),
                        "report_md": str(report_md_path),
                    }
                    task.updated_at = _now()
        except Exception as e:
            logger.error(f"Retrieval tuning task failed: task_id={task_id}, err={e}")
            async with self._lock:
                task = self._tasks.get(task_id)
                if task is not None:
                    task.status = "failed"
                    task.error = str(e)
                    task.finished_at = _now()
                    task.updated_at = task.finished_at

