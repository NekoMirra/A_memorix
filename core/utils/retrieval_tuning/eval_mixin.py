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


class RetrievalTuningEvalMixin:
    """Candidate profile generation, evaluation, and reporting."""
    async def _suggest_profiles_with_llm(
        self,
        *,
        base_profile: Dict[str, Any],
        failure_summary: Dict[str, Any],
        objective: str,
        max_count: int,
        enabled: bool,
    ) -> List[Dict[str, Any]]:
        if not enabled or llm_api is None or max_count <= 0:
            return []
        prompt = (
            "你是检索调参专家。"
            "请基于基础参数与失败摘要，给出最多 "
            f"{int(max_count)} 组候选参数，返回 JSON: {{\"profiles\": [ ... ]}}。\n"
            "字段仅可包含：retrieval.top_k_paragraphs, retrieval.top_k_relations, retrieval.top_k_final, "
            "retrieval.alpha, retrieval.enable_ppr, retrieval.search.smart_fallback.enabled, "
            "retrieval.sparse.enabled, retrieval.sparse.mode, retrieval.sparse.candidate_k, retrieval.sparse.relation_candidate_k, "
            "retrieval.fusion.method, retrieval.fusion.rrf_k, retrieval.fusion.vector_weight, retrieval.fusion.bm25_weight, "
            "threshold.percentile, threshold.min_results。\n"
            f"objective={objective}\n"
            f"base={json.dumps(base_profile, ensure_ascii=False)}\n"
            f"failure_summary={json.dumps(failure_summary, ensure_ascii=False)}"
        )
        try:
            raw = await self._llm_call_text(prompt, request_type="A_Memorix.RetrievalTuning.ProfileSuggest")
            obj = _safe_json_loads(raw)
            if not isinstance(obj, dict):
                return []
            profiles = obj.get("profiles")
            if not isinstance(profiles, list):
                return []
            out = []
            for item in profiles[:max_count]:
                if isinstance(item, dict):
                    out.append(self._normalize_profile(item, fallback=base_profile))
            return out
        except Exception:
            return []

    def _generate_candidate_profile(
        self,
        *,
        task_id: str,
        round_index: int,
        objective: str,
        baseline_profile: Dict[str, Any],
        best_profile: Dict[str, Any],
        llm_suggestions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if llm_suggestions:
            return self._normalize_profile(llm_suggestions.pop(0), fallback=best_profile)

        rng = random.Random(f"{task_id}:{round_index}")
        base = baseline_profile if round_index % 4 == 1 else best_profile
        candidate = copy.deepcopy(base)

        if objective == "precision_priority":
            para_choices = [40, 80, 120, 180, 240, 320]
            rel_choices = [4, 8, 12, 16, 24]
            final_choices = [4, 8, 12, 16, 20, 32, 48, 64]
            alpha_choices = [0.0, 0.35, 0.50, 0.62, 0.72, 0.82, 0.90]
            pct_choices = [55, 60, 65, 72, 80]
            min_results_choices = [1, 2]
        elif objective == "recall_priority":
            para_choices = [120, 220, 300, 420, 560, 720]
            rel_choices = [8, 12, 16, 24, 32]
            final_choices = [8, 16, 32, 48, 64, 96, 128]
            alpha_choices = [0.20, 0.35, 0.45, 0.55, 0.65, 0.75]
            pct_choices = [40, 48, 55, 62]
            min_results_choices = [1, 2, 3]
        else:
            para_choices = [80, 160, 240, 320, 420, 520]
            rel_choices = [6, 10, 14, 18, 24, 30]
            final_choices = [6, 12, 20, 32, 48, 64, 80]
            alpha_choices = [0.25, 0.45, 0.55, 0.65, 0.75, 0.85]
            pct_choices = [48, 55, 62, 70]
            min_results_choices = [1, 2, 3]

        _nested_set(candidate, "retrieval.top_k_paragraphs", rng.choice(para_choices))
        _nested_set(candidate, "retrieval.top_k_relations", rng.choice(rel_choices))
        _nested_set(candidate, "retrieval.top_k_final", rng.choice(final_choices))
        _nested_set(candidate, "retrieval.alpha", rng.choice(alpha_choices))
        # PPR 在 TestClient/异步评估场景下存在偶发长时阻塞风险，调优评估链路固定关闭。
        _nested_set(candidate, "retrieval.enable_ppr", False)
        _nested_set(candidate, "retrieval.search.smart_fallback.enabled", bool(rng.choice([True, True, False])))
        _nested_set(candidate, "retrieval.sparse.enabled", bool(rng.choice([True, True, False])))
        _nested_set(candidate, "retrieval.sparse.mode", rng.choice(["auto", "hybrid", "fallback_only"]))
        _nested_set(candidate, "retrieval.sparse.candidate_k", rng.choice([60, 80, 120, 160, 220, 320]))
        _nested_set(candidate, "retrieval.sparse.relation_candidate_k", rng.choice([40, 60, 90, 120, 180, 260]))
        _nested_set(candidate, "retrieval.fusion.method", rng.choice(["weighted_rrf", "weighted_rrf", "alpha_legacy"]))
        _nested_set(candidate, "retrieval.fusion.rrf_k", rng.choice([30, 45, 60, 75, 90]))
        vec_w = float(rng.choice([0.55, 0.65, 0.72, 0.80, 0.88]))
        _nested_set(candidate, "retrieval.fusion.vector_weight", vec_w)
        _nested_set(candidate, "retrieval.fusion.bm25_weight", 1.0 - vec_w)
        _nested_set(candidate, "threshold.percentile", rng.choice(pct_choices))
        _nested_set(candidate, "threshold.min_results", rng.choice(min_results_choices))

        return self._normalize_profile(candidate, fallback=base)

    def _build_runtime_config(self, normalized_profile: Dict[str, Any]) -> Dict[str, Any]:
        raw_base = getattr(self.plugin, "config", {}) or {}
        if isinstance(raw_base, dict):
            base = {
                key: value
                for key, value in raw_base.items()
                if key not in _RUNTIME_CONFIG_INSTANCE_KEYS
            }
        else:
            base = {}
        merged = _deep_merge(base, normalized_profile)
        # 调优评估场景优先稳定性，避免并发访问共享 SQLite/Faiss 导致长时阻塞。
        _nested_set(merged, "retrieval.enable_parallel", False)
        # 调优评估阶段关闭 PPR，规避 PageRank 线程计算偶发阻塞导致整轮卡死。
        _nested_set(merged, "retrieval.enable_ppr", False)
        merged["vector_store"] = getattr(self.plugin, "vector_store", None)
        merged["graph_store"] = getattr(self.plugin, "graph_store", None)
        merged["metadata_store"] = getattr(self.plugin, "metadata_store", None)
        merged["embedding_manager"] = getattr(self.plugin, "embedding_manager", None)
        merged["sparse_index"] = getattr(self.plugin, "sparse_index", None)
        merged["plugin_instance"] = self.plugin
        return merged

    async def _evaluate_profile(
        self,
        *,
        profile: Dict[str, Any],
        cases: List[RetrievalQueryCase],
        objective: str,
        top_k_eval: int,
        query_timeout_s: float,
    ) -> Dict[str, Any]:
        normalized = self._normalize_profile(profile)
        eval_top_k = _clamp_int(top_k_eval, 20, 1, 1000)
        # 评估时让 top_k_final 参与有效召回深度，避免该参数对评分无影响。
        request_top_k = min(
            int(eval_top_k),
            _clamp_int(_nested_get(normalized, "retrieval.top_k_final", eval_top_k), eval_top_k, 1, 512),
        )
        eval_timeout_s = _clamp_float(
            query_timeout_s,
            self._eval_query_timeout_s(),
            0.01,
            120.0,
        )
        runtime_cfg = self._build_runtime_config(normalized)
        runtime = build_search_runtime(
            plugin_config=runtime_cfg,
            logger_obj=logger,
            owner_tag="retrieval_tuning",
            log_prefix="[RetrievalTuning]",
        )
        if not runtime.ready:
            metrics = {
                "total_text_cases": 0,
                "precision_at_1": 0.0,
                "precision_at_3": 0.0,
                "mrr": 0.0,
                "recall_at_k": 0.0,
                "spo_relation_hit_rate": 0.0,
                "empty_rate": 1.0,
                "avg_elapsed_ms": 0.0,
                "category": {},
                "error": runtime.error or "runtime_not_ready",
            }
            return {"metrics": metrics, "score": -1.0, "avg_elapsed_ms": 0.0, "failure_summary": {"reason": metrics["error"]}}

        text_total = 0
        hit1 = 0
        hit3 = 0
        hitk = 0
        mrr_sum = 0.0
        empty_count = 0
        timeout_count = 0
        elapsed_total = 0.0
        text_failed: List[str] = []

        spo_total = 0
        spo_hit = 0
        spo_failed: List[str] = []

        category_stats: Dict[str, Dict[str, Any]] = {}
        failed_predicates = Counter()

        for case in cases:
            cat = str(case.category)
            if cat not in CATEGORIES:
                continue
            if cat not in category_stats:
                category_stats[cat] = {
                    "total": 0,
                    "hit": 0,
                    "hit_at_1": 0,
                    "hit_at_3": 0,
                    "empty": 0,
                }
            category_stats[cat]["total"] += 1

            if cat == "spo_relation":
                spo_total += 1
                spo = case.expected_spo or {}
                rows = runtime.metadata_store.get_relations(
                    subject=str(spo.get("subject") or ""),
                    predicate=str(spo.get("predicate") or ""),
                    object=str(spo.get("object") or ""),
                )
                expected_hash = str(case.expected_hashes[0]) if case.expected_hashes else ""
                ok = False
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    if expected_hash and str(row.get("hash") or "") == expected_hash:
                        ok = True
                        break
                    if not expected_hash:
                        ok = True
                        break
                if ok:
                    spo_hit += 1
                    category_stats[cat]["hit"] += 1
                    category_stats[cat]["hit_at_1"] += 1
                    category_stats[cat]["hit_at_3"] += 1
                else:
                    spo_failed.append(case.case_id)
                    failed_predicates.update([str(spo.get("predicate") or "").strip() or "__empty__"])
                continue

            text_total += 1
            req = SearchExecutionRequest(
                caller="retrieval_tuning",
                query_type="search",
                query=str(case.query or "").strip(),
                top_k=int(request_top_k),
                use_threshold=True,
                # 调优评估固定关闭 PPR，避免该链路阻塞拖挂整轮任务。
                enable_ppr=False,
            )
            try:
                execution = await asyncio.wait_for(
                    SearchExecutionService.execute(
                        retriever=runtime.retriever,
                        threshold_filter=runtime.threshold_filter,
                        plugin_config=runtime_cfg,
                        request=req,
                        enforce_chat_filter=False,
                        reinforce_access=False,
                    ),
                    timeout=float(eval_timeout_s),
                )
            except asyncio.TimeoutError:
                timeout_count += 1
                empty_count += 1
                category_stats[cat]["empty"] += 1
                text_failed.append(case.case_id)
                failed_predicates.update([str(case.metadata.get("predicate") or "__unknown__")])
                continue

            if execution is None:
                empty_count += 1
                category_stats[cat]["empty"] += 1
                text_failed.append(case.case_id)
                failed_predicates.update([str(case.metadata.get("predicate") or "__unknown__")])
                continue

            elapsed_total += float(getattr(execution, "elapsed_ms", 0.0) or 0.0)

            if not bool(getattr(execution, "success", False)):
                empty_count += 1
                category_stats[cat]["empty"] += 1
                text_failed.append(case.case_id)
                failed_predicates.update([str(case.metadata.get("predicate") or "__unknown__")])
                continue

            hashes = [str(getattr(x, "hash_value", "") or "") for x in (getattr(execution, "results", None) or [])]
            if not hashes:
                empty_count += 1
                category_stats[cat]["empty"] += 1

            expected_set = set(case.expected_hashes or [])
            rank = 0
            for idx, hv in enumerate(hashes, start=1):
                if hv and hv in expected_set:
                    rank = idx
                    break

            if rank > 0:
                category_stats[cat]["hit"] += 1
                hitk += 1
                if rank <= 1:
                    hit1 += 1
                    category_stats[cat]["hit_at_1"] += 1
                if rank <= 3:
                    hit3 += 1
                    category_stats[cat]["hit_at_3"] += 1
                mrr_sum += 1.0 / float(rank)
            else:
                text_failed.append(case.case_id)
                failed_predicates.update([str(case.metadata.get("predicate") or "__unknown__")])

        p1 = (hit1 / text_total) if text_total else 0.0
        p3 = (hit3 / text_total) if text_total else 0.0
        recall = (hitk / text_total) if text_total else 0.0
        mrr = (mrr_sum / text_total) if text_total else 0.0
        spo_rate = (spo_hit / spo_total) if spo_total else 0.0
        empty_rate = (empty_count / text_total) if text_total else 1.0
        avg_elapsed = (elapsed_total / text_total) if text_total else 0.0

        metrics = {
            "total_text_cases": int(text_total),
            "precision_at_1": float(round(p1, 6)),
            "precision_at_3": float(round(p3, 6)),
            "mrr": float(round(mrr, 6)),
            "recall_at_k": float(round(recall, 6)),
            "spo_relation_hit_rate": float(round(spo_rate, 6)),
            "empty_rate": float(round(empty_rate, 6)),
            "timeout_count": int(timeout_count),
            "avg_elapsed_ms": float(round(avg_elapsed, 3)),
            "category": category_stats,
        }
        metrics["category_floor_penalty"] = float(round(self._category_floor_penalty(metrics, objective=objective), 6))

        score = self._score_metrics(metrics, objective=objective)
        failure_summary = {
            "text_failed_count": len(text_failed),
            "spo_failed_count": len(spo_failed),
            "failed_case_ids": text_failed[:50] + spo_failed[:50],
            "failed_by_category": {k: int(v["total"] - v["hit"]) for k, v in category_stats.items()},
            "top_failed_predicates": [
                {"predicate": key, "count": int(cnt)}
                for key, cnt in failed_predicates.most_common(5)
                if key
            ],
            "query_timeout_seconds": float(eval_timeout_s),
            "timeout_count": int(timeout_count),
            "effective_top_k": int(request_top_k),
            "ppr_forced_disabled": True,
        }
        return {
            "metrics": metrics,
            "score": float(round(score, 6)),
            "avg_elapsed_ms": float(avg_elapsed),
            "failure_summary": failure_summary,
        }

    def _score_metrics(self, metrics: Dict[str, Any], *, objective: str) -> float:
        p1 = float(metrics.get("precision_at_1", 0.0) or 0.0)
        p3 = float(metrics.get("precision_at_3", 0.0) or 0.0)
        mrr = float(metrics.get("mrr", 0.0) or 0.0)
        recall = float(metrics.get("recall_at_k", 0.0) or 0.0)
        spo = float(metrics.get("spo_relation_hit_rate", 0.0) or 0.0)
        empty_rate = float(metrics.get("empty_rate", 1.0) or 1.0)
        category_penalty = metrics.get("category_floor_penalty", None)
        if category_penalty is None:
            category_penalty = self._category_floor_penalty(metrics, objective=objective)
        category_penalty = float(max(0.0, category_penalty))

        if objective == "recall_priority":
            raw = 0.15 * p1 + 0.15 * p3 + 0.15 * mrr + 0.40 * recall + 0.15 * spo
            penalty = 0.05 * empty_rate
        elif objective == "balanced":
            raw = 0.25 * p1 + 0.20 * p3 + 0.15 * mrr + 0.25 * recall + 0.15 * spo
            penalty = 0.10 * empty_rate
        else:
            raw = 0.40 * p1 + 0.20 * p3 + 0.15 * mrr + 0.15 * recall + 0.10 * spo
            penalty = 0.15 * empty_rate
        return float(raw - penalty - category_penalty)

    def _category_floor_penalty(self, metrics: Dict[str, Any], *, objective: str) -> float:
        category = metrics.get("category")
        if not isinstance(category, dict) or not category:
            return 0.0

        if objective == "recall_priority":
            floors = {"query_nl": 0.60, "query_kw": 0.48, "spo_search": 0.52, "spo_relation": 0.88}
            scale = 0.12
        elif objective == "balanced":
            floors = {"query_nl": 0.65, "query_kw": 0.52, "spo_search": 0.55, "spo_relation": 0.90}
            scale = 0.18
        else:
            floors = {"query_nl": 0.70, "query_kw": 0.55, "spo_search": 0.58, "spo_relation": 0.92}
            scale = 0.25

        weights = {"query_nl": 1.0, "query_kw": 1.1, "spo_search": 1.0, "spo_relation": 1.2}
        weighted_shortfall = 0.0
        weight_total = 0.0

        for cat, floor in floors.items():
            row = category.get(cat)
            if not isinstance(row, dict):
                continue
            total = int(row.get("total", 0) or 0)
            if total <= 0:
                continue
            hit = float(row.get("hit", 0.0) or 0.0)
            hit_rate = max(0.0, min(1.0, hit / float(max(1, total))))
            shortfall = max(0.0, float(floor) - hit_rate)
            w = float(weights.get(cat, 1.0))
            weighted_shortfall += w * shortfall
            weight_total += w

        if weight_total <= 1e-9:
            return 0.0
        return float(scale * (weighted_shortfall / weight_total))

    def _build_report_payload(self, task: RetrievalTuningTaskRecord) -> Dict[str, Any]:
        baseline = task.baseline_metrics or {}
        best = task.best_metrics or {}

        def delta(name: str) -> float:
            return float(best.get(name, 0.0) or 0.0) - float(baseline.get(name, 0.0) or 0.0)

        return {
            "task_id": task.task_id,
            "objective": task.objective,
            "intensity": task.intensity,
            "status": task.status,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "finished_at": task.finished_at,
            "rounds_total": task.rounds_total,
            "rounds_done": task.rounds_done,
            "best_score": task.best_score,
            "baseline_score": self._score_metrics(baseline, objective=task.objective),
            "query_set_stats": task.query_set_stats,
            "baseline_metrics": baseline,
            "best_metrics": best,
            "deltas": {
                "precision_at_1": delta("precision_at_1"),
                "precision_at_3": delta("precision_at_3"),
                "mrr": delta("mrr"),
                "recall_at_k": delta("recall_at_k"),
                "spo_relation_hit_rate": delta("spo_relation_hit_rate"),
                "empty_rate": delta("empty_rate"),
                "timeout_count": delta("timeout_count"),
                "avg_elapsed_ms": delta("avg_elapsed_ms"),
            },
            "best_profile": task.best_profile,
            "baseline_profile": task.baseline_profile,
            "apply_log": task.apply_log,
        }

    def _build_report_markdown(self, task: RetrievalTuningTaskRecord, payload: Dict[str, Any]) -> str:
        baseline = payload.get("baseline_metrics", {}) or {}
        best = payload.get("best_metrics", {}) or {}
        d = payload.get("deltas", {}) or {}
        lines = [
            f"# 检索调优报告（{task.task_id}）",
            "",
            "## 1. 任务信息",
            f"- 状态: {task.status}",
            f"- 目标函数: {task.objective}",
            f"- 强度: {task.intensity}",
            f"- 轮次: baseline + {task.rounds_total}",
            f"- 创建时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task.created_at))}",
            f"- 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task.started_at)) if task.started_at else '-'}",
            f"- 完成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task.finished_at)) if task.finished_at else '-'}",
            "",
            "## 2. 基线 vs 最优",
            f"- baseline score: {payload.get('baseline_score', 0.0):.6f}",
            f"- best score: {task.best_score:.6f}",
            f"- P@1: {baseline.get('precision_at_1', 0.0):.4f} -> {best.get('precision_at_1', 0.0):.4f} (Δ {d.get('precision_at_1', 0.0):+.4f})",
            f"- P@3: {baseline.get('precision_at_3', 0.0):.4f} -> {best.get('precision_at_3', 0.0):.4f} (Δ {d.get('precision_at_3', 0.0):+.4f})",
            f"- MRR: {baseline.get('mrr', 0.0):.4f} -> {best.get('mrr', 0.0):.4f} (Δ {d.get('mrr', 0.0):+.4f})",
            f"- Recall@K: {baseline.get('recall_at_k', 0.0):.4f} -> {best.get('recall_at_k', 0.0):.4f} (Δ {d.get('recall_at_k', 0.0):+.4f})",
            f"- SPO relation hit: {baseline.get('spo_relation_hit_rate', 0.0):.4f} -> {best.get('spo_relation_hit_rate', 0.0):.4f} (Δ {d.get('spo_relation_hit_rate', 0.0):+.4f})",
            f"- 空结果率: {baseline.get('empty_rate', 0.0):.4f} -> {best.get('empty_rate', 0.0):.4f} (Δ {d.get('empty_rate', 0.0):+.4f})",
            f"- 超时数: {int(baseline.get('timeout_count', 0) or 0)} -> {int(best.get('timeout_count', 0) or 0)} (Δ {int(d.get('timeout_count', 0) or 0):+d})",
            f"- 平均耗时(ms): {baseline.get('avg_elapsed_ms', 0.0):.2f} -> {best.get('avg_elapsed_ms', 0.0):.2f} (Δ {d.get('avg_elapsed_ms', 0.0):+.2f})",
            "",
            "## 3. 最优参数",
            "```json",
            json.dumps(task.best_profile, ensure_ascii=False, indent=2),
            "```",
            "",
            "## 4. 测试集规模",
            f"- {json.dumps(task.query_set_stats, ensure_ascii=False)}",
            "",
            "## 5. 说明",
            "- 本报告仅对当前已存储图谱与向量状态有效。",
            "- 参数应用策略：运行时生效，不自动写入 config.toml。",
        ]
        return "\n".join(lines).strip() + "\n"

