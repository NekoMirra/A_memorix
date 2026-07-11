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


class RetrievalTuningQueryMixin:
    """Query-set sampling and NL/keyword construction."""
    def _sample_triples_for_query_set(
        self,
        *,
        triples: List[Tuple[Any, Any, Any, Any]],
        sample_size: int,
        seed: int,
    ) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, Any]]:
        normalized: List[Tuple[str, str, str, str]] = []
        for row in triples:
            try:
                subject, predicate, obj, rel_hash = row
            except Exception:
                continue
            relation_hash = str(rel_hash or "").strip()
            if not relation_hash:
                continue
            normalized.append((str(subject or ""), str(predicate or ""), str(obj or ""), relation_hash))

        if not normalized:
            return [], {"error": "no_relations"}

        target = min(max(4, int(sample_size)), len(normalized))
        predicate_counter = Counter([str(x[1] or "").strip() or "__empty__" for x in normalized])
        entity_counter = Counter()
        for subj, _, obj, _ in normalized:
            entity_counter.update([str(subj or "").strip().lower() or "__empty__"])
            entity_counter.update([str(obj or "").strip().lower() or "__empty__"])

        if target >= len(normalized):
            return list(normalized), {
                "strategy": "all",
                "sample_size": int(target),
                "total_triples": int(len(normalized)),
                "predicate_total": int(len(predicate_counter)),
                "predicate_sampled": int(len(predicate_counter)),
            }

        rng = random.Random(f"{seed}:triple_sample")
        by_predicate: Dict[str, List[int]] = {}
        for idx, (_, predicate, _, _) in enumerate(normalized):
            key = str(predicate or "").strip() or "__empty__"
            by_predicate.setdefault(key, []).append(idx)
        for pool in by_predicate.values():
            rng.shuffle(pool)

        predicate_order = sorted(by_predicate.keys())
        rng.shuffle(predicate_order)

        selected: List[int] = []
        selected_set = set()

        # First pass: predicate round-robin to avoid head predicate dominating query set.
        while len(selected) < target:
            progressed = False
            for key in predicate_order:
                pool = by_predicate.get(key, [])
                if not pool:
                    continue
                idx = int(pool.pop())
                if idx in selected_set:
                    continue
                selected.append(idx)
                selected_set.add(idx)
                progressed = True
                if len(selected) >= target:
                    break
            if not progressed:
                break

        if len(selected) < target:
            remain = [idx for idx in range(len(normalized)) if idx not in selected_set]
            rng.shuffle(remain)

            # Second pass: prefer lower-frequency entities and predicates for better diversity.
            def _remain_score(idx: int) -> Tuple[int, int]:
                subj, predicate, obj, _ = normalized[idx]
                subject_freq = int(entity_counter.get(str(subj or "").strip().lower() or "__empty__", 0))
                object_freq = int(entity_counter.get(str(obj or "").strip().lower() or "__empty__", 0))
                pred_freq = int(predicate_counter.get(str(predicate or "").strip() or "__empty__", 0))
                return (subject_freq + object_freq, pred_freq)

            remain = sorted(remain, key=_remain_score)
            need = target - len(selected)
            for idx in remain[:need]:
                selected.append(idx)
                selected_set.add(idx)

        selected = selected[:target]
        sampled = [normalized[idx] for idx in selected]
        sampled_predicates = {str(x[1] or "").strip() or "__empty__" for x in sampled}

        return sampled, {
            "strategy": "predicate_round_robin_entity_diversity",
            "sample_size": int(target),
            "total_triples": int(len(normalized)),
            "predicate_total": int(len(predicate_counter)),
            "predicate_sampled": int(len(sampled_predicates)),
        }

    def _select_round_eval_cases(
        self,
        *,
        cases: List[RetrievalQueryCase],
        intensity: str,
        round_index: int,
        seed: int,
    ) -> List[RetrievalQueryCase]:
        if not cases:
            return []
        mode = str(intensity or "standard").strip().lower()
        if mode not in INTENSITIES:
            mode = "standard"
        if mode == "deep":
            return list(cases)

        if mode == "quick":
            ratio = 0.45
            min_total = 16
        else:
            ratio = 0.70
            min_total = 24

        total = len(cases)
        target = max(min_total, int(total * ratio))
        if target >= total:
            return list(cases)

        rng = random.Random(f"{seed}:{round_index}:subset")
        by_cat: Dict[str, List[RetrievalQueryCase]] = {}
        for item in cases:
            by_cat.setdefault(str(item.category), []).append(item)

        selected: List[RetrievalQueryCase] = []
        selected_ids = set()
        cat_names = sorted([x for x in by_cat.keys() if x in CATEGORIES])
        if not cat_names:
            cat_names = sorted(by_cat.keys())
        per_cat = max(1, target // max(1, len(cat_names)))

        for cat in cat_names:
            pool = by_cat.get(cat, [])
            if not pool:
                continue
            picked = list(pool) if len(pool) <= per_cat else rng.sample(pool, per_cat)
            for item in picked:
                if item.case_id in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(item.case_id)

        if len(selected) < target:
            remain = [x for x in cases if x.case_id not in selected_ids]
            if len(remain) > (target - len(selected)):
                remain = rng.sample(remain, target - len(selected))
            for item in remain:
                selected.append(item)
                selected_ids.add(item.case_id)

        return selected[:target]

    async def _build_query_set(self, *, sample_size: int, seed: int, llm_enabled: bool) -> Tuple[List[RetrievalQueryCase], Dict[str, Any]]:
        store = getattr(self.plugin, "metadata_store", None)
        if store is None:
            return [], {"error": "metadata_store_unavailable"}

        triples = list(store.get_all_triples() or [])
        if not triples:
            return [], {"error": "no_relations"}

        sampled, sample_info = self._sample_triples_for_query_set(
            triples=triples,
            sample_size=sample_size,
            seed=seed,
        )
        if not sampled:
            return [], {"error": "no_relations"}

        anchors: List[Dict[str, Any]] = []
        for idx, row in enumerate(sampled):
            subject, predicate, obj, relation_hash = row
            paragraphs = store.get_paragraphs_by_relation(relation_hash)
            para_hash = ""
            para_content = ""
            if paragraphs:
                para_hash = str(paragraphs[0].get("hash") or "").strip()
                para_content = str(paragraphs[0].get("content") or "")
            anchors.append(
                {
                    "anchor_id": f"a{idx+1:04d}",
                    "subject": str(subject or ""),
                    "predicate": str(predicate or ""),
                    "object": str(obj or ""),
                    "relation_hash": relation_hash,
                    "paragraph_hash": para_hash,
                    "paragraph_excerpt": para_content[:300],
                }
            )

        if not anchors:
            return [], {"error": "no_anchors"}

        predicate_groups: Dict[str, List[Dict[str, Any]]] = {}
        for anchor in anchors:
            predicate_groups.setdefault(str(anchor.get("predicate") or ""), []).append(anchor)

        nl_queries = await self._generate_nl_queries_with_llm(anchors, enabled=llm_enabled)
        cases: List[RetrievalQueryCase] = []

        seq = 0
        for anchor in anchors:
            seq += 1
            subject = anchor["subject"]
            predicate = anchor["predicate"]
            obj = anchor["object"]
            rel_hash = anchor["relation_hash"]
            para_hash = anchor["paragraph_hash"]
            expected = [rel_hash]
            if para_hash:
                expected.append(para_hash)
            aid = anchor["anchor_id"]

            common_meta = {
                "anchor_id": aid,
                "relation_hash": rel_hash,
                "paragraph_hash": para_hash,
                "subject": subject,
                "predicate": predicate,
                "object": obj,
            }
            cases.append(
                RetrievalQueryCase(
                    case_id=f"spo_relation_{seq:04d}",
                    category="spo_relation",
                    query=f"{subject}|{predicate}|{obj}",
                    expected_hashes=[rel_hash],
                    expected_spo={"subject": subject, "predicate": predicate, "object": obj},
                    metadata=dict(common_meta),
                )
            )
            cases.append(
                RetrievalQueryCase(
                    case_id=f"spo_search_{seq:04d}",
                    category="spo_search",
                    query=self._build_spo_search_query(
                        anchor=anchor,
                        seq=seq,
                        predicate_groups=predicate_groups,
                    ),
                    expected_hashes=list(expected),
                    metadata=dict(common_meta),
                )
            )
            cases.append(
                RetrievalQueryCase(
                    case_id=f"query_kw_{seq:04d}",
                    category="query_kw",
                    query=self._build_keyword_query(
                        anchor=anchor,
                        seq=seq,
                        predicate_groups=predicate_groups,
                    ),
                    expected_hashes=list(expected),
                    metadata=dict(common_meta),
                )
            )
            nl_query = nl_queries.get(aid) or self._build_nl_template(
                anchor=anchor,
                seq=seq,
                predicate_groups=predicate_groups,
            )
            cases.append(
                RetrievalQueryCase(
                    case_id=f"query_nl_{seq:04d}",
                    category="query_nl",
                    query=nl_query,
                    expected_hashes=list(expected),
                    metadata=dict(common_meta),
                )
            )

        counts = Counter([c.category for c in cases])
        stats = {
            "anchors": len(anchors),
            "case_total": len(cases),
            "category_counts": {k: int(v) for k, v in counts.items()},
            "seed": int(seed),
            "sample_size": int(sample_info.get("sample_size", len(anchors))),
            "sampling": dict(sample_info),
            "llm_nl_enabled": bool(llm_enabled),
            "llm_nl_generated": int(len(nl_queries)),
        }
        return cases, stats

    def _pick_contrast_anchor(
        self,
        *,
        anchor: Dict[str, Any],
        predicate_groups: Dict[str, List[Dict[str, Any]]],
        seq: int,
    ) -> Optional[Dict[str, Any]]:
        predicate = str(anchor.get("predicate") or "")
        pool = predicate_groups.get(predicate, [])
        if not pool:
            return None
        candidates = [x for x in pool if x is not anchor and str(x.get("object") or "") != str(anchor.get("object") or "")]
        if not candidates:
            return None
        return candidates[seq % len(candidates)]

    def _build_spo_search_query(
        self,
        *,
        anchor: Dict[str, Any],
        seq: int,
        predicate_groups: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        subject = str(anchor.get("subject") or "")
        predicate = str(anchor.get("predicate") or "")
        obj = str(anchor.get("object") or "")
        contrast = self._pick_contrast_anchor(anchor=anchor, predicate_groups=predicate_groups, seq=seq)
        contrast_obj = str(contrast.get("object") or "").strip() if contrast else ""

        variants = [
            f"{subject} {predicate} {obj}",
            f"{subject} {obj} relation {predicate}",
            f"{predicate} {subject} {obj} evidence",
            f"{subject} {predicate} {obj} not {contrast_obj}".strip(),
        ]
        return variants[seq % len(variants)].strip()

    def _build_keyword_query(
        self,
        *,
        anchor: Dict[str, Any],
        seq: int,
        predicate_groups: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        subject = str(anchor.get("subject") or "")
        predicate = str(anchor.get("predicate") or "")
        obj = str(anchor.get("object") or "")
        excerpt = str(anchor.get("paragraph_excerpt") or "")
        tokens = re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]{2,}", excerpt)
        extras: List[str] = []
        seen = set()
        for token in tokens:
            key = token.lower()
            if key in seen:
                continue
            if key in {subject.lower(), predicate.lower(), obj.lower()}:
                continue
            seen.add(key)
            extras.append(token)
            if len(extras) >= 2:
                break
        contrast = self._pick_contrast_anchor(anchor=anchor, predicate_groups=predicate_groups, seq=seq)
        contrast_obj = str(contrast.get("object") or "").strip() if contrast else ""

        variants = [
            [subject, obj] + extras[:2],
            [predicate, obj] + extras[:2],
            [subject, predicate] + extras[:2],
            [subject, obj, predicate, contrast_obj] + extras[:1],
        ]
        parts = variants[seq % len(variants)]
        return " ".join([x for x in parts if x]).strip()

    def _build_nl_template(
        self,
        *,
        anchor: Dict[str, Any],
        seq: int,
        predicate_groups: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        subject = str(anchor.get("subject") or "")
        predicate = str(anchor.get("predicate") or "")
        obj = str(anchor.get("object") or "")
        contrast = self._pick_contrast_anchor(anchor=anchor, predicate_groups=predicate_groups, seq=seq)
        contrast_obj = str(contrast.get("object") or "").strip() if contrast else ""
        templates = [
            f"请问 {subject} 与 {obj} 的关系是什么，是否是“{predicate}”？",
            f"在当前知识库中，哪条信息说明 {subject} 对应的是 {obj}，关系词接近“{predicate}”？",
            f"我想确认：{subject} 和 {obj} 之间是不是“{predicate}”这层关系，而不是 {contrast_obj}？",
            f"帮我查一下关于 {subject} 与 {obj} 的证据，重点看 {predicate} 相关描述。",
        ]
        return templates[seq % len(templates)]

    async def _select_llm_model(self) -> Optional[Any]:
        if llm_api is None:
            return None
        try:
            models = llm_api.get_available_models() or {}
        except Exception:
            return None
        if not models:
            return None

        cfg_model = str(self._cfg("advanced.extraction_model", "auto") or "auto").strip()
        if cfg_model.lower() != "auto" and cfg_model in models:
            return models[cfg_model]
        for task_name in ["utils", "planner", "tool_use", "replyer", "embedding"]:
            if task_name in models:
                return models[task_name]
        return models[next(iter(models))]

    async def _llm_call_text(self, prompt: str, *, request_type: str) -> str:
        if llm_api is None:
            raise RuntimeError("llm_api unavailable")
        model_cfg = await self._select_llm_model()
        if model_cfg is None:
            raise RuntimeError("no_llm_model")
        task_name = llm_api.resolve_task_name_from_model_config(model_cfg)

        retry = self._llm_retry_cfg()
        max_attempts = int(retry["max_attempts"])
        min_wait = float(retry["min_wait_seconds"])
        max_wait = float(retry["max_wait_seconds"])
        backoff = float(retry["backoff_multiplier"])

        last_error: Optional[Exception] = None
        for idx in range(max_attempts):
            try:
                result = await llm_api.generate(
                    llm_api.LLMServiceRequest(
                        task_name=task_name,
                        request_type=request_type,
                        prompt=prompt,
                        temperature=getattr(model_cfg, "temperature", None),
                        max_tokens=getattr(model_cfg, "max_tokens", None),
                    )
                )
                success = bool(result.success)
                response = str(result.completion.response or "")
                if not success:
                    raise RuntimeError("llm_generation_failed")
                text = str(response or "").strip()
                if text:
                    return text
                raise RuntimeError("empty_llm_response")
            except Exception as e:
                last_error = e
                if idx >= max_attempts - 1:
                    break
                delay = min(max_wait, min_wait * (backoff ** idx))
                await asyncio.sleep(max(0.05, delay))
        raise RuntimeError(f"LLM call failed: {last_error}")

    async def _generate_nl_queries_with_llm(self, anchors: List[Dict[str, Any]], *, enabled: bool) -> Dict[str, str]:
        if not enabled or llm_api is None or not anchors:
            return {}
        payload = [
            {
                "anchor_id": x["anchor_id"],
                "subject": x["subject"],
                "predicate": x["predicate"],
                "object": x["object"],
                "paragraph_excerpt": x["paragraph_excerpt"][:180],
            }
            for x in anchors[:60]
        ]
        prompt = (
            "你是检索评估问题生成器。"
            "请基于给定 SPO 与简短上下文，为每条样本生成 1 条自然语言检索问题，返回 JSON："
            "{\"items\":[{\"anchor_id\":\"...\",\"query\":\"...\"}]}。\n"
            f"样本：\n{json.dumps(payload, ensure_ascii=False)}"
        )
        try:
            raw = await self._llm_call_text(prompt, request_type="A_Memorix.RetrievalTuning.NLCaseGen")
            obj = _safe_json_loads(raw)
            if not isinstance(obj, dict):
                return {}
            items = obj.get("items")
            if not isinstance(items, list):
                return {}
            out: Dict[str, str] = {}
            for row in items:
                if not isinstance(row, dict):
                    continue
                anchor_id = str(row.get("anchor_id") or "").strip()
                query = str(row.get("query") or "").strip()
                if anchor_id and query:
                    out[anchor_id] = query
            return out
        except Exception:
            return {}

