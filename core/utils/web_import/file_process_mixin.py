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


class ImportFileProcessMixin:
    """Per-file extraction, LLM, and persistence."""
    async def _read_file_content(self, file_record: ImportFileRecord) -> str:
        if file_record.inline_content is not None:
            return file_record.inline_content
        if file_record.source_path and Path(file_record.source_path).exists():
            data = Path(file_record.source_path).read_bytes()
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data.decode("utf-8", errors="replace")
        if file_record.temp_path and Path(file_record.temp_path).exists():
            data = Path(file_record.temp_path).read_bytes()
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data.decode("utf-8", errors="replace")
        raise RuntimeError("读取文件失败：输入内容缺失")

    async def _process_text_file(
        self,
        task_id: str,
        file_record: ImportFileRecord,
        content: str,
        chunk_semaphore: asyncio.Semaphore,
    ) -> None:
        task = self._tasks[task_id]
        async with self._lock:
            t = self._tasks.get(task_id)
            if t and not t.schema_detected:
                t.schema_detected = "plain_text"
        strategy = self._determine_strategy(
            file_record.name,
            content,
            task.params["strategy_override"],
            chat_log=bool(task.params.get("chat_log")),
        )
        await self._set_file_strategy(task_id, file_record.file_id, strategy)
        await self._set_file_state(task_id, file_record.file_id, "splitting", "splitting")
        await self._ensure_embedding_runtime_ready()

        chunks = strategy.split(content)
        selected_chunks = list(chunks)
        if file_record.retry_mode == "chunk":
            retry_index_set = set()
            for idx in file_record.retry_chunk_indexes:
                try:
                    retry_index_set.add(int(idx))
                except Exception:
                    continue
            selected_chunks = [chunk for chunk in chunks if int(chunk.chunk.index) in retry_index_set]
            if not selected_chunks:
                raise RuntimeError("失败分块重试索引无效，未匹配到可执行分块")
            logger.info(
                "重试任务按失败分块执行: "
                f"file={file_record.name} "
                f"selected={len(selected_chunks)} "
                f"total={len(chunks)}"
            )

        await self._register_chunks(task_id, file_record.file_id, selected_chunks)

        await self._set_file_state(task_id, file_record.file_id, "extracting", "extracting")
        model_cfg = None
        if task.params["llm_enabled"]:
            model_cfg = await self._select_model()

        jobs = []
        for chunk in selected_chunks:
            jobs.append(
                asyncio.create_task(
                    self._process_text_chunk(
                        task_id=task_id,
                        file_record=file_record,
                        chunk=chunk,
                        strategy=strategy,
                        llm_enabled=task.params["llm_enabled"],
                        model_cfg=model_cfg,
                        chunk_semaphore=chunk_semaphore,
                        chat_log=bool(task.params.get("chat_log")),
                        chat_reference_time=str(task.params.get("chat_reference_time") or "").strip() or None,
                    )
                )
            )
        await asyncio.gather(*jobs, return_exceptions=True)

        if await self._is_cancel_requested(task_id):
            await self._set_file_cancelled(task_id, file_record.file_id, "任务已取消")
            return

        await self._set_file_state(task_id, file_record.file_id, "saving", "saving")
        async with self._storage_lock:
            self.plugin.vector_store.save()
            self.plugin.graph_store.save()

        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            f = self._find_file(task, file_record.file_id)
            if not f:
                return
            if f.failed_chunks > 0:
                f.status = "failed"
                f.current_step = "failed"
                if not f.error:
                    f.error = f"存在失败分块: {f.failed_chunks}"
            elif task.status == "cancel_requested":
                f.status = "cancelled"
                f.current_step = "cancelled"
            else:
                f.status = "completed"
                f.current_step = "completed"
                f.progress = 1.0
            f.updated_at = _now()
            self._recompute_task_progress(task)

    async def _process_text_chunk(
        self,
        task_id: str,
        file_record: ImportFileRecord,
        chunk: ProcessedChunk,
        strategy: Any,
        llm_enabled: bool,
        model_cfg: Any,
        chunk_semaphore: asyncio.Semaphore,
        chat_log: bool = False,
        chat_reference_time: Optional[str] = None,
    ) -> None:
        async with chunk_semaphore:
            chunk_id = chunk.chunk.chunk_id
            if await self._is_cancel_requested(task_id):
                await self._set_chunk_cancelled(task_id, file_record.file_id, chunk_id, "任务已取消")
                return

            await self._set_chunk_state(task_id, file_record.file_id, chunk_id, "extracting", "extracting", 0.25)

            processed = chunk
            rescue_strategy = self._chunk_rescue(chunk, file_record.name)
            current_strategy = strategy
            if rescue_strategy:
                chunk.type = StrategyKnowledgeType.QUOTE
                chunk.flags.verbatim = True
                chunk.flags.requires_llm = False
                current_strategy = rescue_strategy
            try:
                if llm_enabled and chunk.flags.requires_llm:
                    processed = await current_strategy.extract(
                        chunk,
                        lambda prompt: self._llm_call(prompt, model_cfg),
                    )
                elif chunk.type == StrategyKnowledgeType.QUOTE:
                    processed = await current_strategy.extract(chunk)
            except Exception as e:
                await self._set_chunk_failed(task_id, file_record.file_id, chunk_id, f"抽取失败: {e}")
                return

            if await self._is_cancel_requested(task_id):
                await self._set_chunk_cancelled(task_id, file_record.file_id, chunk_id, "任务已取消")
                return

            await self._set_chunk_state(task_id, file_record.file_id, chunk_id, "writing", "writing", 0.7)
            try:
                time_meta = None
                if chat_log and llm_enabled and model_cfg is not None:
                    time_meta = await self._extract_chat_time_meta_with_llm(
                        processed.chunk.text,
                        model_cfg,
                        reference_time=chat_reference_time,
                    )
                async with self._storage_lock:
                    await self._persist_processed_chunk(file_record, processed, time_meta=time_meta)
                await self._set_chunk_completed(task_id, file_record.file_id, chunk_id)
            except Exception as e:
                await self._set_chunk_failed(task_id, file_record.file_id, chunk_id, f"写入失败: {e}")

    async def _process_json_file(
        self,
        task_id: str,
        file_record: ImportFileRecord,
        content: str,
        chunk_semaphore: asyncio.Semaphore,
    ) -> None:
        await self._set_file_strategy(task_id, file_record.file_id, "json")
        await self._set_file_state(task_id, file_record.file_id, "splitting", "splitting")
        await self._ensure_embedding_runtime_ready()

        try:
            data = json.loads(content)
        except Exception as e:
            raise RuntimeError(f"JSON 解析失败: {e}")

        schema = self._detect_json_schema(data)
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.schema_detected = schema
                task.updated_at = _now()
        units, build_warnings = self._build_json_units(data, file_record.file_id, file_record.name, schema)
        if build_warnings:
            await self._append_file_warnings(task_id, file_record.file_id, build_warnings)
        await self._register_json_units(task_id, file_record.file_id, units)

        await self._set_file_state(task_id, file_record.file_id, "extracting", "extracting")
        jobs = [
            asyncio.create_task(self._process_json_unit(task_id, file_record, unit, chunk_semaphore))
            for unit in units
        ]
        await asyncio.gather(*jobs, return_exceptions=True)

        if await self._is_cancel_requested(task_id):
            await self._set_file_cancelled(task_id, file_record.file_id, "任务已取消")
            return

        await self._set_file_state(task_id, file_record.file_id, "saving", "saving")
        async with self._storage_lock:
            self.plugin.vector_store.save()
            self.plugin.graph_store.save()

        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            f = self._find_file(task, file_record.file_id)
            if not f:
                return
            if f.failed_chunks > 0:
                f.status = "failed"
                f.current_step = "failed"
                if not f.error:
                    f.error = f"存在失败分块: {f.failed_chunks}"
            elif task.status == "cancel_requested":
                f.status = "cancelled"
                f.current_step = "cancelled"
            else:
                f.status = "completed"
                f.current_step = "completed"
                f.progress = 1.0
            f.updated_at = _now()
            self._recompute_task_progress(task)

    def _detect_json_schema(self, data: Any) -> str:
        if isinstance(data, dict) and isinstance(data.get("docs"), list):
            return "lpmm_openie"
        if isinstance(data, dict) and isinstance(data.get("paragraphs"), list):
            paragraphs = data.get("paragraphs", [])
            for p in paragraphs:
                if isinstance(p, dict) and any(
                    key in p for key in ("entities", "relations", "time_meta", "source", "type", "knowledge_type")
                ):
                    return "script_json"
            return "web_json"
        raise RuntimeError("不支持的 JSON 格式：需要 paragraphs 或 docs")

    def _build_json_units(
        self,
        data: Any,
        file_id: str,
        filename: str,
        schema: str,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        units: List[Dict[str, Any]] = []
        warnings: List[str] = []
        paragraphs: List[Any] = []
        entities: List[Any] = []
        relations: List[Any] = []

        if schema in {"web_json", "script_json"}:
            paragraphs = data.get("paragraphs", [])
            entities = data.get("entities", [])
            relations = data.get("relations", [])
        elif schema == "lpmm_openie":
            docs = data.get("docs", [])
            for d in docs:
                if not isinstance(d, dict):
                    continue
                content = str(d.get("passage", "") or "").strip()
                if not content:
                    continue
                triples = d.get("extracted_triples", []) or []
                rels = []
                for t in triples:
                    if isinstance(t, list) and len(t) == 3:
                        rels.append(
                            {
                                "subject": str(t[0]),
                                "predicate": str(t[1]),
                                "object": str(t[2]),
                            }
                        )
                para_item = {
                    "content": content,
                    "source": f"lpmm_openie:{filename}",
                    "entities": d.get("extracted_entities", []) or [],
                    "relations": rels,
                    "knowledge_type": "factual",
                }
                paragraphs.append(para_item)

        for paragraph_index, p in enumerate(paragraphs):
            try:
                paragraph = normalize_paragraph_import_item(
                    p,
                    default_source=f"web_import:{filename}",
                )
            except ImportPayloadValidationError as exc:
                warnings.append(
                    f"跳过段落[{paragraph_index}]：{exc} (code={exc.code})"
                )
                continue
            units.append(
                {
                    "chunk_id": f"{file_id}_json_{len(units)}",
                    "kind": "paragraph",
                    "content": paragraph["content"],
                    "time_meta": paragraph["time_meta"],
                    "knowledge_type": paragraph["knowledge_type"],
                    "chunk_type": paragraph["knowledge_type"],
                    "source": paragraph["source"],
                    "entities": paragraph["entities"],
                    "relations": paragraph["relations"],
                    "preview": paragraph["content"][:120],
                }
            )

        for entity_index, e in enumerate(entities):
            name = normalize_entity_import_item(e)
            if not name:
                raw = str(e or "").strip()
                warnings.append(
                    f"跳过实体[{entity_index}]：无效名称或疑似哈希值 ({raw[:80]})"
                )
                continue
            units.append(
                {
                    "chunk_id": f"{file_id}_json_{len(units)}",
                    "kind": "entity",
                    "name": name,
                    "chunk_type": "entity",
                    "preview": name[:120],
                }
            )

        for relation_index, r in enumerate(relations):
            relation = normalize_relation_import_item(r)
            if relation is None:
                if isinstance(r, dict):
                    raw = (
                        f"{str(r.get('subject', '')).strip()} | "
                        f"{str(r.get('predicate', '')).strip()} | "
                        f"{str(r.get('object', '')).strip()}"
                    )
                else:
                    raw = str(r or "").strip()
                warnings.append(
                    f"跳过关系[{relation_index}]：无效三元组或疑似哈希值 ({raw[:120]})"
                )
                continue
            units.append(
                {
                    "chunk_id": f"{file_id}_json_{len(units)}",
                    "kind": "relation",
                    "subject": relation["subject"],
                    "predicate": relation["predicate"],
                    "object": relation["object"],
                    "chunk_type": "relation",
                    "preview": f"{relation['subject']} {relation['predicate']} {relation['object']}"[:120],
                }
            )
        return units, warnings

    async def _register_json_units(self, task_id: str, file_id: str, units: List[Dict[str, Any]]) -> None:
        records = [
            ImportChunkRecord(
                chunk_id=u["chunk_id"],
                index=i,
                chunk_type=u.get("chunk_type", "json"),
                status="queued",
                step="queued",
                progress=0.0,
                content_preview=str(u.get("preview", "")),
            )
            for i, u in enumerate(units)
        ]
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            f = self._find_file(task, file_id)
            if not f:
                return
            f.chunks = records
            f.total_chunks = len(records)
            f.done_chunks = 0
            f.failed_chunks = 0
            f.cancelled_chunks = 0
            f.progress = 0.0 if records else 1.0
            f.updated_at = _now()
            self._recompute_task_progress(task)

    async def _process_json_unit(
        self,
        task_id: str,
        file_record: ImportFileRecord,
        unit: Dict[str, Any],
        chunk_semaphore: asyncio.Semaphore,
    ) -> None:
        chunk_id = unit["chunk_id"]
        async with chunk_semaphore:
            if await self._is_cancel_requested(task_id):
                await self._set_chunk_cancelled(task_id, file_record.file_id, chunk_id, "任务已取消")
                return

            await self._set_chunk_state(task_id, file_record.file_id, chunk_id, "writing", "writing", 0.7)
            try:
                chunk_warnings: List[str] = []
                skip_write = False
                async with self._storage_lock:
                    kind = unit["kind"]
                    if kind == "paragraph":
                        content = str(unit.get("content", ""))
                        if not content.strip():
                            chunk_warnings.append(f"跳过分块[{chunk_id}]：段落内容为空")
                            skip_write = True
                        elif is_probable_hash_token(content):
                            chunk_warnings.append(f"跳过分块[{chunk_id}]：段落内容疑似哈希值")
                            skip_write = True
                        if skip_write:
                            pass
                        k_type = resolve_stored_knowledge_type(
                            unit.get("knowledge_type"),
                            content=content,
                        ).value
                        source = str(unit.get("source") or f"web_import:{file_record.name}")
                        if not skip_write:
                            para_hash = self.plugin.metadata_store.add_paragraph(
                                content=content,
                                source=source,
                                knowledge_type=k_type,
                                time_meta=unit.get("time_meta"),
                            )
                            vector_result = await self._write_paragraph_vector_or_enqueue(
                                paragraph_hash=para_hash,
                                content=content,
                                context="web_import_json",
                            )
                            if str(vector_result.get("warning", "") or "").strip():
                                logger.warning(
                                    f"web_import json paragraph 向量写入降级: hash={para_hash[:8]} detail={vector_result.get('detail')}"
                                )
                            for name in unit.get("entities", []) or []:
                                n = str(name or "").strip()
                                if not n:
                                    continue
                                if is_probable_hash_token(n):
                                    chunk_warnings.append(
                                        f"跳过分块[{chunk_id}]中的实体：疑似哈希值 ({n[:32]})"
                                    )
                                    continue
                                await self._add_entity_with_vector(n, source_paragraph=para_hash)
                            for rel in unit.get("relations", []) or []:
                                if not isinstance(rel, dict):
                                    continue
                                s = str(rel.get("subject", "")).strip()
                                p = str(rel.get("predicate", "")).strip()
                                o = str(rel.get("object", "")).strip()
                                if not (s and p and o):
                                    continue
                                if any(is_probable_hash_token(token) for token in (s, p, o)):
                                    chunk_warnings.append(
                                        f"跳过分块[{chunk_id}]中的关系：疑似哈希值 ({s[:24]}|{p[:24]}|{o[:24]})"
                                    )
                                    continue
                                await self._add_relation(s, p, o, source_paragraph=para_hash)
                    elif kind == "entity":
                        entity_name = str(unit.get("name", "")).strip()
                        if not entity_name:
                            chunk_warnings.append(f"跳过分块[{chunk_id}]：实体名为空")
                            skip_write = True
                        elif is_probable_hash_token(entity_name):
                            chunk_warnings.append(f"跳过分块[{chunk_id}]：实体名疑似哈希值")
                            skip_write = True
                        if not skip_write:
                            await self._add_entity_with_vector(entity_name)
                    elif kind == "relation":
                        subject = str(unit.get("subject", "")).strip()
                        predicate = str(unit.get("predicate", "")).strip()
                        obj = str(unit.get("object", "")).strip()
                        if not (subject and predicate and obj):
                            chunk_warnings.append(f"跳过分块[{chunk_id}]：关系字段不完整")
                            skip_write = True
                        elif any(is_probable_hash_token(token) for token in (subject, predicate, obj)):
                            chunk_warnings.append(f"跳过分块[{chunk_id}]：关系字段疑似哈希值")
                            skip_write = True
                        if not skip_write:
                            await self._add_relation(subject, predicate, obj)
                    else:
                        raise RuntimeError(f"未知 JSON 导入单元类型: {kind}")
                if chunk_warnings:
                    await self._append_file_warnings(task_id, file_record.file_id, chunk_warnings)
                await self._set_chunk_completed(task_id, file_record.file_id, chunk_id)
            except Exception as e:
                await self._set_chunk_failed(task_id, file_record.file_id, chunk_id, f"写入失败: {e}")

    def _source_label(self, file_record: ImportFileRecord) -> str:
        if file_record.source_path:
            return f"{file_record.source_kind}:{file_record.source_path}"
        return f"web_import:{file_record.name}"

    async def _ensure_embedding_runtime_ready(self) -> None:
        report = await ensure_runtime_self_check(self.plugin)
        if bool(report.get("ok", False)):
            return
        if self._allow_metadata_only_write():
            logger.warning(
                "web_import embedding runtime self-check 失败，进入 metadata-only 回退模式: "
                f"{report.get('message', 'unknown')}"
            )
            return
        raise RuntimeError(
            "embedding runtime self-check failed: "
            f"{report.get('message', 'unknown')} "
            f"(configured={report.get('configured_dimension', 0)}, "
            f"store={report.get('vector_store_dimension', 0)}, "
            f"encoded={report.get('encoded_dimension', 0)})"
        )

    async def _persist_processed_chunk(
        self,
        file_record: ImportFileRecord,
        processed: ProcessedChunk,
        *,
        time_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        content = str(processed.chunk.text or "")
        if is_probable_hash_token(content):
            logger.warning("跳过疑似哈希段落写入: source=%s preview=%s", self._source_label(file_record), content[:32])
            return
        data = _coerce_import_data_dict(processed.data, context="分块抽取结果")
        para_hash = self.plugin.metadata_store.add_paragraph(
            content=content,
            source=self._source_label(file_record),
            knowledge_type=_storage_type_from_strategy(processed.type),
            time_meta=time_meta,
        )

        vector_result = await self._write_paragraph_vector_or_enqueue(
            paragraph_hash=para_hash,
            content=content,
            context="web_import_text",
        )
        if str(vector_result.get("warning", "") or "").strip():
            logger.warning(
                f"web_import text paragraph 向量写入降级: hash={para_hash[:8]} detail={vector_result.get('detail')}"
            )

        entities: List[str] = []
        relations: List[Tuple[str, str, str]] = []

        for triple in _normalize_import_relation_list(data.get("triples")):
            s = triple["subject"]
            p = triple["predicate"]
            o = triple["object"]
            relations.append((s, p, o))
            entities.extend([s, o])

        for rel in _normalize_import_relation_list(data.get("relations")):
            s = rel["subject"]
            p = rel["predicate"]
            o = rel["object"]
            relations.append((s, p, o))
            entities.extend([s, o])

        for k in ("entities", "events", "verbatim_entities"):
            entities.extend(_normalize_import_entity_list(data.get(k)))

        uniq_entities = list({x.strip().lower(): x.strip() for x in entities if str(x).strip()}.values())
        for name in uniq_entities:
            await self._add_entity_with_vector(name, source_paragraph=para_hash)

        for s, p, o in relations:
            await self._add_relation(s, p, o, source_paragraph=para_hash)

    async def _add_entity_with_vector(self, name: str, source_paragraph: str = "") -> str:
        name_token = str(name or "").strip()
        if not name_token:
            return ""
        if is_probable_hash_token(name_token):
            logger.warning(f"跳过疑似哈希实体写入: entity={name_token[:32]}")
            return ""

        hash_value = self.plugin.metadata_store.add_entity(name=name_token, source_paragraph=source_paragraph)
        self.plugin.graph_store.add_nodes([name_token])
        if hash_value not in self.plugin.vector_store:
            try:
                if self._is_embedding_degraded():
                    raise RuntimeError("embedding_degraded")
                emb = await self.plugin.embedding_manager.encode(name_token)
                self.plugin.vector_store.add(emb.reshape(1, -1), [hash_value])
            except Exception as exc:
                if not self._allow_metadata_only_write():
                    raise
                logger.warning(f"实体向量写入降级，保留 metadata/graph: entity={name_token} error={exc}")
        return hash_value

    async def _add_relation(self, subject: str, predicate: str, obj: str, source_paragraph: str = "") -> str:
        subject_token = str(subject or "").strip()
        predicate_token = str(predicate or "").strip()
        object_token = str(obj or "").strip()
        if not (subject_token and predicate_token and object_token):
            return ""
        if any(is_probable_hash_token(token) for token in (subject_token, predicate_token, object_token)):
            logger.warning(
                "跳过疑似哈希关系写入: %s | %s | %s",
                subject_token[:24],
                predicate_token[:24],
                object_token[:24],
            )
            return ""

        await self._add_entity_with_vector(subject_token, source_paragraph=source_paragraph)
        await self._add_entity_with_vector(object_token, source_paragraph=source_paragraph)
        rv_cfg = self.plugin.get_config("retrieval.relation_vectorization", {}) or {}
        if not isinstance(rv_cfg, dict):
            rv_cfg = {}
        write_vector = bool(rv_cfg.get("enabled", False)) and bool(rv_cfg.get("write_on_import", True))

        relation_service = getattr(self.plugin, "relation_write_service", None)
        if relation_service is not None:
            result = await relation_service.upsert_relation_with_vector(
                subject=subject_token,
                predicate=predicate_token,
                obj=object_token,
                confidence=1.0,
                source_paragraph=source_paragraph,
                write_vector=write_vector,
            )
            return result.hash_value

        rel_hash = self.plugin.metadata_store.add_relation(
            subject=subject_token,
            predicate=predicate_token,
            obj=object_token,
            source_paragraph=source_paragraph,
            confidence=1.0,
        )
        self.plugin.graph_store.add_edges([(subject_token, object_token)], relation_hashes=[rel_hash])
        try:
            self.plugin.metadata_store.set_relation_vector_state(rel_hash, "none")
        except Exception:
            pass
        return rel_hash

    async def _select_model(self) -> Any:
        models = llm_api.get_available_models()
        if not models:
            raise RuntimeError("没有可用 LLM 模型")

        config_model = str(self._cfg("advanced.extraction_model", "auto") or "auto").strip()
        if config_model.lower() != "auto" and config_model in models:
            return models[config_model]

        for task_name in [
            "lpmm_entity_extract",
            "lpmm_rdf_build",
            "replyer",
            "utils",
            "planner",
            "tool_use",
        ]:
            if task_name in models:
                return models[task_name]

        return models[next(iter(models))]

    async def _llm_call(self, prompt: str, model_config: Any) -> Dict[str, Any]:
        cfg = self._llm_retry_config()
        retries = int(cfg["retries"])
        task_name = llm_api.resolve_task_name_from_model_config(model_config)
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                result = await llm_api.generate(
                    llm_api.LLMServiceRequest(
                        task_name=task_name,
                        request_type="A_Memorix.WebImport",
                        prompt=prompt,
                        temperature=getattr(model_config, "temperature", None),
                        max_tokens=getattr(model_config, "max_tokens", None),
                    )
                )
                success = bool(result.success)
                response = str(result.completion.response or "")
                if not success or not response:
                    raise RuntimeError("LLM 生成失败")

                txt = str(response or "").strip()
                if "```" in txt:
                    txt = txt.split("```json")[-1].split("```")[0].strip()
                    if txt.startswith("json"):
                        txt = txt[4:].strip()

                try:
                    return _coerce_import_data_dict(json.loads(txt), context="LLM 抽取结果")
                except Exception:
                    s = txt.find("{")
                    e = txt.rfind("}")
                    if s >= 0 and e > s:
                        return _coerce_import_data_dict(json.loads(txt[s : e + 1]), context="LLM 抽取结果")
                    raise
            except Exception as err:
                last_error = err
                if attempt >= retries:
                    break
                delay = min(cfg["max_wait"], cfg["min_wait"] * (cfg["multiplier"] ** attempt))
                await asyncio.sleep(max(0.0, float(delay)))
        raise RuntimeError(f"LLM 抽取失败: {last_error}")

    def _parse_reference_time(self, value: Optional[str]) -> datetime:
        if not value:
            return datetime.now()
        text = str(value).strip()
        formats = [
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return datetime.now()

    async def _extract_chat_time_meta_with_llm(
        self,
        text: str,
        model_config: Any,
        *,
        reference_time: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not str(text or "").strip():
            return None
        ref_dt = self._parse_reference_time(reference_time)
        reference_now = ref_dt.strftime("%Y/%m/%d %H:%M")
        prompt = f"""You are a time extraction engine for chat logs.
Extract temporal information from the following chat paragraph.

Rules:
1. Use semantic understanding, not regex matching.
2. Convert relative expressions to absolute local datetime using reference_now.
3. If a time span exists, return event_time_start/event_time_end.
4. If only one point in time exists, return event_time.
5. If no reliable time info exists, keep all event_time fields null.
6. Return JSON only.

reference_now: {reference_now}
text:
{text}

JSON schema:
{{
  "event_time": null,
  "event_time_start": null,
  "event_time_end": null,
  "time_range": null,
  "time_granularity": null,
  "time_confidence": 0.0
}}
"""
        try:
            result = await self._llm_call(prompt, model_config)
        except Exception as e:
            logger.warning(f"chat_log 时间语义抽取失败: {e}")
            return None

        result = _coerce_import_data_dict(result, context="chat_log 时间抽取结果")
        raw_time_meta = {
            "event_time": result.get("event_time"),
            "event_time_start": result.get("event_time_start"),
            "event_time_end": result.get("event_time_end"),
            "time_range": result.get("time_range"),
            "time_granularity": result.get("time_granularity"),
            "time_confidence": result.get("time_confidence"),
        }
        try:
            normalized = normalize_time_meta(raw_time_meta)
        except Exception:
            return None
        has_effective = any(k in normalized for k in ("event_time", "event_time_start", "event_time_end"))
        if not has_effective:
            return None
        return normalized

    def _chunk_rescue(self, chunk: ProcessedChunk, filename: str) -> Optional[Any]:
        if chunk.type == StrategyKnowledgeType.QUOTE:
            return None
        if looks_like_quote_text(chunk.chunk.text):
            return QuoteStrategy(filename)
        return None

    def _instantiate_strategy(self, filename: str, strategy: ImportStrategy) -> Any:
        if strategy == ImportStrategy.FACTUAL:
            return FactualStrategy(filename)
        if strategy == ImportStrategy.QUOTE:
            return QuoteStrategy(filename)
        return NarrativeStrategy(filename)

    def _determine_strategy(self, filename: str, content: str, override: str, *, chat_log: bool = False) -> Any:
        strategy = select_import_strategy(
            content,
            override=override,
            chat_log=chat_log,
        )
        return self._instantiate_strategy(filename, strategy)

    async def _set_file_strategy(self, task_id: str, file_id: str, strategy: Any) -> None:
        if isinstance(strategy, str):
            strategy_type = strategy
        elif isinstance(strategy, NarrativeStrategy):
            strategy_type = "narrative"
        elif isinstance(strategy, FactualStrategy):
            strategy_type = "factual"
        elif isinstance(strategy, QuoteStrategy):
            strategy_type = "quote"
        else:
            strategy_type = "unknown"

        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            f = self._find_file(task, file_id)
            if not f:
                return
            f.detected_strategy_type = strategy_type
            f.updated_at = _now()
            task.updated_at = _now()

    async def _register_chunks(self, task_id: str, file_id: str, chunks: List[ProcessedChunk]) -> None:
        records = [
            ImportChunkRecord(
                chunk_id=chunk.chunk.chunk_id,
                index=index,
                chunk_type=chunk.type.value,
                status="queued",
                step="queued",
                progress=0.0,
                content_preview=str(chunk.chunk.text or "")[:120],
            )
            for index, chunk in enumerate(chunks)
        ]

        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            f = self._find_file(task, file_id)
            if not f:
                return
            f.chunks = records
            f.total_chunks = len(records)
            f.done_chunks = 0
            f.failed_chunks = 0
            f.cancelled_chunks = 0
            f.progress = 0.0 if records else 1.0
            f.updated_at = _now()
            self._recompute_task_progress(task)

