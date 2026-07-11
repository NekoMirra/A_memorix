from __future__ import annotations

import asyncio
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Sequence

from src.common.logger import get_logger
from src.config.config import global_config
from src.services.llm_service import LLMServiceClient

from ...paths import default_data_dir, resolve_repo_path
from ..embedding import create_embedding_api_adapter
from ..retrieval import SparseBM25Config, SparseBM25Index
from ..storage import GraphStore, MetadataStore, QuantizationType, SparseMatrixFormat, VectorStore
from ..utils.aggregate_query_service import AggregateQueryService
from ..utils.episode_retrieval_service import EpisodeRetrievalService
from ..utils.episode_segmentation_service import EpisodeSegmentationService
from ..utils.episode_service import EpisodeService
from ..utils.person_profile_service import PersonProfileService
from ..utils.relation_write_service import RelationWriteService
from ..utils.retrieval_tuning_manager import RetrievalTuningManager
from ..utils.runtime_self_check import run_embedding_runtime_self_check
from ..utils.summary_importer import SummaryImporter
from ..utils.web_import_manager import ImportTaskManager
from .kernel_admin_mixin import KernelAdminMixin
from .kernel_delete_mixin import KernelDeleteMixin
from .kernel_feedback_mixin import KernelFeedbackMixin
from .kernel_graph_mixin import KernelGraphMixin
from .kernel_helpers_mixin import KernelHelpersMixin
from .kernel_ingest_mixin import KernelIngestMixin
from .kernel_maintenance_mixin import KernelMaintenanceMixin
from .kernel_profile_mixin import KernelProfileMixin
from .kernel_search_mixin import KernelSearchMixin
from .kernel_types import KernelSearchRequest, NormalizedSearchTimeWindow
from .search_runtime_initializer import SearchRuntimeBundle, build_search_runtime

# Public re-export compatibility
_NormalizedSearchTimeWindow = NormalizedSearchTimeWindow

logger = get_logger("A_Memorix.SDKMemoryKernel")


class _KernelRuntimeFacade:
    def __init__(self, kernel: "SDKMemoryKernel") -> None:
        self._kernel = kernel
        self.config = kernel.config
        self._plugin_config = kernel.config
        self._runtime_self_check_report: Dict[str, Any] = {}

    def get_config(self, key: str, default: Any = None) -> Any:
        return self._kernel._cfg(key, default)

    def is_runtime_ready(self) -> bool:
        return self._kernel.is_runtime_ready()

    def is_chat_enabled(self, stream_id: str, group_id: str | None = None, user_id: str | None = None) -> bool:
        return self._kernel.is_chat_enabled(stream_id=stream_id, group_id=group_id, user_id=user_id)

    async def reinforce_access(self, relation_hashes: Sequence[str]) -> None:
        if self._kernel.metadata_store is None:
            return
        hashes = [str(item or "").strip() for item in relation_hashes if str(item or "").strip()]
        if not hashes:
            return
        self._kernel.metadata_store.reinforce_relations(hashes)
        self._kernel._last_maintenance_at = time.time()

    async def execute_request_with_dedup(
        self,
        request_key: str,
        executor: Callable[[], Coroutine[Any, Any, Dict[str, Any]]],
    ) -> tuple[bool, Dict[str, Any]]:
        return await self._kernel.execute_request_with_dedup(request_key, executor)

    @property
    def vector_store(self) -> Optional[VectorStore]:
        return self._kernel.vector_store

    @property
    def graph_store(self) -> Optional[GraphStore]:
        return self._kernel.graph_store

    @property
    def metadata_store(self) -> Optional[MetadataStore]:
        return self._kernel.metadata_store

    @property
    def embedding_manager(self):
        return self._kernel.embedding_manager

    @property
    def sparse_index(self):
        return self._kernel.sparse_index

    @property
    def relation_write_service(self) -> Optional[RelationWriteService]:
        return self._kernel.relation_write_service

    def is_embedding_degraded(self) -> bool:
        return self._kernel._is_embedding_degraded()

    def allow_metadata_only_write(self) -> bool:
        return self._kernel._allow_metadata_only_write()

    async def write_paragraph_vector_or_enqueue(
        self,
        *,
        paragraph_hash: str,
        content: str,
        context: str = "",
    ) -> Dict[str, Any]:
        return await self._kernel._write_paragraph_vector_or_enqueue(
            paragraph_hash=paragraph_hash,
            content=content,
            context=context,
        )

    def enqueue_paragraph_vector_backfill(
        self,
        paragraph_hash: str,
        *,
        error: str = "",
    ) -> None:
        self._kernel._enqueue_paragraph_vector_backfill(paragraph_hash, error=error)


class SDKMemoryKernel(
    KernelIngestMixin,
    KernelSearchMixin,
    KernelProfileMixin,
    KernelMaintenanceMixin,
    KernelAdminMixin,
    KernelFeedbackMixin,
    KernelGraphMixin,
    KernelDeleteMixin,
    KernelHelpersMixin,
):
    def __init__(self, *, plugin_root: Path, config: Optional[Dict[str, Any]] = None) -> None:
        self.plugin_root = Path(plugin_root).resolve()
        self.config = config or {}
        storage_cfg = self._cfg("storage", {}) or {}
        data_dir = str(storage_cfg.get("data_dir", "./data") or "./data")
        self.data_dir = resolve_repo_path(data_dir, fallback=default_data_dir())
        self.embedding_dimension = max(1, int(self._cfg("embedding.dimension", 1024)))
        self.relation_vectors_enabled = bool(self._cfg("retrieval.relation_vectorization.enabled", False))

        self.embedding_manager = None
        self.vector_store: Optional[VectorStore] = None
        self.graph_store: Optional[GraphStore] = None
        self.metadata_store: Optional[MetadataStore] = None
        self.relation_write_service: Optional[RelationWriteService] = None
        self.sparse_index: Optional[SparseBM25Index] = None
        self.retriever = None
        self.threshold_filter = None
        self.episode_retriever: Optional[EpisodeRetrievalService] = None
        self.aggregate_query_service: Optional[AggregateQueryService] = None
        self.person_profile_service: Optional[PersonProfileService] = None
        self.episode_segmentation_service: Optional[EpisodeSegmentationService] = None
        self.episode_service: Optional[EpisodeService] = None
        self.summary_importer: Optional[SummaryImporter] = None
        self.import_task_manager: Optional[ImportTaskManager] = None
        self.retrieval_tuning_manager: Optional[RetrievalTuningManager] = None
        self._runtime_bundle: Optional[SearchRuntimeBundle] = None
        self._runtime_facade = _KernelRuntimeFacade(self)
        self._initialized = False
        self._last_maintenance_at: Optional[float] = None
        self._request_dedup_tasks: Dict[str, asyncio.Task] = {}
        self._background_tasks: Dict[str, asyncio.Task] = {}
        self._background_lock = asyncio.Lock()
        self._background_stopping = False
        self._active_person_timestamps: Dict[str, float] = {}
        self._embedding_degraded: Dict[str, Any] = {
            "active": False,
            "reason": "",
            "since": None,
            "last_check": None,
        }
        self._feedback_classifier: Optional[LLMServiceClient] = None

    def _cfg(self, key: str, default: Any = None) -> Any:
        current: Any = self.config
        if key in {"storage", "embedding", "retrieval", "graph", "episode", "web", "advanced", "threshold", "summarization"} and isinstance(current, dict):
            return current.get(key, default)
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def _set_cfg(self, key: str, value: Any) -> None:
        current: Dict[str, Any] = self.config
        parts = [part for part in str(key or "").split(".") if part]
        if not parts:
            return
        for part in parts[:-1]:
            next_value = current.get(part)
            if not isinstance(next_value, dict):
                next_value = {}
                current[part] = next_value
            current = next_value
        current[parts[-1]] = value

    def _build_runtime_config(self) -> Dict[str, Any]:
        runtime_config = dict(self.config)
        runtime_config.update(
            {
                "vector_store": self.vector_store,
                "graph_store": self.graph_store,
                "metadata_store": self.metadata_store,
                "embedding_manager": self.embedding_manager,
                "sparse_index": self.sparse_index,
                "relation_write_service": self.relation_write_service,
                "plugin_instance": self._runtime_facade,
            }
        )
        return runtime_config

    def is_runtime_ready(self) -> bool:
        return bool(
            self._initialized
            and self.vector_store is not None
            and self.graph_store is not None
            and self.metadata_store is not None
            and self.embedding_manager is not None
            and self.retriever is not None
        )

    def is_chat_enabled(self, stream_id: str, group_id: str | None = None, user_id: str | None = None) -> bool:
        filter_config = self._cfg("filter", {}) or {}
        if not isinstance(filter_config, dict) or not filter_config:
            return True

        if not bool(filter_config.get("enabled", True)):
            return True

        mode = str(filter_config.get("mode", "blacklist") or "blacklist").strip().lower()
        patterns = filter_config.get("chats") or []
        if not isinstance(patterns, list):
            patterns = []

        if not patterns:
            return mode == "blacklist"

        stream_token = str(stream_id or "").strip()
        group_token = str(group_id or "").strip()
        user_token = str(user_id or "").strip()
        candidates = {token for token in (stream_token, group_token, user_token) if token}

        matched = False
        for raw_pattern in patterns:
            pattern = str(raw_pattern or "").strip()
            if not pattern:
                continue
            if ":" in pattern:
                prefix, value = pattern.split(":", 1)
                prefix = prefix.strip().lower()
                value = value.strip()
                if prefix == "group" and value and value == group_token:
                    matched = True
                elif prefix in {"user", "private"} and value and value == user_token:
                    matched = True
                elif prefix == "stream" and value and value == stream_token:
                    matched = True
            elif pattern in candidates:
                matched = True

            if matched:
                break

        if mode == "blacklist":
            return not matched
        return matched

    def _is_chat_filtered(
        self,
        *,
        respect_filter: bool,
        stream_id: str = "",
        group_id: str = "",
        user_id: str = "",
    ) -> bool:
        if not bool(respect_filter):
            return False

        stream_token = str(stream_id or "").strip()
        group_token = str(group_id or "").strip()
        user_token = str(user_id or "").strip()
        if not (stream_token or group_token or user_token):
            return False
        return not self.is_chat_enabled(stream_token, group_token, user_token)

    def _stored_vector_dimension(self) -> Optional[int]:
        meta_path = self.data_dir / "vectors" / "vectors_metadata.pkl"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, "rb") as handle:
                meta = pickle.load(handle)
        except Exception as exc:
            logger.warning(f"读取向量元数据失败，将回退到 runtime self-check: {exc}")
            return None
        try:
            value = int(meta.get("dimension") or 0)
        except Exception:
            return None
        return value if value > 0 else None

    def _vector_mismatch_error(self, *, stored_dimension: int, detected_dimension: int) -> str:
        return (
            "检测到现有向量库与当前 embedding 输出维度不一致："
            f"stored={stored_dimension}, encoded={detected_dimension}。"
            " 当前版本不会兼容 hash 时代或其他维度的旧向量，请改回原 embedding 配置，"
            "或执行重嵌入/重建向量。"
        )

    def _embedding_fallback_enabled(self) -> bool:
        return bool(self._cfg("embedding.fallback.enabled", True))

    def _allow_metadata_only_write(self) -> bool:
        return bool(self._cfg("embedding.fallback.allow_metadata_only_write", True))

    def _embedding_probe_interval_seconds(self) -> float:
        return max(10.0, float(self._cfg("embedding.fallback.probe_interval_seconds", 180) or 180))

    def _paragraph_vector_backfill_enabled(self) -> bool:
        return bool(self._cfg("embedding.paragraph_vector_backfill.enabled", True))

    def _paragraph_vector_backfill_interval_seconds(self) -> float:
        return max(10.0, float(self._cfg("embedding.paragraph_vector_backfill.interval_seconds", 60) or 60))

    def _paragraph_vector_backfill_batch_size(self) -> int:
        return max(1, int(self._cfg("embedding.paragraph_vector_backfill.batch_size", 64) or 64))

    def _paragraph_vector_backfill_max_retry(self) -> int:
        return max(1, int(self._cfg("embedding.paragraph_vector_backfill.max_retry", 5) or 5))

    def _is_embedding_degraded(self) -> bool:
        return bool(self._embedding_degraded.get("active", False))

    def _embedding_degraded_snapshot(self) -> Dict[str, Any]:
        return {
            "active": bool(self._embedding_degraded.get("active", False)),
            "reason": str(self._embedding_degraded.get("reason", "") or ""),
            "since": self._embedding_degraded.get("since"),
            "last_check": self._embedding_degraded.get("last_check"),
        }

    def _set_embedding_degraded(self, *, active: bool, reason: str = "", checked_at: Optional[float] = None) -> None:
        now = float(checked_at or time.time())
        prev = self._embedding_degraded_snapshot()
        if active:
            since = prev.get("since") if bool(prev.get("active", False)) else now
            self._embedding_degraded = {
                "active": True,
                "reason": str(reason or "").strip(),
                "since": since,
                "last_check": now,
            }
        else:
            self._embedding_degraded = {
                "active": False,
                "reason": "",
                "since": None,
                "last_check": now,
            }
        if bool(prev.get("active", False)) != bool(active):
            if active:
                logger.warning(
                    "embedding 进入降级态，将启用 sparse-only 与 metadata-only 写入回退: "
                    f"reason={self._embedding_degraded.get('reason', '')}"
                )
            else:
                logger.info("embedding 已恢复，退出降级态")
        self._apply_runtime_sparse_mode()

    def _apply_runtime_sparse_mode(self) -> None:
        retriever = self.retriever
        if retriever is None:
            return
        setter = getattr(retriever, "set_runtime_sparse_only", None)
        if not callable(setter):
            return
        try:
            setter(self._is_embedding_degraded())
        except Exception as exc:
            logger.warning(f"设置 retriever sparse-only 运行时状态失败: {exc}")

    async def _refresh_runtime_self_check(self, *, sample_text: str = "A_Memorix runtime self check") -> Dict[str, Any]:
        report = await run_embedding_runtime_self_check(
            config=self._build_runtime_config(),
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            sample_text=sample_text,
        )
        self._runtime_facade._runtime_self_check_report = dict(report)
        checked_at = float(report.get("checked_at") or time.time())
        self._embedding_degraded["last_check"] = checked_at
        return report

    def _enqueue_paragraph_vector_backfill(self, paragraph_hash: str, *, error: str = "") -> None:
        if self.metadata_store is None:
            return
        try:
            self.metadata_store.enqueue_paragraph_vector_backfill(
                paragraph_hash,
                error=str(error or ""),
            )
        except Exception as exc:
            logger.warning(f"登记 paragraph 向量回填任务失败: {exc}")

    async def _write_paragraph_vector_or_enqueue(
        self,
        *,
        paragraph_hash: str,
        content: str,
        context: str = "",
    ) -> Dict[str, Any]:
        token = str(paragraph_hash or "").strip()
        text = str(content or "").strip()
        if not token or not text:
            return {
                "success": False,
                "vector_written": False,
                "queued": False,
                "warning": "",
                "detail": "invalid_paragraph_input",
            }

        allow_metadata_only = self._allow_metadata_only_write()

        if self.vector_store is None or self.embedding_manager is None:
            if not allow_metadata_only:
                raise RuntimeError("向量写入依赖未初始化")
            self._enqueue_paragraph_vector_backfill(token, error="vector_runtime_components_missing")
            return {
                "success": True,
                "vector_written": False,
                "queued": True,
                "warning": "vector_degraded_write",
                "detail": "vector_runtime_components_missing",
            }

        if self._is_embedding_degraded():
            if not allow_metadata_only:
                raise RuntimeError("embedding 处于降级态，metadata-only 写入已禁用")
            self._enqueue_paragraph_vector_backfill(token, error="embedding_degraded")
            return {
                "success": True,
                "vector_written": False,
                "queued": True,
                "warning": "vector_degraded_write",
                "detail": "embedding_degraded",
            }

        if token in self.vector_store:
            return {
                "success": True,
                "vector_written": True,
                "queued": False,
                "warning": "",
                "detail": "vector_already_exists",
            }

        try:
            embedding = await self.embedding_manager.encode(text)
            if getattr(embedding, "ndim", 1) == 1:
                embedding = embedding.reshape(1, -1)
            self.vector_store.add(vectors=embedding, ids=[token])
            return {
                "success": True,
                "vector_written": True,
                "queued": False,
                "warning": "",
                "detail": "",
            }
        except Exception as exc:
            error_text = str(exc)
            if self._embedding_fallback_enabled():
                self._set_embedding_degraded(active=True, reason=error_text[:500], checked_at=time.time())
            if not allow_metadata_only:
                raise
            self._enqueue_paragraph_vector_backfill(token, error=error_text)
            return {
                "success": True,
                "vector_written": False,
                "queued": True,
                "warning": "vector_degraded_write",
                "detail": f"{str(context or 'paragraph')} vector write failed: {error_text}",
            }

    def _paragraph_vector_backfill_counts(self) -> Dict[str, int]:
        if self.metadata_store is None:
            return {"pending": 0, "running": 0, "failed": 0, "done": 0}
        try:
            return self.metadata_store.get_paragraph_vector_backfill_status_counts()
        except Exception as exc:
            logger.warning(f"读取 paragraph 回填状态失败: {exc}")
            return {"pending": 0, "running": 0, "failed": 0, "done": 0}

    async def _run_paragraph_backfill_once(
        self,
        *,
        limit: Optional[int] = None,
        max_retry: Optional[int] = None,
        trigger: str = "manual",
    ) -> Dict[str, Any]:
        if self.metadata_store is None or self.vector_store is None or self.embedding_manager is None:
            return {"success": False, "processed": 0, "done": 0, "failed": 0, "trigger": trigger}
        if self._is_embedding_degraded():
            return {
                "success": False,
                "processed": 0,
                "done": 0,
                "failed": 0,
                "trigger": trigger,
                "detail": "embedding_degraded",
            }

        safe_limit = max(1, int(limit or self._paragraph_vector_backfill_batch_size()))
        safe_retry = max(1, int(max_retry or self._paragraph_vector_backfill_max_retry()))
        rows = self.metadata_store.fetch_paragraph_vector_backfill_batch(limit=safe_limit, max_retry=safe_retry)
        if not rows:
            return {"success": True, "processed": 0, "done": 0, "failed": 0, "trigger": trigger}

        pending_hashes = [
            str(row.get("paragraph_hash", "") or "").strip()
            for row in rows
            if str(row.get("paragraph_hash", "") or "").strip()
        ]
        if pending_hashes:
            self.metadata_store.mark_paragraph_vector_backfill_running(pending_hashes)

        done_hashes: List[str] = []
        failed_count = 0
        for row in rows:
            paragraph_hash = str(row.get("paragraph_hash", "") or "").strip()
            if not paragraph_hash:
                continue
            if paragraph_hash in self.vector_store:
                done_hashes.append(paragraph_hash)
                continue
            paragraph = self.metadata_store.get_paragraph(paragraph_hash)
            if paragraph is None:
                done_hashes.append(paragraph_hash)
                continue
            content = str(paragraph.get("content", "") or "").strip()
            if not content:
                done_hashes.append(paragraph_hash)
                continue
            try:
                embedding = await self.embedding_manager.encode(content)
                if getattr(embedding, "ndim", 1) == 1:
                    embedding = embedding.reshape(1, -1)
                self.vector_store.add(vectors=embedding, ids=[paragraph_hash])
                done_hashes.append(paragraph_hash)
            except Exception as exc:
                failed_count += 1
                self.metadata_store.mark_paragraph_vector_backfill_failed(paragraph_hash, str(exc))
                if self._embedding_fallback_enabled():
                    self._set_embedding_degraded(active=True, reason=str(exc)[:500], checked_at=time.time())

        if done_hashes:
            self.metadata_store.mark_paragraph_vector_backfill_done(done_hashes)
            self._persist()

        return {
            "success": failed_count == 0,
            "processed": len(done_hashes) + failed_count,
            "done": len(done_hashes),
            "failed": failed_count,
            "trigger": trigger,
        }

    async def _recover_embedding_once(self, *, sample_text: str = "A_Memorix runtime self check") -> Dict[str, Any]:
        report = await self._refresh_runtime_self_check(sample_text=sample_text)
        checked_at = float(report.get("checked_at") or time.time())
        ok = bool(report.get("ok", False))
        if ok:
            self._set_embedding_degraded(active=False, checked_at=checked_at)
            backfill_result: Dict[str, Any] = {}
            if self._paragraph_vector_backfill_enabled():
                backfill_result = await self._run_paragraph_backfill_once(
                    limit=self._paragraph_vector_backfill_batch_size(),
                    max_retry=self._paragraph_vector_backfill_max_retry(),
                    trigger="embedding_recovered",
                )
            return {
                "success": True,
                "recovered": True,
                "report": report,
                "backfill": backfill_result,
            }

        reason = str(report.get("message", "runtime self-check failed") or "runtime self-check failed")
        if self._embedding_fallback_enabled():
            self._set_embedding_degraded(active=True, reason=reason, checked_at=checked_at)
            return {
                "success": False,
                "recovered": False,
                "report": report,
                "detail": "still_degraded",
            }
        return {
            "success": False,
            "recovered": False,
            "report": report,
            "detail": "fallback_disabled",
        }

    async def initialize(self) -> None:
        if self._initialized:
            self._apply_runtime_sparse_mode()
            await self._start_background_tasks()
            return

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_manager = create_embedding_api_adapter(
            batch_size=int(self._cfg("embedding.batch_size", 32)),
            max_concurrent=int(self._cfg("embedding.max_concurrent", 5)),
            default_dimension=self.embedding_dimension,
            enable_cache=bool(self._cfg("embedding.enable_cache", False)),
            model_name=str(self._cfg("embedding.model_name", "auto") or "auto"),
            retry_config=self._cfg("embedding.retry", {}) or {},
        )
        dimension_detection_task = asyncio.create_task(
            asyncio.to_thread(lambda: asyncio.run(self.embedding_manager._detect_dimension()))
        )
        await asyncio.sleep(0)
        stored_dimension = self._stored_vector_dimension()
        provisional_dimension = stored_dimension or self.embedding_dimension

        matrix_format = str(self._cfg("graph.sparse_matrix_format", "csr") or "csr").strip().lower()
        graph_format = SparseMatrixFormat.CSC if matrix_format == "csc" else SparseMatrixFormat.CSR

        self.vector_store = VectorStore(
            dimension=provisional_dimension,
            quantization_type=QuantizationType.INT8,
            data_dir=self.data_dir / "vectors",
        )
        self.graph_store = GraphStore(matrix_format=graph_format, data_dir=self.data_dir / "graph")
        self.metadata_store = MetadataStore(data_dir=self.data_dir / "metadata")
        self.metadata_store.connect()

        vector_store_loaded = False
        if stored_dimension is not None and self.vector_store.has_data():
            self.vector_store.load()
            self.vector_store.warmup_index(force_train=True)
            vector_store_loaded = True
        if self.graph_store.has_data():
            self.graph_store.load()

        sparse_cfg_raw = self._cfg("retrieval.sparse", {}) or {}
        try:
            sparse_cfg = SparseBM25Config(**sparse_cfg_raw)
        except Exception as exc:
            logger.warning(f"sparse 配置非法，回退默认: {exc}")
            sparse_cfg = SparseBM25Config()
        self.sparse_index = SparseBM25Index(metadata_store=self.metadata_store, config=sparse_cfg)
        if getattr(self.sparse_index.config, "enabled", False):
            self.sparse_index.ensure_loaded()

        try:
            detected_dimension = int(await dimension_detection_task)
        except Exception:
            if not dimension_detection_task.done():
                dimension_detection_task.cancel()
            raise
        self.embedding_dimension = detected_dimension

        if stored_dimension is not None and stored_dimension != detected_dimension:
            raise RuntimeError(
                self._vector_mismatch_error(
                    stored_dimension=stored_dimension,
                    detected_dimension=detected_dimension,
                )
            )

        if self.vector_store.dimension != detected_dimension:
            self.vector_store = VectorStore(
                dimension=detected_dimension,
                quantization_type=QuantizationType.INT8,
                data_dir=self.data_dir / "vectors",
            )

        if not vector_store_loaded and self.vector_store.has_data():
            self.vector_store.load()
            self.vector_store.warmup_index(force_train=True)

        self.relation_write_service = RelationWriteService(
            metadata_store=self.metadata_store,
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
        )

        runtime_config = self._build_runtime_config()
        self._runtime_bundle = build_search_runtime(
            plugin_config=runtime_config,
            logger_obj=logger,
            owner_tag="sdk_kernel",
            log_prefix="[sdk]",
        )
        if not self._runtime_bundle.ready:
            raise RuntimeError(self._runtime_bundle.error or "检索运行时初始化失败")

        self.retriever = self._runtime_bundle.retriever
        self.threshold_filter = self._runtime_bundle.threshold_filter
        self.sparse_index = self._runtime_bundle.sparse_index or self.sparse_index
        self._apply_runtime_sparse_mode()

        runtime_config = self._build_runtime_config()
        self.episode_retriever = EpisodeRetrievalService(metadata_store=self.metadata_store, retriever=self.retriever)
        self.aggregate_query_service = AggregateQueryService(plugin_config=runtime_config)
        self.person_profile_service = PersonProfileService(
            metadata_store=self.metadata_store,
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            sparse_index=self.sparse_index,
            plugin_config=runtime_config,
            retriever=self.retriever,
        )
        self.episode_segmentation_service = EpisodeSegmentationService(plugin_config=runtime_config)
        self.episode_service = EpisodeService(
            metadata_store=self.metadata_store,
            plugin_config=runtime_config,
            segmentation_service=self.episode_segmentation_service,
        )
        self.summary_importer = SummaryImporter(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            metadata_store=self.metadata_store,
            embedding_manager=self.embedding_manager,
            plugin_config=runtime_config,
        )
        self.import_task_manager = ImportTaskManager(self._runtime_facade)
        self.retrieval_tuning_manager = RetrievalTuningManager(
            self._runtime_facade,
            import_write_blocked_provider=self.import_task_manager.is_write_blocked,
        )

        report = await self._refresh_runtime_self_check(sample_text="A_Memorix runtime self check")
        if not bool(report.get("ok", False)):
            message = str(report.get("message", "runtime self-check failed") or "runtime self-check failed")
            checked_at = float(report.get("checked_at") or time.time())
            if self._embedding_fallback_enabled():
                self._set_embedding_degraded(active=True, reason=message, checked_at=checked_at)
            else:
                raise RuntimeError(f"{message}；请改回原 embedding 配置，或执行重嵌入/重建向量。")
        else:
            self._set_embedding_degraded(active=False, checked_at=float(report.get("checked_at") or time.time()))

        self._initialized = True
        await self._start_background_tasks()

    async def shutdown(self) -> None:
        await self._stop_background_tasks()
        if self.import_task_manager is not None:
            try:
                await self.import_task_manager.shutdown()
            except Exception as exc:
                logger.warning(f"关闭导入任务管理器失败: {exc}")
        if self.retrieval_tuning_manager is not None:
            try:
                await self.retrieval_tuning_manager.shutdown()
            except Exception as exc:
                logger.warning(f"关闭调优任务管理器失败: {exc}")
        self.close()

    def close(self) -> None:
        try:
            self._persist()
        finally:
            if self.metadata_store is not None:
                self.metadata_store.close()
            self._initialized = False
            self._request_dedup_tasks.clear()
            self._runtime_facade._runtime_self_check_report = {}
            self._background_tasks.clear()
            self._active_person_timestamps.clear()
            self._embedding_degraded = {
                "active": False,
                "reason": "",
                "since": None,
                "last_check": None,
            }

    async def execute_request_with_dedup(
        self,
        request_key: str,
        executor: Callable[[], Coroutine[Any, Any, Dict[str, Any]]],
    ) -> tuple[bool, Dict[str, Any]]:
        token = str(request_key or "").strip()
        if not token:
            return False, await executor()

        existing = self._request_dedup_tasks.get(token)
        if existing is not None:
            return True, await existing

        task = asyncio.create_task(executor())
        self._request_dedup_tasks[token] = task
        try:
            payload = await task
            return False, payload
        finally:
            current = self._request_dedup_tasks.get(token)
            if current is task:
                self._request_dedup_tasks.pop(token, None)

    def _persist(self) -> None:
        if self.vector_store is not None:
            self.vector_store.save()
        if self.graph_store is not None:
            self.graph_store.save()
        if self.sparse_index is not None and getattr(self.sparse_index.config, "enabled", False):
            self.sparse_index.ensure_loaded()

    async def _start_background_tasks(self) -> None:
        async with self._background_lock:
            self._background_stopping = False
            self._ensure_background_task("auto_save", self._auto_save_loop)
            self._ensure_background_task("episode_pending", self._episode_pending_loop)
            self._ensure_background_task("embedding_probe", self._embedding_probe_loop)
            self._ensure_background_task("paragraph_vector_backfill", self._paragraph_vector_backfill_loop)
            self._ensure_background_task("memory_maintenance", self._memory_maintenance_loop)
            self._ensure_background_task("person_profile_refresh", self._person_profile_refresh_loop)
            self._ensure_background_task("feedback_correction", self._feedback_correction_loop)
            self._ensure_background_task("feedback_correction_reconcile", self._feedback_correction_reconcile_loop)

    def _ensure_background_task(
        self,
        name: str,
        factory: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        task = self._background_tasks.get(name)
        if task is not None and not task.done():
            return
        self._background_tasks[name] = asyncio.create_task(factory(), name=f"A_Memorix.{name}")

    async def _stop_background_tasks(self) -> None:
        async with self._background_lock:
            self._background_stopping = True
            tasks = [task for task in self._background_tasks.values() if task is not None and not task.done()]
            for task in tasks:
                task.cancel()
            for task in tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    logger.warning(f"后台任务退出异常: {exc}")
            self._background_tasks.clear()

    async def _auto_save_loop(self) -> None:
        try:
            while not self._background_stopping:
                interval_minutes = max(1.0, float(self._cfg("advanced.auto_save_interval_minutes", 5) or 5))
                await asyncio.sleep(interval_minutes * 60.0)
                if self._background_stopping:
                    break
                if bool(self._cfg("advanced.enable_auto_save", True)):
                    self._persist()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"auto_save loop 异常: {exc}")

    async def _episode_pending_loop(self) -> None:
        try:
            while not self._background_stopping:
                await asyncio.sleep(60.0)
                if self._background_stopping:
                    break
                if not bool(self._cfg("episode.enabled", True)):
                    continue
                if not bool(self._cfg("episode.generation_enabled", True)):
                    continue
                await self.process_episode_pending_batch(
                    limit=max(1, int(self._cfg("episode.pending_batch_size", 20) or 20)),
                    max_retry=max(1, int(self._cfg("episode.pending_max_retry", 3) or 3)),
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"episode_pending loop 异常: {exc}")

    async def _embedding_probe_loop(self) -> None:
        try:
            while not self._background_stopping:
                await asyncio.sleep(self._embedding_probe_interval_seconds())
                if self._background_stopping:
                    break
                if not self._embedding_fallback_enabled():
                    continue
                if not self._is_embedding_degraded():
                    continue
                try:
                    await self._recover_embedding_once()
                except Exception as exc:
                    logger.warning(f"embedding 恢复探测失败: {exc}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"embedding_probe loop 异常: {exc}")

    async def _paragraph_vector_backfill_loop(self) -> None:
        try:
            while not self._background_stopping:
                await asyncio.sleep(self._paragraph_vector_backfill_interval_seconds())
                if self._background_stopping:
                    break
                if not self._paragraph_vector_backfill_enabled():
                    continue
                if self._is_embedding_degraded():
                    continue
                await self._run_paragraph_backfill_once(
                    limit=self._paragraph_vector_backfill_batch_size(),
                    max_retry=self._paragraph_vector_backfill_max_retry(),
                    trigger="loop",
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"paragraph_vector_backfill loop 异常: {exc}")

    async def _person_profile_refresh_loop(self) -> None:
        try:
            while not self._background_stopping:
                interval_minutes = max(1.0, float(self._cfg("person_profile.refresh_interval_minutes", 30) or 30))
                await asyncio.sleep(max(60.0, interval_minutes * 60.0))
                if self._background_stopping:
                    break
                if not bool(self._cfg("person_profile.enabled", True)):
                    continue
                active_window_hours = max(1.0, float(self._cfg("person_profile.active_window_hours", 72.0) or 72.0))
                max_refresh = max(1, int(self._cfg("person_profile.max_refresh_per_cycle", 50) or 50))
                cutoff = time.time() - active_window_hours * 3600.0
                candidates = [
                    person_id
                    for person_id, seen_at in sorted(
                        self._active_person_timestamps.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                    if seen_at >= cutoff
                ][:max_refresh]
                for person_id in candidates:
                    try:
                        await self.refresh_person_profile(person_id, limit=max(4, int(self._cfg("person_profile.top_k_evidence", 12) or 12)), mark_active=False)
                    except Exception as exc:
                        logger.warning(f"刷新人物画像失败: {exc}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"person_profile_refresh loop 异常: {exc}")
