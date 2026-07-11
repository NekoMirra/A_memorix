"""Rust 内核软依赖封装。

为 A_memorix 热点（PageRank 计算）提供可选的 Rust 加速路径。
按 A_memorix MODIFICATION_POLICY 的约定，Rust crate 的实现位于
`crates/pagerank_rust/`；本模块是 Python 侧的接入点。
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from src.common.logger import get_logger

logger = get_logger("A_Memorix.PagerankKernel")

try:
    from a_memorix_kernel_rust import pagerank as _rust_pagerank
    from a_memorix_kernel_rust import pagerank_csr as _rust_pagerank_csr

    _HAS_RUST_KERNEL = True
    _HAS_RUST_CSR = True
    logger.info("已加载 Rust PageRank 内核 (dense + csr)")
except ImportError:
    _HAS_RUST_KERNEL = False
    _HAS_RUST_CSR = False
    _rust_pagerank = None
    _rust_pagerank_csr = None
    logger.debug("Rust PageRank 内核不可用，使用 Python/scipy 路径")
else:
    # 兼容旧 wheel：仅 dense 导出时 CSR 不可用
    if _rust_pagerank_csr is None:
        _HAS_RUST_CSR = False


def has_rust_kernel() -> bool:
    """Rust dense 内核是否可用。"""
    return bool(_HAS_RUST_KERNEL)


def has_rust_csr_kernel() -> bool:
    """Rust CSR 内核是否可用。"""
    return bool(_HAS_RUST_CSR)


def rust_pagerank_dense_uniform(
    adjacency: np.ndarray,
    damping: float,
    max_iter: int,
    tol: float,
) -> Optional[np.ndarray]:
    """Rust 实现的均匀 personalization PageRank（dense matrix）。

    返回与 `graph_store.compute_pagerank` 等价的 dense score 向量，
    或 None 表示 Rust 不可用（调用方应走 Python fallback）。

    仅当以下条件同时满足时才走 Rust 路径：
    - `a_memorix_kernel_rust` 包已安装
    - adjacency 是 dense float matrix（非稀疏）
    - matrix 维度 < 阈值（避免 dense 转换得不偿失）
    """
    if not _HAS_RUST_KERNEL or _rust_pagerank is None:
        return None
    # 当前 PoC 仅支持 dense matrix；sparse 走 Python fallback
    if not isinstance(adjacency, np.ndarray):
        return None
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        return None
    # Rust 接收 list of list；为避免 N>5000 时的 list 转换开销，
    # 仅在中小规模图上使用 Rust 路径
    n = adjacency.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n > 5000:
        logger.debug("图规模过大 (n=%d)，跳过 Rust dense 路径", n)
        return None

    scores_list = _rust_pagerank(
        adjacency.astype(np.float64).tolist(),
        damping,
        max_iter,
        tol,
    )
    return np.asarray(scores_list, dtype=np.float64)


def rust_pagerank_csr(
    adjacency_or_parts: Union[
        object,
        Tuple[Sequence[int], Sequence[int], Sequence[float], int],
    ],
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Optional[np.ndarray]:
    """Rust 实现的均匀 personalization PageRank（CSR / GraphStore 语义）。

    参数:
        adjacency_or_parts:
            - scipy.sparse csr_matrix / 可 `.tocsr()` 的稀疏矩阵；或
            - (indptr, indices, data, n) 元组
        damping / max_iter / tol: 与 GraphStore.compute_pagerank 一致

    邻接语义（与 GraphStore 一致）:
        adj[src, tgt] = src -> tgt，出度 = 行和

    返回:
        float64 score 向量，或 None（Rust 不可用 / 输入非法）
    """
    if not _HAS_RUST_CSR or _rust_pagerank_csr is None:
        return None

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    n: int

    if isinstance(adjacency_or_parts, tuple) and len(adjacency_or_parts) == 4:
        indptr_in, indices_in, data_in, n_in = adjacency_or_parts
        n = int(n_in)
        indptr = np.asarray(indptr_in, dtype=np.int64)
        indices = np.asarray(indices_in, dtype=np.int32)
        data = np.asarray(data_in, dtype=np.float64)
    else:
        mat = adjacency_or_parts
        # 避免硬依赖 scipy 类型注解；duck-type
        if hasattr(mat, "tocsr"):
            csr = mat.tocsr()
        else:
            return None
        if csr.ndim != 2 or csr.shape[0] != csr.shape[1]:
            return None
        n = int(csr.shape[0])
        # 确保标准 CSR 布局
        if not getattr(csr, "has_sorted_indices", True):
            csr.sort_indices()
        indptr = np.asarray(csr.indptr, dtype=np.int64)
        indices = np.asarray(csr.indices, dtype=np.int32)
        data = np.asarray(csr.data, dtype=np.float64)

    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if indptr.shape[0] != n + 1:
        return None
    if indices.shape[0] != data.shape[0]:
        return None

    scores_list = _rust_pagerank_csr(
        n,
        indptr.tolist(),
        indices.tolist(),
        data.tolist(),
        damping,
        max_iter,
        tol,
    )
    return np.asarray(scores_list, dtype=np.float64)
