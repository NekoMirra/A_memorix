"""Rust 内核软依赖封装。

为 A_memorix 热点（PageRank 计算）提供可选的 Rust 加速路径。
按 A_memorix MODIFICATION_POLICY 的约定，Rust crate 的实现位于
`crates/pagerank_rust/`；本模块是 Python 侧的接入点。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.common.logger import get_logger

logger = get_logger("A_Memorix.PagerankKernel")

try:
    from a_memorix_kernel_rust import pagerank as _rust_pagerank

    _HAS_RUST_KERNEL = True
    logger.info("已加载 Rust PageRank 内核")
except ImportError:
    _HAS_RUST_KERNEL = False
    logger.debug("Rust PageRank 内核不可用，使用 Python/scipy 路径")


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
    if not _HAS_RUST_KERNEL:
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
        logger.debug("图规模过大 (n=%d)，跳过 Rust 路径", n)
        return None

    scores_list = _rust_pagerank(
        adjacency.astype(np.float64).tolist(),
        damping,
        max_iter,
        tol,
    )
    return np.asarray(scores_list, dtype=np.float64)