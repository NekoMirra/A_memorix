"""PageRank Python baseline 实现。

与 `crates/pagerank_rust/src/lib.rs` 的 Rust 实现做语义对照。
两者的核心循环结构完全一致，便于数值正确性验证。

邻接矩阵语义:
    adj[i][j] 表示 j -> i 的边权（i 接收 j 的链接）
    - 节点 i 的入度 = adj[i].sum()
    - 节点 j 的出度 = adj[:, j].sum()（列和）

PageRank 公式（支持 personalization 向量 p_orig）:
    scores[i] = (1 - damping) * p_orig[i]
              + damping * Σ_j (adj[i][j] / out_sum[j]) * scores[j]
              + damping * dangling_sum * p_orig[i]

其中 dangling_sum = Σ_{k: out_sum[k]=0} scores[k]
等价于 GraphStore scipy 路径的 (1 - sum) * p_orig 回注。
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def pagerank_python(
    adjacency: np.ndarray,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    personalization: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Power iteration PageRank，对照 Rust 实现的 Python 版本。

    算法语义与 `crates/pagerank_rust/src/lib.rs::pagerank` 完全等价。
    数值结果与 Rust 在 1e-9 量级一致。
    """
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            f"adjacency 必须是方形矩阵，得到 shape={adjacency.shape}"
        )
    n = adjacency.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    adj = adjacency.astype(np.float64)

    # 1) 出度 = 列和（每列 = 该节点指向其他节点的边权和）
    out_sum = adj.sum(axis=0)

    # 2) 个性化向量
    if personalization is None:
        p_orig = np.ones(n) / n
    else:
        p = personalization.astype(np.float64).copy()
        total = p.sum()
        if total > 0:
            p_orig = p / total
        else:
            p_orig = np.ones(n) / n

    # 3) 幂迭代
    scores = p_orig.copy()
    for _ in range(max_iter):
        # 处理 dangling 节点（出度为 0）：按 p_orig 分配
        dangling_sum = float(scores[out_sum == 0.0].sum())
        teleport_scale = (1.0 - damping) + damping * dangling_sum

        # acc[i] = Σ_j adj[i][j] / out_sum[j] * scores[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_out = np.where(out_sum > 0.0, 1.0 / out_sum, 0.0)
        weighted = inv_out * scores  # (n,)
        acc = adj @ weighted  # (n,) = adj (n,n) @ weighted (n,)

        next_scores = damping * acc + teleport_scale * p_orig

        # 收敛判断：L1 差值
        diff = float(np.abs(next_scores - scores).sum())
        scores = next_scores
        if diff < tol:
            break

    return scores
