"""GraphStore.compute_pagerank 与 Rust 内核集成对照。

直接运行:
    python crates/pagerank_rust/tests/test_graph_store_pagerank_rust.py
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "core").is_dir():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

# 提供最小 src.common.logger，避免依赖 MaiBot 宿主
if "src" not in sys.modules:
    src_mod = types.ModuleType("src")
    common_mod = types.ModuleType("src.common")
    logger_mod = types.ModuleType("src.common.logger")

    def get_logger(name: str):
        import logging

        return logging.getLogger(name)

    logger_mod.get_logger = get_logger
    sys.modules["src"] = src_mod
    sys.modules["src.common"] = common_mod
    sys.modules["src.common.logger"] = logger_mod

from core.storage.graph_store import GraphStore  # noqa: E402


def _scipy_only_pagerank(store: GraphStore, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-9):
    """绕过 Rust，直接走 scipy 路径（复制原算法）。"""
    from scipy.sparse import diags

    adj = store._adjacency.astype(np.float32)
    n = len(store._nodes)
    out_degrees = np.array(adj.sum(axis=1)).flatten()
    dangling = out_degrees == 0
    out_degrees_inv = np.zeros_like(out_degrees)
    out_degrees_inv[~dangling] = 1.0 / out_degrees[~dangling]
    D_inv = diags(out_degrees_inv)
    M = adj.T @ D_inv
    p = np.ones(n) / n
    p_orig = p.copy()
    for _ in range(max_iter):
        p_new = alpha * (M @ p) + (1 - alpha) * p_orig
        current_sum = p_new.sum()
        if current_sum < 1.0:
            p_new += (1.0 - current_sum) * p_orig
        if float(np.linalg.norm(p_new - p, 1)) < tol:
            p = p_new
            break
        p = p_new
    return {store._nodes[i]: float(v) for i, v in enumerate(p)}


def main() -> int:
    try:
        from a_memorix_kernel_rust import pagerank as _  # noqa: F401

        rust_ok = True
    except ImportError:
        rust_ok = False
        print("SKIP: Rust 内核未安装")
        return 0

    print(f"Rust available: {rust_ok}")

    store = GraphStore()
    # A->B, A->C, B->C, C->A
    store.add_edges(
        [("A", "B"), ("A", "C"), ("B", "C"), ("C", "A")],
        weights=[1.0, 1.0, 1.0, 1.0],
    )

    rust_path = store.compute_pagerank(personalization=None, alpha=0.85, max_iter=100, tol=1e-9)
    scipy_path = _scipy_only_pagerank(store, alpha=0.85, max_iter=100, tol=1e-9)

    nodes = sorted(set(rust_path) | set(scipy_path))
    diffs = [abs(rust_path[n] - scipy_path[n]) for n in nodes]
    max_diff = max(diffs) if diffs else 0.0
    l1 = sum(diffs)

    print("rust:", {k: round(v, 6) for k, v in sorted(rust_path.items())})
    print("scipy:", {k: round(v, 6) for k, v in sorted(scipy_path.items())})
    print(f"max_diff={max_diff:.3e} L1={l1:.3e}")

    # 浮点路径不同（dense power vs sparse scipy），放宽到 1e-5
    if max_diff < 1e-5 and l1 < 1e-5:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
