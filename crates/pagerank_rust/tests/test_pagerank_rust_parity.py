"""PageRank Python vs Rust 数值对照测试。

不依赖 pytest，直接 `python tests/test_pagerank_rust_parity.py` 运行。
输出 PASS/FAIL 与详细数值。
"""
from __future__ import annotations

import os
import sys
import importlib.util
import random
from pathlib import Path

import numpy as np

# 让脚本可作为独立脚本运行
HERE = Path(__file__).resolve().parent
# 向上找到 A_memorix 仓根（含 core/ 目录的位置）
REPO_ROOT = HERE
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "core").is_dir():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

# 独立加载 pagerank_python_baseline（避免触发 core/__init__.py 整链）
_baseline_spec = importlib.util.spec_from_file_location(
    "pagerank_python_baseline",
    REPO_ROOT / "core" / "retrieval" / "pagerank_python_baseline.py",
)
_baseline_mod = importlib.util.module_from_spec(_baseline_spec)
_baseline_spec.loader.exec_module(_baseline_mod)
pagerank_python = _baseline_mod.pagerank_python

# 尝试导入 Rust 内核
try:
    from a_memorix_kernel_rust import pagerank as rust_pagerank

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    rust_pagerank = None


def build_random_graph(n: int, density: float, seed: int) -> np.ndarray:
    """构建 n 节点稀疏随机图，返回 dense 邻接矩阵（行=入边，列=出边）。"""
    rng = random.Random(seed)
    adj = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            if rng.random() < density:
                adj[i][j] = rng.random()  # j -> i
    # 保证没有全零列（防止节点孤立但保留 dangling 测试）
    return adj


def compare_one(name: str, adj: np.ndarray, damping: float, max_iter: int, tol: float) -> bool:
    """对一组参数跑 Python 和 Rust，比较输出。"""
    py_scores = pagerank_python(adj, damping=damping, max_iter=max_iter, tol=tol)

    if RUST_AVAILABLE:
        rust_scores_list = rust_pagerank(adj.tolist(), damping, max_iter, tol)
        rust_scores = np.asarray(rust_scores_list, dtype=np.float64)
    else:
        print(f"[{name}] SKIP: Rust 内核未安装（pip install -e crates/pagerank_rust）")
        return True

    # 排序一致化（PageRank 节点顺序按输入矩阵定义，两者一致）
    diff_l1 = float(np.abs(py_scores - rust_scores).sum())
    diff_max = float(np.abs(py_scores - rust_scores).max())
    # 期望：浮点累加顺序差异，误差量级在 1e-9 ~ 1e-12
    threshold_l1 = 1e-6
    threshold_max = 1e-6
    passed = diff_l1 < threshold_l1 and diff_max < threshold_max

    status = "PASS" if passed else "FAIL"
    print(
        f"[{name}] {status}  "
        f"L1_diff={diff_l1:.3e} (tol={threshold_l1:.0e})  "
        f"max_diff={diff_max:.3e} (tol={threshold_max:.0e})  "
        f"sum(py)={py_scores.sum():.6f} sum(rust)={rust_scores.sum():.6f}"
    )
    return passed


def main() -> int:
    print(f"Rust 内核可用: {RUST_AVAILABLE}")
    print(f"Python: {sys.version.split()[0]}  numpy: {np.__version__}")
    print()

    cases: list[tuple[str, np.ndarray, dict]] = []

    # Case 1: 经典 3 节点
    cases.append((
        "classic_3node",
        np.array([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]),
        {"damping": 0.85, "max_iter": 100, "tol": 1e-9},
    ))

    # Case 2: 完全图 K4
    cases.append((
        "k4",
        np.ones((4, 4), dtype=np.float64) - np.eye(4),
        {"damping": 0.85, "max_iter": 100, "tol": 1e-9},
    ))

    # Case 3: 含 dangling 节点（出度=0）
    cases.append((
        "with_dangling",
        np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],  # node 2 是 dangling
        ]),
        {"damping": 0.85, "max_iter": 100, "tol": 1e-9},
    ))

    # Case 4: 中等规模稀疏随机图
    cases.append((
        "random_sparse_50",
        build_random_graph(50, density=0.1, seed=42),
        {"damping": 0.85, "max_iter": 100, "tol": 1e-8},
    ))

    # Case 5: 较密集随机图
    cases.append((
        "random_dense_30",
        build_random_graph(30, density=0.3, seed=123),
        {"damping": 0.9, "max_iter": 200, "tol": 1e-9},
    ))

    # Case 6: 极端 damping
    cases.append((
        "low_damping",
        np.array([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]),
        {"damping": 0.5, "max_iter": 100, "tol": 1e-9},
    ))

    all_passed = True
    for name, adj, kwargs in cases:
        ok = compare_one(name, adj, **kwargs)
        all_passed = all_passed and ok

    print()
    if all_passed:
        print("ALL PASSED")
        return 0
    print("SOME FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())