"""PageRank Python vs Rust 数值对照测试。

不依赖 pytest，直接 `python tests/test_pagerank_rust_parity.py` 运行。
输出 PASS/FAIL 与详细数值。
"""
from __future__ import annotations

import importlib.util
import random
import sys
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

try:
    from a_memorix_kernel_rust import pagerank_csr as rust_pagerank_csr

    RUST_CSR_AVAILABLE = True
except ImportError:
    RUST_CSR_AVAILABLE = False
    rust_pagerank_csr = None


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
    return adj


def dense_j_to_i_to_csr_src_tgt(adj_j_to_i: np.ndarray):
    """将 dense j->i 矩阵转为 GraphStore CSR (src->tgt = adj.T)。"""
    from scipy.sparse import csr_matrix

    src_tgt = adj_j_to_i.T  # src->tgt
    return csr_matrix(src_tgt)


def compare_one(name: str, adj: np.ndarray, damping: float, max_iter: int, tol: float) -> bool:
    """对一组参数跑 Python dense / Rust dense / Rust CSR，比较输出。"""
    py_scores = pagerank_python(adj, damping=damping, max_iter=max_iter, tol=tol)

    rust_scores = None
    if RUST_AVAILABLE:
        rust_scores_list = rust_pagerank(adj.tolist(), damping, max_iter, tol)
        rust_scores = np.asarray(rust_scores_list, dtype=np.float64)
    else:
        print(f"[{name}] SKIP dense: Rust 内核未安装（pip install wheel）")

    csr_scores = None
    if RUST_CSR_AVAILABLE:
        csr = dense_j_to_i_to_csr_src_tgt(adj)
        csr_list = rust_pagerank_csr(
            adj.shape[0],
            csr.indptr.astype(np.int64).tolist(),
            csr.indices.astype(np.int32).tolist(),
            csr.data.astype(np.float64).tolist(),
            damping,
            max_iter,
            tol,
        )
        csr_scores = np.asarray(csr_list, dtype=np.float64)
    else:
        print(f"[{name}] SKIP csr: Rust CSR 未安装")

    threshold_l1 = 1e-6
    threshold_max = 1e-6
    passed = True

    if rust_scores is not None:
        diff_l1 = float(np.abs(py_scores - rust_scores).sum())
        diff_max = float(np.abs(py_scores - rust_scores).max())
        ok = diff_l1 < threshold_l1 and diff_max < threshold_max
        passed = passed and ok
        status = "PASS" if ok else "FAIL"
        print(
            f"[{name}/dense] {status}  "
            f"L1_diff={diff_l1:.3e}  max_diff={diff_max:.3e}  "
            f"sum(py)={py_scores.sum():.6f} sum(rust)={rust_scores.sum():.6f}"
        )

    if csr_scores is not None:
        diff_l1 = float(np.abs(py_scores - csr_scores).sum())
        diff_max = float(np.abs(py_scores - csr_scores).max())
        ok = diff_l1 < threshold_l1 and diff_max < threshold_max
        passed = passed and ok
        status = "PASS" if ok else "FAIL"
        print(
            f"[{name}/csr]   {status}  "
            f"L1_diff={diff_l1:.3e}  max_diff={diff_max:.3e}  "
            f"sum(py)={py_scores.sum():.6f} sum(csr)={csr_scores.sum():.6f}"
        )

    if rust_scores is not None and csr_scores is not None:
        diff_l1 = float(np.abs(rust_scores - csr_scores).sum())
        diff_max = float(np.abs(rust_scores - csr_scores).max())
        ok = diff_l1 < threshold_l1 and diff_max < threshold_max
        passed = passed and ok
        status = "PASS" if ok else "FAIL"
        print(
            f"[{name}/dense-vs-csr] {status}  "
            f"L1_diff={diff_l1:.3e}  max_diff={diff_max:.3e}"
        )

    if rust_scores is None and csr_scores is None:
        return True  # 全 skip 视为不失败

    return passed


def main() -> int:
    print(f"Rust dense 可用: {RUST_AVAILABLE}")
    print(f"Rust CSR   可用: {RUST_CSR_AVAILABLE}")
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

    # Case 3: 含 dangling 节点（出度=0）—— dense 语义下列和=0
    cases.append((
        "with_dangling",
        np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],  # node 2 是 dangling (列 2 全 0? 这里行2全0是入边)
            # 在 j->i 语义中，列 j 全 0 => j 无出边 => dangling
            # 上面矩阵列 0 全 0 => node 0 dangling
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

    # Case 7: 明确多 dangling（列 2、3 全 0）
    cases.append((
        "multi_dangling",
        np.array([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]),
        {"damping": 0.85, "max_iter": 100, "tol": 1e-9},
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
