"""GraphStore.compute_pagerank 与 Rust CSR / dense / scipy 对照。

直接运行:
    python crates/pagerank_rust/tests/test_graph_store_pagerank_rust.py
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Dict, Optional

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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# 预注册包结构，满足 graph_store 的相对导入
for pkg in ("core", "core.utils", "core.storage", "core.retrieval"):
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [str(REPO_ROOT / pkg.replace(".", "/"))]  # type: ignore[attr-defined]
        sys.modules[pkg] = m

_load_module("core.utils.hash", REPO_ROOT / "core" / "utils" / "hash.py")
_load_module("core.utils.io", REPO_ROOT / "core" / "utils" / "io.py")
_load_module(
    "core.retrieval.pagerank_kernel",
    REPO_ROOT / "core" / "retrieval" / "pagerank_kernel.py",
)
_gs = _load_module("core.storage.graph_store", REPO_ROOT / "core" / "storage" / "graph_store.py")
GraphStore = _gs.GraphStore


def _scipy_only_pagerank(
    store: GraphStore,
    personalization: Optional[Dict[str, float]] = None,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-9,
):
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

    if personalization is None:
        p = np.ones(n) / n
    else:
        p = np.zeros(n)
        total_weight = sum(personalization.values())
        for node, weight in personalization.items():
            canon = store._canonicalize(node)
            if canon in store._node_to_idx:
                idx = store._node_to_idx[canon]
                p[idx] = weight / total_weight
        if p.sum() == 0:
            p = np.ones(n) / n
        else:
            p = p / p.sum()

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


def _dict_diff(a: dict, b: dict):
    nodes = sorted(set(a) | set(b))
    diffs = [abs(a.get(n, 0.0) - b.get(n, 0.0)) for n in nodes]
    return (max(diffs) if diffs else 0.0), (sum(diffs) if diffs else 0.0)


def _run_case(name: str, edges, weights, personalization=None) -> bool:
    store = GraphStore()
    store.add_edges(edges, weights=weights)

    path_scores = store.compute_pagerank(
        personalization=personalization, alpha=0.85, max_iter=100, tol=1e-9
    )
    scipy_scores = _scipy_only_pagerank(
        store, personalization=personalization, alpha=0.85, max_iter=100, tol=1e-9
    )

    max_diff, l1 = _dict_diff(path_scores, scipy_scores)
    print(f"--- {name} ---")
    print("path :", {k: round(v, 6) for k, v in sorted(path_scores.items())})
    print("scipy:", {k: round(v, 6) for k, v in sorted(scipy_scores.items())})
    print(f"max_diff={max_diff:.3e} L1={l1:.3e}")

    if max_diff < 1e-5 and l1 < 1e-5:
        print(f"[{name}] PASS")
        return True
    print(f"[{name}] FAIL")
    return False


def main() -> int:
    try:
        from a_memorix_kernel_rust import pagerank as _  # noqa: F401

        rust_dense = True
    except ImportError:
        rust_dense = False

    try:
        from a_memorix_kernel_rust import pagerank_csr as _  # noqa: F401

        rust_csr = True
    except ImportError:
        rust_csr = False

    if not rust_dense and not rust_csr:
        print("SKIP: Rust 内核未安装")
        return 0

    print(f"Rust dense available: {rust_dense}")
    print(f"Rust CSR   available: {rust_csr}")

    ok = True
    ok = _run_case(
        "triangle_plus",
        [("A", "B"), ("A", "C"), ("B", "C"), ("C", "A")],
        [1.0, 1.0, 1.0, 1.0],
    ) and ok

    ok = _run_case(
        "with_dangling",
        [("A", "B"), ("B", "C"), ("C", "A"), ("A", "D")],
        [1.0, 1.0, 1.0, 1.0],
    ) and ok

    ok = _run_case(
        "weighted",
        [("X", "Y"), ("Y", "Z"), ("Z", "X"), ("X", "Z")],
        [2.0, 1.0, 0.5, 3.0],
    ) and ok

    # 个性化: 单 seed
    ok = _run_case(
        "pers_seed_A",
        [("A", "B"), ("A", "C"), ("B", "C"), ("C", "A")],
        [1.0, 1.0, 1.0, 1.0],
        personalization={"A": 1.0},
    ) and ok

    # 个性化: 双 seed + dangling
    ok = _run_case(
        "pers_two_seeds_dangling",
        [("A", "B"), ("B", "C"), ("C", "A"), ("A", "D")],
        [1.0, 1.0, 1.0, 1.0],
        personalization={"A": 2.0, "C": 1.0},
    ) and ok

    # 个性化: 非归一化权重
    ok = _run_case(
        "pers_unnormalized",
        [("X", "Y"), ("Y", "Z"), ("Z", "X"), ("X", "Z")],
        [2.0, 1.0, 0.5, 3.0],
        personalization={"X": 3.0, "Y": 1.0, "Z": 5.0},
    ) and ok

    print()
    if ok:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
