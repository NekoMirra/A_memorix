//! PoC: Rust 加速的 PageRank 实现
//!
//! 目标：替换 `core/storage/graph_store.py:873 compute_pagerank` 中的 Python 循环。
//!
//! 邻接矩阵语义:
//!     adjacency[i][j] 表示 j -> i 的边权（i 接收 j 的链接）
//!     - 节点 i 的入度 = adjacency[i].sum()
//!     - 节点 j 的出度 = adjacency[:, j].sum()（列和）
//!
//! PageRank 公式:
//!     scores[i] = (1 - damping) / n
//!               + damping * Σ_j (adjacency[i][j] / out_sum[j]) * scores[j]
//!               + damping * dangling_sum / n
//!
//! 其中 dangling_sum = Σ_{k: out_sum[k]=0} scores[k]
//!
//! **重要：本 PoC 仅做技术验证（PyO3 通路 + 数值正确性），不直接绑定到 graph_store。**
//!
//! ## 接入路径
//!
//! 1. 在 `core/storage/graph_store.py` 加 try/except 软依赖:
//!    ```python
//!    try:
//!        from a_memorix_kernel_rust import pagerank as _rust_pagerank
//!        _HAS_RUST_KERNEL = True
//!    except ImportError:
//!        _HAS_RUST_KERNEL = False
//!    ```
//! 2. `compute_pagerank` 在 `_HAS_RUST_KERNEL` 且 `personalization is None` 时调用 Rust
//! 3. 通过 `requirements.txt` 的 `[kernel-rust] extra` 让用户可选安装
//!
//! ## 后续可扩展
//!
//! - `graph_store` 稀疏邻接批量运算（CSR/CSC 转换）
//! - personalization 向量化（当前 PoC 仅支持均匀）
//! - `metadata_store` 批量 SQLite upsert 预编译

use pyo3::prelude::*;

/// Power iteration 实现的 PageRank。
///
/// 参数:
/// - `adjacency`: 邻接矩阵 `[[w_ij, ...], ...]`，w_ij 表示 j -> i 的权重
/// - `damping`: 阻尼系数（典型 0.85）
/// - `max_iter`: 最大迭代次数
/// - `tol`: 收敛阈值（L1 差值）
///
/// 返回:
/// - 各节点得分列表，长度 = `len(adjacency)`
#[pyfunction]
#[pyo3(signature = (adjacency, damping = 0.85, max_iter = 100, tol = 1e-6))]
fn pagerank(
    adjacency: Vec<Vec<f64>>,
    damping: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Vec<f64>> {
    let n = adjacency.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // 1) 计算每个节点的出度 = 列和（j -> all i 的边权和）
    let mut out_sum = vec![0.0_f64; n];
    for (i, row) in adjacency.iter().enumerate() {
        if row.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "邻接矩阵第 {} 行长度 {} != 节点数 {}",
                i,
                row.len(),
                n
            )));
        }
        for (j, &w) in row.iter().enumerate() {
            // row[i] = adjacency[i]，累加到 out_sum[j] 表示 j 的出度
            out_sum[j] += w;
        }
    }

    // 2) 处理 dangling node（出度为 0）：均匀分配给所有节点
    let mut scores = vec![1.0_f64 / n as f64; n];

    // 3) 幂迭代
    let teleport = (1.0 - damping) / n as f64;
    let mut next = vec![0.0_f64; n];
    for _ in 0..max_iter {
        let dangling_sum: f64 = scores
            .iter()
            .zip(out_sum.iter())
            .filter_map(|(s, out)| if *out == 0.0 { Some(s) } else { None })
            .sum();
        let dangling_term = damping * dangling_sum / n as f64;

        for i in 0..n {
            let mut acc = 0.0_f64;
            for j in 0..n {
                if out_sum[j] > 0.0 {
                    acc += adjacency[i][j] / out_sum[j] * scores[j];
                }
            }
            next[i] = teleport + damping * acc + dangling_term;
        }

        // 收敛判断：L1 差值
        let diff: f64 = scores
            .iter()
            .zip(next.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        std::mem::swap(&mut scores, &mut next);
        if diff < tol {
            break;
        }
    }

    Ok(scores)
}

/// 简单求和 PoC，验证 PyO3 链路。
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// 模块入口。
#[pymodule]
mod a_memorix_kernel_rust {
    #[pymodule_export]
    use super::pagerank;

    #[pymodule_export]
    use super::sum_as_string;
}