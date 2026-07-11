//! PoC: Rust 加速的 PageRank 实现
//!
//! 目标：替换 `core/storage/graph_store.py` 中的 Python 循环。
//!
//! ## Dense 邻接语义 (`pagerank`)
//!
//!     adjacency[i][j] 表示 j -> i 的边权（i 接收 j 的链接）
//!     - 节点 i 的入度 = adjacency[i].sum()
//!     - 节点 j 的出度 = adjacency[:, j].sum()（列和）
//!
//! ## CSR 邻接语义 (`pagerank_csr`) —— 与 GraphStore 一致
//!
//!     CSR 行 = source，列 = target，即 adj[src, tgt] = src -> tgt
//!     - 节点 src 的出度 = 行和
//!     - 转移: M = A^T * D_inv（D_inv 作用于出度）
//!
//! PageRank 公式（支持 personalization 向量 p_orig，和为 1）:
//!     scores[i] = (1 - damping) * p_orig[i]
//!               + damping * Σ_j (edge_j_to_i / out_sum[j]) * scores[j]
//!               + damping * dangling_sum * p_orig[i]
//!
//! 其中 dangling_sum = Σ_{k: out_sum[k]=0} scores[k]
//!
//! 等价于 GraphStore scipy 路径:
//!     p_new = alpha * M @ p + (1 - alpha) * p_orig
//!     p_new += (1 - p_new.sum()) * p_orig   # 悬挂节点流失质量按 p_orig 回注
//!
//! 当 personalization 为 None 时 p_orig = 1/n（均匀）。

use pyo3::prelude::*;

/// 解析并归一化 personalization 向量；None 或全 0 时回退均匀分布。
fn resolve_personalization(n: usize, personalization: Option<Vec<f64>>) -> PyResult<Vec<f64>> {
    match personalization {
        None => Ok(vec![1.0_f64 / n as f64; n]),
        Some(p) => {
            if p.len() != n {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "personalization 长度 {} != 节点数 {}",
                    p.len(),
                    n
                )));
            }
            let total: f64 = p.iter().sum();
            if total <= 0.0 {
                Ok(vec![1.0_f64 / n as f64; n])
            } else {
                Ok(p.iter().map(|x| x / total).collect())
            }
        }
    }
}

/// Power iteration 实现的 PageRank（dense，列出度语义）。
///
/// 参数:
/// - `adjacency`: 邻接矩阵 `[[w_ij, ...], ...]`，w_ij 表示 j -> i 的权重
/// - `damping`: 阻尼系数（典型 0.85）
/// - `max_iter`: 最大迭代次数
/// - `tol`: 收敛阈值（L1 差值）
/// - `personalization`: 可选长度 n 的 teleport 向量（会归一化）；None=均匀
///
/// 返回:
/// - 各节点得分列表，长度 = `len(adjacency)`
#[pyfunction]
#[pyo3(signature = (adjacency, damping = 0.85, max_iter = 100, tol = 1e-6, personalization = None))]
fn pagerank(
    adjacency: Vec<Vec<f64>>,
    damping: f64,
    max_iter: usize,
    tol: f64,
    personalization: Option<Vec<f64>>,
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

    let p_orig = resolve_personalization(n, personalization)?;

    // 2) 初始分数 = personalization
    let mut scores = p_orig.clone();

    // 3) 幂迭代
    // next[i] = damping * acc[i] + (1 - damping + damping * dangling_sum) * p_orig[i]
    let mut next = vec![0.0_f64; n];
    for _ in 0..max_iter {
        let dangling_sum: f64 = scores
            .iter()
            .zip(out_sum.iter())
            .filter_map(|(s, out)| if *out == 0.0 { Some(*s) } else { None })
            .sum();
        let teleport_scale = (1.0 - damping) + damping * dangling_sum;

        for i in 0..n {
            let mut acc = 0.0_f64;
            for j in 0..n {
                if out_sum[j] > 0.0 {
                    acc += adjacency[i][j] / out_sum[j] * scores[j];
                }
            }
            next[i] = damping * acc + teleport_scale * p_orig[i];
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

/// Power iteration PageRank（CSR，GraphStore src->tgt 语义）。
///
/// 参数:
/// - `n`: 节点数
/// - `indptr`: CSR indptr，长度 n+1（i64，兼容 scipy）
/// - `indices`: CSR 列索引（目标节点，i32 兼容 scipy）
/// - `data`: CSR 非零边权
/// - `damping` / `max_iter` / `tol`: 同 dense 路径
/// - `personalization`: 可选长度 n 的 teleport 向量（会归一化）；None=均匀
///
/// 邻接语义:
/// - 行 = source，列 = target（adj[src, tgt] = src -> tgt）
/// - 出度 = 行和
/// - 转移贡献: next[tgt] += (w / out[src]) * scores[src]
///
/// 返回长度 n 的得分向量。
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, data, damping = 0.85, max_iter = 100, tol = 1e-6, personalization = None))]
fn pagerank_csr(
    n: usize,
    indptr: Vec<i64>,
    indices: Vec<i32>,
    data: Vec<f64>,
    damping: f64,
    max_iter: usize,
    tol: f64,
    personalization: Option<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if indptr.len() != n + 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "indptr 长度 {} != n+1 ({})",
            indptr.len(),
            n + 1
        )));
    }
    if indices.len() != data.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "indices 长度 {} != data 长度 {}",
            indices.len(),
            data.len()
        )));
    }
    let nnz = data.len();
    let last = indptr[n];
    if last < 0 || last as usize != nnz {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "indptr[n]={} 与 nnz={} 不一致",
            last, nnz
        )));
    }
    for (i, &p) in indptr.iter().enumerate() {
        if p < 0 || p as usize > nnz {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "indptr[{}]={} 越界 (nnz={})",
                i, p, nnz
            )));
        }
        if i > 0 && indptr[i] < indptr[i - 1] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "indptr 非单调: indptr[{}]={} < indptr[{}]={}",
                i,
                indptr[i],
                i - 1,
                indptr[i - 1]
            )));
        }
    }
    for (k, &col) in indices.iter().enumerate() {
        if col < 0 || col as usize >= n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "indices[{}]={} 越界 (n={})",
                k, col, n
            )));
        }
    }

    // 出度 = 行和（src 行）
    let mut out_sum = vec![0.0_f64; n];
    for src in 0..n {
        let start = indptr[src] as usize;
        let end = indptr[src + 1] as usize;
        let mut s = 0.0_f64;
        for k in start..end {
            s += data[k];
        }
        out_sum[src] = s;
    }

    let p_orig = resolve_personalization(n, personalization)?;
    let mut scores = p_orig.clone();
    let mut next = vec![0.0_f64; n];

    for _ in 0..max_iter {
        // dangling mass: 出度为 0 的节点上的分数和
        let dangling_sum: f64 = scores
            .iter()
            .zip(out_sum.iter())
            .filter_map(|(s, out)| if *out == 0.0 { Some(*s) } else { None })
            .sum();
        let teleport_scale = (1.0 - damping) + damping * dangling_sum;

        // next[tgt] 累加来自各 src 的转移
        next.fill(0.0);
        for src in 0..n {
            let out = out_sum[src];
            if out <= 0.0 {
                continue;
            }
            let scale = scores[src] / out;
            let start = indptr[src] as usize;
            let end = indptr[src + 1] as usize;
            for k in start..end {
                let tgt = indices[k] as usize;
                next[tgt] += data[k] * scale;
            }
        }

        for i in 0..n {
            next[i] = damping * next[i] + teleport_scale * p_orig[i];
        }

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
    use super::pagerank_csr;

    #[pymodule_export]
    use super::sum_as_string;
}
