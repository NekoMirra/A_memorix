# a_memorix_kernel_rust

> **状态：PoC 骨架** —— 仅做技术验证（PyO3 通路 + 数值正确性 + 工具链可行性）。**不直接绑定到 `graph_store.compute_pagerank`**。

## 目的

为 A_memorix 长期记忆子系统探索 Rust 加速路径。PageRank 是 `core/storage/graph_store.py:873 compute_pagerank` 的核心循环，目前用 scipy 稀疏矩阵乘法 + Python 循环。Rust 实现的潜在收益是去掉 Python 循环开销（5-10x 加速在中等规模图上常见）。

本 crate 仅承担：

1. **PyO3 工具链可行性验证** —— Windows / Linux / macOS 三平台编译
2. **数值正确性 baseline** —— 与 Python `graph_store.compute_pagerank` 输出对照
3. **集成路径样板** —— 给上游 A_memorix 仓库的 `compute_pagerank` 调用点提供参考

## 工具链

- Rust 1.97.0 stable
- maturin 1.14+
- pyo3 0.29 (with `extension-module` feature)

工具链通过 `rust-toolchain.toml` 锁定到 `stable-x86_64-pc-windows-gnu`（Windows）。Linux/macOS 上切换：

```bash
rustup target add x86_64-unknown-linux-gnu   # Linux
rustup target add x86_64-apple-darwin         # macOS
```

## 构建

```bash
# 开发构建
cd crates/pagerank_rust
maturin develop --release

# 仅检查编译
cargo check --release
```

## 暴露给 Python 的 API

```python
import a_memorix_kernel_rust as k

# PageRank：纯 Rust 实现，幂迭代法
# 与 graph_store.compute_pagerank 的密集路径语义等价（personalization=None, alpha=damping）
scores = k.pagerank(
    adjacency=[[0.0, 1.0, 1.0],
               [1.0, 0.0, 0.0],
               [0.0, 1.0, 0.0]],
    damping=0.85,
    max_iter=100,
    tol=1e-6,
)
# scores == [0.3878..., 0.3418..., 0.2703...] (近似)

# PoC 探针：仅用于验证 PyO3 通路
k.sum_as_string(2, 3)  # "5"
```

## 与 `graph_store.compute_pagerank` 的对比计划

1. 在 `core/storage/graph_store.py` 加软依赖包装:
   ```python
   try:
       from a_memorix_kernel_rust import pagerank as _rust_pagerank
       _HAS_RUST_KERNEL = True
   except ImportError:
       _HAS_RUST_KERNEL = False

   def compute_pagerank(self, ...):
       ...
       if _HAS_RUST_KERNEL and personalization is None:
           # 当前 PoC 仅支持均匀 personalization 路径
           scores = _rust_pagerank(adj, alpha, max_iter, tol)
           return {self._nodes[i]: float(v) for i, v in enumerate(scores)}
       # fallback to numpy/scipy impl
       ...
   ```
2. 在 `requirements.txt` 加 `[kernel-rust]`
3. A_memorix 增量引入：先小规模图（< 1000 节点）做对照测试，再放开

## 当前 PoC 限制

- **仅 dense matrix**：未优化稀疏 CSR/CSC（这是 A_memorix 的真实场景）
- **不支持 personalization**：PoC 仅覆盖均匀 teleportation，向量化 personalization 是下一阶段
- **未启用 numpy-binding**：pyo3-ffi 与 numpy 版本需严格对齐，集成时单独处理
- **未做 SIMD/并行化**：单线程基准
- **未做 memory pool**：每次调用重新分配 Vec

## 后续路线（按依赖顺序）

| 阶段 | 目标 | 位置 |
|---|---|---|
| 阶段 0（当前） | PyO3 工具链 + 数值正确性 PoC | 本 crate |
| 阶段 1 | personalization 向量化 + numpy binding | 本 crate |
| 阶段 2 | 接入 `graph_store.compute_pagerank` 软依赖包装 | A_memorix upstream |
| 阶段 3 | 稀疏 CSR/CSC + SIMD | 本 crate |
| 阶段 4 | 批量 SQLite upsert / 图邻接批量运算 | 本 crate + upstream |

## 跨平台注意事项

| 平台 | 工具链 | 备注 |
|---|---|---|
| Windows | `stable-x86_64-pc-windows-gnu` | 需 MinGW-w64 GCC；MSVC 链路更复杂 |
| Linux | `stable-x86_64-unknown-linux-gnu` | glibc 兼容性需考虑 musl alternative |
| macOS | `stable-x86_64-apple-darwin` | universal binary 需额外步骤 |

CI 配置（不在本 PoC 范围）：在 A_memorix 仓库加 matrix build workflow。