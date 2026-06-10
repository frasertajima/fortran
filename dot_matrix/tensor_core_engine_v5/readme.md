June 10, 2026: after code review by Claude Fable 5, code cleanup for v5.1; smoke test passed and new notebook benchmark for v5.1 is saved.

- R1/B2  Every py_* entry point is now an integer(c_int) FUNCTION returning 0 on success / first failing CUDA-or-cuBLAS status (nonzero) on error.  Existing ctypes callers with restype=None are unaffected.
- B4/R3  No private cuBLAS handles: the module borrows resources(1) from cuda_batch_state (handle + stream).  The unused second handle and stream are gone.
- B5/M1  improved_matmul64 deleted (Python wrapper uses cp.matmul); dead kernels, tensor_5d type, FP16 path, CUDA_R_TF32 constant removed.
- M2     Split routines (matmul_tc_split, improved_matmul32, batched_matmul_tc_split) now run from the workspace pool, so py_workspace_cleanup really frees ALL persistent device memory.
- P1     matrix_dot / tensor_matrix_multiply ping-pong between pool buffers instead of copying the full matrix every iteration; redundant zero-fills removed (beta=0 GEMMs overwrite the output).
- P3     tensor_4d_matmul / tensor_5d_matmul use chunked cublasSgemmStridedBatched instead of one GEMM per slice.
- B6     64-bit offsets in strided_batch_multiply.
- BUG    matrix_dot non-power-of-two path computed A^(p+1) instead of A^p (loop ran iterations-1 multiplies on top of the initial A²).

### New: matmul_tc_split — Ozaki Split Precision Matrix Multiplication

A hybrid GEMM that delivers FP32-level accuracy (~1e-6 to 1e-7 relative error) at 5–19× the speed of stock FP64, by combining one exact-FP32 GEMM on CUDA cores with two TF32 correction GEMMs on tensor cores. This means that we hit the *same accuracy* as improved_fp32 but with 2x the speed:

<img width="2234" height="740" alt="fig_matmul_4tier" src="https://github.com/user-attachments/assets/b593984a-cd70-4b93-8753-f9cfdbc741f1" />


In the context of other batched operations:
<img width="2084" height="741" alt="fig_batched_matmul_4tier" src="https://github.com/user-attachments/assets/a839e579-755c-41c8-8c33-0a20c44ca74f" />

Replaced improved matrix operations with near FP32 accuracy and true FP64 workflows. FP64 does not use split precision due to lack of FP64 tensor cores on consumer GPUs. Still, we get a modest uplift by avoiding python overhead. Added Jupyter notebook to test in quantum simulation and Padé approximation tests.

## When to use each tier

| Situation | Recommended |
|---|---|
| Training loop, loss is the only output | `matmul` (TF32) |
| Gradient accumulation, weight updates | `matmul_tc_split` |
| Numerical solver inner loop, error tolerance 1e-5 | `matmul_tc_split` |
| Numerical solver, error tolerance 1e-10 | FP64 |
| Ill-conditioned system, need FP64 solution | MPDOK (GMRES-IR or LU-IR) |
| Single-kernel bias+relu fusion needed | `batched_matmul_bias_relu` (TF32) |

---

After reviewing https://github.com/NVIDIA/cudnn-frontend, I am relieved to find our tensor core engine has incorporated most of the learnings employed by Nvidia (and confirms our approach in the matter). One interesting discovery did come up: **epilogue fusion via cuBLAS-lt**. 
Thus v5 incorporates these changes with modest improvement:

<img width="1489" height="515" alt="fused" src="https://github.com/user-attachments/assets/da81601b-d9e7-4172-b26b-e48c447a3a0c" />

It involves the use of two independent components:

### Component A: Extend `cuda_matlib.cuf` with fused variants (new functions)

Add fused bias+activation entry points to the existing `cuda_matlib.cuf` module. These are
**new functions** — all v4 functions remain unchanged and the Python API is backward-compatible.

New functions:
```
py_batched_matmul_bias_relu   — batched GEMM + bias + ReLU, single kernel
py_batched_matmul_bias        — batched GEMM + bias only, single kernel
```

These cover the most common forward-pass pattern in fully-connected layers:
`output = relu(input @ weight.T + bias)`.

### Component B: Restore and upgrade `tensor_core_gemm8.cuf` with cuBLAS-lt

v5 brings back `tensor_core_gemm8.cuf` (dropped in v3/v4) but applies:
- v4's workspace pool for temp allocations
- cuBLAS-lt epilogue fusion replacing the 2-kernel patterns in `tc_gemm_bias_act` and
  `tc_linear_forward`
- Persistent algo caching for fixed shapes (fc1/fc2/fc3 of CIFAR-10 CNN)
