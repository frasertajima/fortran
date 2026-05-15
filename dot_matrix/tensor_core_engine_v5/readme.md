### New: matmul_tc_split — Ozaki Split Precision Matrix Multiplication

A hybrid GEMM that delivers FP32-level accuracy (~1e-6 to 1e-7 relative error) at 5–19× the speed of stock FP64, by combining one exact-FP32 GEMM on CUDA cores with two TF32 correction GEMMs on tensor cores. This means that we hit the *same accuracy* as improved_fp32 but with 2x the speed:

<img width="2234" height="740" alt="fig_matmul_4tier" src="https://github.com/user-attachments/assets/b593984a-cd70-4b93-8753-f9cfdbc741f1" />


In the context of other batched operations:
<img width="2084" height="741" alt="fig_batched_matmul_4tier" src="https://github.com/user-attachments/assets/a839e579-755c-41c8-8c33-0a20c44ca74f" />

Replaced improved matrix operations with near FP32 accuracy and true FP64 workflows. FP64 does not use split precision due to lack of FP64 tensor cores on consumer GPUs. Still, we get a modest uplift by avoiding python overhead. Added Jupyter notebook to test in quantum simulation and Padé approximation tests.

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
