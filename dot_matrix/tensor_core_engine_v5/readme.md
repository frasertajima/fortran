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
