"""tensor_expm.py — GPU matrix exponential using tensor core engine v5.

Polynomial evaluation uses FP64 (cp.matmul / cuBLAS DGEMM).
Squaring uses one of three modes:
  default:                       FP64 cp.matmul  (~1e-15, baseline = CuPy speed)
  use_tensor_cores=True:         TF32 ops.matmul (~1e-5, ~60× CuPy; NaN if entries > 1e4)
  improved='improved32':         Ozaki 5×TF32 GEMMs → FP64 accum (~1e-7, ~6× FP64)

Usage:
    from tensor_expm import expm, batched_expm, init_engine
    ops = init_engine()
    E   = expm(A, ops)
    E   = expm(A, ops, use_tensor_cores=True)
    E   = expm(A, ops, improved='improved32')
    Es  = batched_expm(As, ops, use_tensor_cores=True)
"""
import os, sys
import numpy as np
import cupy as cp

_B13 = np.array([
    64764752532480000., 32382376266240000., 7771770303897600.,
     1187353796428800.,   129060195264000.,   10559470521600.,
       670442572800.,       33522128640.,        1323241920.,
            40840800.,            960960.,            16380.,
                182.,                 1.
], dtype=np.float64)
_THETA13 = 5.371920351148152
_TC_SAFE  = 1e4

def init_engine(lib_path=None):
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    from tensor_matrix_ops import TensorMatrixOps
    return TensorMatrixOps(lib_path)

def _pade13(A):
    b = _B13
    I = cp.eye(A.shape[-1], dtype=cp.float64)
    if A.ndim == 3:
        I = cp.tile(I[None], (A.shape[0], 1, 1))
    A2 = cp.matmul(A, A); A4 = cp.matmul(A2, A2); A6 = cp.matmul(A2, A4)
    W1 = b[13]*A6 + b[11]*A4 + b[9]*A2
    W2 = b[12]*A6 + b[10]*A4 + b[8]*A2
    Z1 = cp.matmul(A6, W1); Z2 = cp.matmul(A6, W2)
    U = cp.matmul(A, Z1 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*I)
    V = Z2 + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*I
    return U, V

def _square(R, ops, use_tensor_cores, improved):
    if improved == 'improved32':
        return ops.improved_matmul32(R, R)
    if use_tensor_cores and float(cp.max(cp.abs(R))) < _TC_SAFE:
        return ops.matmul(R, R)
    return cp.matmul(R, R)

def expm(A, ops, use_tensor_cores=False, improved=None):
    is_np = isinstance(A, np.ndarray)
    Ag = cp.asarray(A, dtype=cp.float64)
    norm1 = float(cp.linalg.norm(Ag, ord=1))
    s = max(0, int(np.ceil(np.log2(norm1 / _THETA13)))) if norm1 > 0 else 0
    U, V = _pade13(Ag * (2.0 ** -s))
    R = cp.linalg.solve(V - U, V + U)
    for _ in range(s):
        R = _square(R, ops, use_tensor_cores, improved)
    cp.cuda.Stream.null.synchronize()
    return R.get() if is_np else R

def batched_expm(batch_A, ops, use_tensor_cores=False, improved=None):
    is_np = isinstance(batch_A, np.ndarray)
    A = cp.asarray(batch_A, dtype=cp.float64)
    batch, n, _ = A.shape
    nm = float(cp.linalg.norm(A.reshape(batch, -1), ord=np.inf, axis=1).max())
    s = max(0, int(np.ceil(np.log2(nm / _THETA13)))) if nm > 0 else 0
    U, V = _pade13(A * (2.0 ** -s))
    R = cp.linalg.solve(V - U, V + U)
    for _ in range(s):
        if improved == 'improved32':
            R = cp.stack([ops.improved_matmul32(R[k], R[k]) for k in range(batch)])
        elif use_tensor_cores and float(cp.max(cp.abs(R))) < _TC_SAFE:
            R = ops.batched_matmul(R, R)
        else:
            R = cp.matmul(R, R)
    cp.cuda.Stream.null.synchronize()
    return R.get() if is_np else R
