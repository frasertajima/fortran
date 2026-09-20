"""rust_matrix_ops -- CuPy-facing wrapper for librust_matlib.so, the Rust
(cuda-oxide) counterpart of tensor_matrix_ops.py / cuda_matlib.so.

Same method names and contracts as TensorMatrixOps, so a benchmark cell can
swap `TensorMatrixOps()` for `RustMatrixOps()` and run unchanged: NumPy or
CuPy in, same kind out; f64 interfaces; row-major (CuPy's default C order),
so the Fortran wrapper's asfortranarray copies are simply not needed.

    from rust_matrix_ops import RustMatrixOps
    ops = RustMatrixOps()
    ops.matrix_power(a, 4)                # TF32, repeated squaring
    ops.matmul(a, b)                      # TF32
    ops.matmul_tc_split(a, b)             # 1 exact + 2 TF32 GEMMs, ~1e-6
    ops.improved_matmul32(a, b)           # 5 exact GEMMs, ~1e-7
    ops.batched_matmul(a, b)              # TF32 strided batched
    ops.batched_matmul_tc_split(a, b)
    ops.batched_matmul_fp64(a, b)         # cublasDgemmStridedBatched
    ops.batched_matmul_bias_relu(a, b, bias)   # cuBLASLt fused epilogue, exact FP32
                                               # (TME_EPILOGUE_TF32=1 -> TF32)
    ops.batched_matmul_bias(a, b, bias)
    ops.vector_matmul(v, a); ops.matmul_vector(a, v)   # FP64 Dgemv
    ops.batched_vector_matmul(v, a)       # TF32
    ops.strided_batch_matmul(m, k, n, batch, a, b)
    ops.tensor_4d_matmul(a, b); ops.tensor_5d_matmul(a, b)
    ops.tf32_gemm(a32, b32)               # this crate's own mma.sync kernel
    ops.cublas_tf32_gemm(a32, b32)

Build (inside the fedora43-oxide distrobox; see tensor_core_engine_rust/RESULTS.md):
    cd tensor_core_engine_rust/rust_matlib && cargo oxide build --materialize-cubin --arch sm_86
"""
import ctypes
import glob
import os

# The py314 env bundles an older libcublasLt.so.13 (v13.1) that CuPy loads
# first; librust_matlib.so (like cuda_matlib.so) links the newer 13.x and then
# fails with "undefined symbol: cublasLtZZZMatmulAlgoGetHeuristicForStream".
# Loading the matching libraries here, BEFORE cupy is imported, makes the
# dynamic loader resolve that SONAME to them. Only effective if this module
# is imported before cupy (or the notebook preloads them itself, as
# benchmark_v5_1_A1000*.ipynb cell 0 does); harmless otherwise.
# cublas_preload resolves the directory (it is NOT always /usr/local/cuda-13.4:
# inside the `fedora` distrobox the host toolkit lives at /run/host/usr/local).
# Failure is not fatal here -- the notebook may have preloaded already, and the
# real error surfaces at CDLL(librust_matlib.so) time -- but do say so, because
# a silent miss turns into a confusing "undefined symbol" later.
try:
    from cublas_preload import preload_system_cublas
    _CUBLAS_DIR = preload_system_cublas(quiet=True)
except Exception as _e:  # noqa: BLE001 - diagnostics only
    _CUBLAS_DIR = None
    print(f"rust_matrix_ops: system cuBLAS preload skipped ({_e})")

import cupy as cp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.join(_HERE, "tensor_core_engine_rust", "rust_matlib", "target", "release")


def _default_lib():
    """Locate librust_matlib.so.

    `cargo oxide build --materialize-cubin` does its final link inside the build
    script's OUT_DIR, and does not always leave a copy at target/release/ the way
    a plain `cargo build` does. Prefer target/release when it is there, else fall
    back to the newest build-script output, so a fresh build is usable either way.
    """
    direct = os.path.join(_TARGET, "librust_matlib.so")
    if os.path.exists(direct):
        return direct
    outs = glob.glob(os.path.join(_TARGET, "build", "rust_matlib", "*", "out", "librust_matlib.so"))
    if not outs:
        raise RuntimeError(
            f"librust_matlib.so not found under {_TARGET} -- build it with\n"
            "  cd tensor_core_engine_rust/rust_matlib && "
            "cargo oxide build --materialize-cubin --arch sm_86"
        )
    return max(outs, key=os.path.getmtime)

_TC_MODES = {"rust": 0, "cublas": 1, "3xtf32": 2}
_BATCH_TF32, _BATCH_FP64, _BATCH_TC_SPLIT = 0, 1, 2


class RustMatrixOps:
    def __init__(self, lib_path=None):
        path = lib_path or _default_lib()
        self.lib = ctypes.CDLL(path)
        u64, i32 = ctypes.c_uint64, ctypes.c_int32
        sigs = {
            "rs_version": ([], ctypes.c_uint32),
            "rs_init": ([], i32),
            "rs_workspace_cleanup": ([], i32),
            # (a, c, n, power, mode) -- mode 0 = TF32, 1 = tc_split chain
            "rs_matrix_power": ([u64, u64, u64, i32, i32], i32),
            "rs_matmul_tf32": ([u64] * 6 + [i32, i32], i32),
            "rs_improved_matmul32": ([u64] * 6 + [i32, i32], i32),
            "rs_matmul_tc_split": ([u64] * 6 + [i32, u64, i32, i32], i32),
            "rs_batched_matmul": ([u64] * 7 + [i32, i32], i32),
            "rs_batched_matmul_bias": ([u64] * 8 + [i32], i32),
            # v5.3 FP32-in/FP32-out twin; see _batched_bias
            "rs_batched_matmul_bias_f32": ([u64] * 8 + [i32, i32, i32], i32),
            "rs_vector_matmul": ([u64] * 4 + [i32], i32),
            "rs_matmul_vector": ([u64] * 4 + [i32], i32),
            "rs_batched_vector_matmul": ([u64] * 5, i32),
            "rs_tf32_gemm": ([u64] * 6, i32),
            "rs_cublas_tf32_gemm": ([u64] * 6, i32),
        }
        for name, (argtypes, restype) in sigs.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = argtypes, restype
        rc = self.lib.rs_init()
        if rc != 0:
            raise RuntimeError(f"rs_init failed with code {rc}")
        self.version = self.lib.rs_version()

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _check(rc, what):
        if rc != 0:
            raise RuntimeError(f"{what} failed with code {rc} (see stderr)")

    @staticmethod
    def _dev(x, dtype=cp.float64):
        """Contiguous row-major CuPy array of `dtype` (no copy if already so)."""
        return cp.ascontiguousarray(cp.asarray(x, dtype=dtype))

    @staticmethod
    def _dev2(x):
        """2-D f64 operand without copying: returns (C-contiguous buffer, trans).
        An F-order array is handed over as its C-contiguous transpose view with
        trans=1, and the library consumes it in place via OP_T -- the
        counterpart of the Fortran wrapper's asfortranarray being a no-op on
        F-order inputs. Anything else becomes C-contiguous (copy if needed)."""
        x = cp.asarray(x, dtype=cp.float64)
        if x.ndim == 2 and x.flags.f_contiguous and not x.flags.c_contiguous:
            return x.T, 1
        return cp.ascontiguousarray(x), 0

    @staticmethod
    def _out(result, like):
        return cp.asnumpy(result) if isinstance(like, np.ndarray) else result

    @staticmethod
    def _sq_ptr(x):
        """(kept-alive array, device pointer, trans) for a square f64 operand.

        The pointer-only twin of _dev2, for the Dgemv paths. _dev2 hands back
        `x.T` for an F-order input, and building that view costs ~1.5 us of
        Python -- which is real money on a call whose GPU work is a few
        microseconds and whose total is ~20-35 us. A transpose view shares its
        base pointer, so for a square matrix the pointer and the shape are the
        same either way and only the trans flag is needed. Returns the array
        too, so the caller keeps the (possibly freshly copied) buffer alive
        until the library has read it.
        """
        x = cp.asarray(x, dtype=cp.float64)
        if x.ndim == 2 and x.flags.f_contiguous and not x.flags.c_contiguous:
            return x, x.data.ptr, 1
        x = cp.ascontiguousarray(x)
        return x, x.data.ptr, 0

    @staticmethod
    def _vec_ptr(x):
        """(kept-alive array, device pointer) for a 1-D f64 operand."""
        x = cp.asarray(x, dtype=cp.float64)
        if not x.flags.c_contiguous:
            x = cp.ascontiguousarray(x)
        return x, x.data.ptr

    # -- 2-D ---------------------------------------------------------------
    def matrix_power(self, a, power, mode="tf32"):
        """A^power on tensor cores.

        mode:
            "tf32"     — TF32 tensor cores per multiply (default). Fastest;
                         ~6e-5 to 1.7e-4 relative error, which compounds over
                         the chain, so a power of 4 is ~3x the error of a
                         single matmul.
            "tc_split" — every multiply goes through matmul_tc_split (1 exact
                         FP32 + 2 TF32 GEMMs). ~2.6e-7 to 4.8e-6 relative,
                         roughly 100x better, at 2.2-5x the cost. Intermediates
                         stay FP64 between multiplies, so nothing is lost
                         between steps. Measured on an RTX 4060 at power=4:
                         n=256 942 -> 417 GFLOPS, n=4048 23166 -> 4571.

        The chain is binary exponentiation either way, so A^4 costs 2 multiplies.
        """

        if mode not in ("tf32", "tc_split"):
            raise ValueError(f"mode must be 'tf32' or 'tc_split', got {mode!r}")
        if not isinstance(power, (int, np.integer)) or power < 1:
            raise ValueError(f"power must be an integer >= 1, got {power!r}")
        a_g, trans = self._dev2(a)  # F-order A: a_g = A^T (C-order), no copy
        n = a_g.shape[0]
        if a_g.shape != (n, n):
            raise ValueError("Input must be a square matrix")
        c = cp.empty((n, n), dtype=cp.float64)
        self._check(self.lib.rs_matrix_power(a_g.data.ptr, c.data.ptr, n, int(power), 0 if mode == "tf32" else 1), "rs_matrix_power")
        # (A^T)^p = (A^p)^T, so for an F-order input the C-order result viewed
        # transposed is A^p in F-order -- still no copy.
        return self._out(c.T if trans else c, a)

    @staticmethod
    def _mkn(a_g, at, b_g, bt):
        m, k = (a_g.shape[1], a_g.shape[0]) if at else a_g.shape
        k2, n = (b_g.shape[1], b_g.shape[0]) if bt else b_g.shape
        if k != k2:
            raise ValueError("inner dimensions must match")
        return m, k, n

    def matmul(self, a, b):
        a_g, at = self._dev2(a)
        b_g, bt = self._dev2(b)
        m, k, n = self._mkn(a_g, at, b_g, bt)
        c = cp.empty((m, n), dtype=cp.float64)
        self._check(self.lib.rs_matmul_tf32(a_g.data.ptr, b_g.data.ptr, c.data.ptr, m, k, n, at, bt), "rs_matmul_tf32")
        return self._out(c, a)

    def improved_matmul32(self, a, b):
        a_g, at = self._dev2(a)
        b_g, bt = self._dev2(b)
        m, k, n = self._mkn(a_g, at, b_g, bt)
        c = cp.empty((m, n), dtype=cp.float64)
        self._check(self.lib.rs_improved_matmul32(a_g.data.ptr, b_g.data.ptr, c.data.ptr, m, k, n, at, bt), "rs_improved_matmul32")
        return self._out(c, a)

    def improved_matmul64(self, a, b):
        return cp.matmul(cp.asarray(a, dtype=cp.float64), cp.asarray(b, dtype=cp.float64))

    def matmul_tc_split(self, a, b, mode="cublas", kchunk=0):
        if mode == "cublas":
            a_g, at = self._dev2(a)
            b_g, bt = self._dev2(b)
        else:  # the hand-written kernels and 3xTF32 need row-major buffers
            a_g, at = self._dev(a), 0
            b_g, bt = self._dev(b), 0
        m, k, n = self._mkn(a_g, at, b_g, bt)
        c = cp.empty((m, n), dtype=cp.float64)
        self._check(
            self.lib.rs_matmul_tc_split(a_g.data.ptr, b_g.data.ptr, c.data.ptr, m, k, n, _TC_MODES[mode], int(kchunk), at, bt),
            f"rs_matmul_tc_split[{mode}]",
        )
        return self._out(c, a)

    # -- batched -----------------------------------------------------------
    def _batched(self, a, b, mode):
        """Batched GEMM. C-order (batch,m,k) buffers are consumed in place via
        C^T = B^T A^T. F-order ones (asfortranarray, batch axis innermost) have
        no unit stride in either matrix dimension, so cuBLAS cannot read them
        directly; for the TF32 / tc_split paths the relayout is fused into the
        f64->f32 conversion kernel (one strided pass, no f64 temporary) instead
        of ascontiguousarray + convert. The FP64 path has no conversion pass to
        fuse into and copies. Output is always C-order (values == cp.matmul)."""
        a_g = cp.asarray(a, dtype=cp.float64)
        b_g = cp.asarray(b, dtype=cp.float64)
        if a_g.shape[:-2] != b_g.shape[:-2]:
            raise ValueError("Batch dimensions must match")
        lead = a_g.shape[:-2]
        batch = int(np.prod(lead)) if lead else 1
        m, k = a_g.shape[-2:]
        n = b_g.shape[-1]
        f_order = (mode != _BATCH_FP64 and a_g.ndim >= 3
                   and a_g.flags.f_contiguous and b_g.flags.f_contiguous and not a_g.flags.c_contiguous)
        if f_order:
            a3 = cp.reshape(a_g, (batch, m, k), order="F")
            b3 = cp.reshape(b_g, (batch, k, n), order="F")
            if not (a3.flags.f_contiguous and b3.flags.f_contiguous):
                f_order = False
        if not f_order:
            a3 = cp.ascontiguousarray(a_g.reshape(batch, m, k))
            b3 = cp.ascontiguousarray(b_g.reshape(batch, k, n))
        c = cp.empty((batch, m, n), dtype=cp.float64)
        self._check(
            self.lib.rs_batched_matmul(a3.data.ptr, b3.data.ptr, c.data.ptr, batch, m, k, n, mode, int(f_order)),
            "rs_batched_matmul",
        )
        if f_order and len(lead) > 1:
            # The F-order reshape merged the leading axes with the FIRST one
            # fastest (B = i0 + b*i1 + ...); a C-order reshape of the output
            # batch axis merges them the other way round, so reshape with the
            # leading axes reversed and transpose them back -- a view, no copy.
            nl = len(lead)
            c = c.reshape((*lead[::-1], m, n)).transpose(*range(nl - 1, -1, -1), nl, nl + 1)
            return self._out(c, a)
        return self._out(c.reshape((*lead, m, n)), a)

    def batched_matmul(self, a, b):
        return self._batched(a, b, _BATCH_TF32)

    def batched_matmul_fp64(self, a, b):
        return self._batched(a, b, _BATCH_FP64)

    def batched_matmul_tc_split(self, a, b):
        return self._batched(a, b, _BATCH_TC_SPLIT)

    @staticmethod
    def _all_f32(*arrays):
        return all(getattr(x, "dtype", None) == np.float32 for x in arrays)

    def _batched_bias_f32(self, a, b, bias, relu):
        """FP32 in, FP32 out -- no narrowing, no range guard, no widening.

        v5.3. The FP64 entry point converts both operands down and the result
        back up on every call, which is amortised for one big op but paid per
        layer in a chained model; `mlp_example_fortran_vs_rust_vs_cupy.ipynb`
        showed that costing more than the fused epilogue saves. Here the
        operands reach cuBLASLt untouched.

        **Layout follows the activation block.** cuBLASLt's bias epilogue only
        emits column-major, so a C-order caller needs a transpose pass on the
        way out -- measured at ~1.3 ms for (1024 x 16384) f32, against the
        ~1.0 ms bias+ReLU pass the fusion saves, which is most of why the fused
        path did not beat unfused CuPy. Hand `b` in as an **F-order** 2-D array
        and that pass disappears: B is described column-major with TRANSB=N,
        the epilogue writes straight into an F-order result, and the result is
        itself a valid F-order `b` for the next layer. Weights stay C-order
        either way.
        """
        two_d = getattr(b, "ndim", None) == 2
        b_cm = bool(two_d and getattr(b, "flags", None) is not None
                    and b.flags.f_contiguous and not b.flags.c_contiguous)

        a_g = cp.ascontiguousarray(cp.asarray(a, dtype=cp.float32))
        bias_g = cp.ascontiguousarray(cp.asarray(bias, dtype=cp.float32))
        b_g = cp.asarray(b, dtype=cp.float32) if b_cm else cp.ascontiguousarray(cp.asarray(b, dtype=cp.float32))

        lead = a_g.shape[:-2]
        batch = int(np.prod(lead)) if lead else 1
        m, k = a_g.shape[-2:]
        n = b_g.shape[-1]
        if bias_g.shape != (m,):
            raise ValueError(f"bias must have shape ({m},), got {bias_g.shape}")
        if b_cm and batch != 1:
            raise ValueError("F-order activations are only supported for a single batch")
        a3 = a_g.reshape(batch, m, k)
        b3 = b_g if b_cm else b_g.reshape(batch, k, n)
        c = (cp.empty((m, n), dtype=cp.float32, order="F") if b_cm
             else cp.empty((batch, m, n), dtype=cp.float32))
        self._check(
            self.lib.rs_batched_matmul_bias_f32(
                a3.data.ptr, b3.data.ptr, bias_g.data.ptr, c.data.ptr,
                batch, m, k, n, int(relu), int(b_cm), int(b_cm)),
            "rs_batched_matmul_bias_f32",
        )
        if b_cm:
            return self._out(c, a)
        return self._out(c.reshape((*lead, m, n)), a)

    def _batched_bias(self, a, b, bias, relu):
        # float32 in -> float32 out, through the v5.3 conversion-free path.
        if self._all_f32(a, b, bias):
            return self._batched_bias_f32(a, b, bias, relu)
        a_g, b_g, bias_g = self._dev(a), self._dev(b), self._dev(bias)
        lead = a_g.shape[:-2]
        batch = int(np.prod(lead)) if lead else 1
        m, k = a_g.shape[-2:]
        n = b_g.shape[-1]
        if bias_g.shape != (m,):
            raise ValueError(f"bias must have shape ({m},), got {bias_g.shape}")
        a3 = a_g.reshape(batch, m, k)
        b3 = b_g.reshape(batch, k, n)
        c = cp.empty((batch, m, n), dtype=cp.float64)
        self._check(
            self.lib.rs_batched_matmul_bias(a3.data.ptr, b3.data.ptr, bias_g.data.ptr, c.data.ptr, batch, m, k, n, int(relu)),
            "rs_batched_matmul_bias",
        )
        return self._out(c.reshape((*lead, m, n)), a)

    def batched_matmul_bias_relu(self, a, b, bias):
        return self._batched_bias(a, b, bias, True)

    def batched_matmul_bias(self, a, b, bias):
        return self._batched_bias(a, b, bias, False)

    def strided_batch_matmul(self, m, k, n, batch_size, a, b):
        """Same as batched_matmul, with the dimensions given explicitly.

        The shape check must NOT go through _dev(): that calls
        ascontiguousarray, which for an F-order input is a full FP64 relayout
        -- and its result was thrown away, because _batched() is handed the
        ORIGINAL arrays and does its own (fused, in-kernel) layout handling.
        Measured at (8, 2048^3) F-order: 25.4 ms of wasted copying on a call
        that otherwise takes 11.3 ms. Shapes alone need no materialisation.
        """
        sa = tuple(a.shape) if hasattr(a, "shape") else tuple(np.shape(a))
        sb = tuple(b.shape) if hasattr(b, "shape") else tuple(np.shape(b))
        if sa != (batch_size, m, k) or sb != (batch_size, k, n):
            raise ValueError("a must be (batch,m,k) and b (batch,k,n)")
        return self._batched(a, b, _BATCH_TF32)

    def tensor_4d_matmul(self, a, b):
        """(b1, b2, m, n) @ (b1, b2, n, n) -> (b1, b2, m, n), TF32 batched."""
        return self._batched(a, b, _BATCH_TF32)

    def tensor_5d_matmul(self, a, b):
        """(b, c, d, h, w) @ (b, c, d, w, nw) -> (b, c, d, h, nw), TF32 batched."""
        return self._batched(a, b, _BATCH_TF32)

    # -- vectors -----------------------------------------------------------
    def vector_matmul(self, v, a):
        v_g, pv = self._vec_ptr(v)
        a_g, pa, at = self._sq_ptr(a)
        n = v_g.shape[0]
        if a_g.shape != (n, n):
            raise ValueError("a must be (n, n) with n = len(v)")
        y = cp.empty(n, dtype=cp.float64)
        self._check(self.lib.rs_vector_matmul(pv, pa, y.data.ptr, n, at), "rs_vector_matmul")
        return self._out(y, v)

    def matmul_vector(self, a, v):
        a_g, pa, at = self._sq_ptr(a)
        v_g, pv = self._vec_ptr(v)
        n = v_g.shape[0]
        if a_g.shape != (n, n):
            raise ValueError("a must be (n, n) with n = len(v)")
        y = cp.empty(n, dtype=cp.float64)
        self._check(self.lib.rs_matmul_vector(pa, pv, y.data.ptr, n, at), "rs_matmul_vector")
        return self._out(y, a)

    def batched_vector_matmul(self, v, a):
        """v: (n, batch) columns are vectors; returns (batch, n) with row i = a @ v[:, i]."""
        v_g, a_g = self._dev(v), self._dev(a)
        n, batch = v_g.shape
        if a_g.shape != (n, n):
            raise ValueError("a must be (n, n) with n = v.shape[0]")
        y = cp.empty((batch, n), dtype=cp.float64)
        self._check(self.lib.rs_batched_vector_matmul(v_g.data.ptr, a_g.data.ptr, y.data.ptr, n, batch), "rs_batched_vector_matmul")
        return self._out(y, v)

    # -- f32 single GEMMs (this crate's own kernel vs cuBLAS) ----------------
    def tf32_gemm(self, a, b):
        a_g, b_g = self._dev(a, cp.float32), self._dev(b, cp.float32)
        m, k = a_g.shape
        _, n = b_g.shape
        c = cp.empty((m, n), dtype=cp.float32)
        self._check(self.lib.rs_tf32_gemm(a_g.data.ptr, b_g.data.ptr, c.data.ptr, m, k, n), "rs_tf32_gemm")
        return self._out(c, a)

    def cublas_tf32_gemm(self, a, b):
        a_g, b_g = self._dev(a, cp.float32), self._dev(b, cp.float32)
        m, k = a_g.shape
        _, n = b_g.shape
        c = cp.empty((m, n), dtype=cp.float32)
        self._check(self.lib.rs_cublas_tf32_gemm(a_g.data.ptr, b_g.data.ptr, c.data.ptr, m, k, n), "rs_cublas_tf32_gemm")
        return self._out(c, a)

    def workspace_cleanup(self):
        self._check(self.lib.rs_workspace_cleanup(), "rs_workspace_cleanup")
