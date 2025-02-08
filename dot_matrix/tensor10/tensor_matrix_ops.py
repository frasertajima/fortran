import numpy as np
import cupy as cp
import ctypes
import time

class TensorMatrixOps:
    """GPU-accelerated matrix operations using tensor cores.

    Provides high-performance matrix operations using CUDA Fortran tensor cores:
    - Matrix multiplication (A*B)
    - Matrix power (A^n)
    - Batched matrix multiply
    - Vector-matrix multiply
    - Matrix-vector multiply
    - Batched vector multiply
    - Strided batch multiply
    """

    def __init__(self, lib_path='./cuda_matlib.so'):
        """Initialize the tensor matrix operations library."""
        self.lib = ctypes.CDLL(lib_path)

        # Initialize cuBLAS and CUDA resources done by fortran

        # Set up all function signatures
        self.lib.py_matrix_dot.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int
        ]

        self.lib.py_tensor_matrix_multiply.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        self.lib.py_batched_matmul.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        self.lib.py_vector_matrix_multiply.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int
        ]

        self.lib.py_matrix_vector_multiply.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int
        ]

        self.lib.py_batched_vector_multiply.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int
        ]

        self.lib.py_strided_batch_multiply.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

    def matrix_power(self, a, power):
        """Compute matrix power A^n using tensor cores.

        Args:
            a: Square matrix (numpy or cupy array)
            power: Integer power to raise matrix to

        Returns:
            Result of A^power, same type as input
        """
        if not isinstance(a, (np.ndarray, cp.ndarray)):
            raise TypeError("Input must be numpy or cupy array")

        a_gpu = cp.asarray(a, dtype=cp.float64)
        if a_gpu.shape[0] != a_gpu.shape[1]:
            raise ValueError("Matrix must be square")

        n = a_gpu.shape[0]
        c_gpu = cp.empty_like(a_gpu)

        self.lib.py_matrix_dot(
            ctypes.c_void_p(a_gpu.data.ptr),
            ctypes.c_void_p(c_gpu.data.ptr),
            n, power
        )

        return cp.asnumpy(c_gpu) if isinstance(a, np.ndarray) else c_gpu

    # works!!
    def matmul(self, a, b):
        """Compute matrix multiplication with Fortran BLAS conventions.
        Args:
            a: Matrix (batch_size x hidden_size)
            b: Matrix (hidden_size x output_size)
        """
        # Convert to Fortran ordering
        a_f = cp.asfortranarray(a, dtype=cp.float64)  # (M x K)
        b_f = cp.asfortranarray(b, dtype=cp.float64)  # (K x N)

        M, K = a_f.shape
        _, N = b_f.shape

        # Create output array in Fortran order
        c_f = cp.empty((M, N), dtype=cp.float64, order='F')

        #print(f"DEBUG:")
        #print(f"a shape: {a_f.shape}, order F: {a_f.flags.f_contiguous}")
        #print(f"b shape: {b_f.shape}, order F: {b_f.flags.f_contiguous}")
        #print(f"c shape: {c_f.shape}, order F: {c_f.flags.f_contiguous}")
        #print(f"M={M}, K={K}, N={N}")

        # BLAS DGEMM: C = alpha*A*B + beta*C
        self.lib.py_tensor_matrix_multiply(
            ctypes.c_void_p(a_f.data.ptr),
            ctypes.c_void_p(b_f.data.ptr),
            ctypes.c_void_p(c_f.data.ptr),
            ctypes.c_int(M),   # rows of A
            ctypes.c_int(K),   # cols of A = rows of B
            ctypes.c_int(N),   # cols of B
            ctypes.c_int(1)    # iterations
        )

        return c_f  # Already in correct shape
    # works!
    def batched_vector_matmul(self, v, a):
        """Compute batched vector-matrix multiplication matching Fortran dimensions.
        Args:
            v: Batch of vectors (input_size x batch_size)
            a: Matrix (input_size x hidden_size)
        """
        input_size, batch_size = v.shape
        _, hidden_size = a.shape

        print(f"DEBUG batched dimensions:")
        print(f"v: {v.shape} - (input_size x batch_size)")
        print(f"a: {a.shape} - (input_size x hidden_size)")

        # Need to pad matrix 'a' to be square (n x n) as expected by Fortran
        n = max(input_size, hidden_size)
        a_padded = cp.zeros((n, n), dtype=cp.float64, order='F')
        a_padded[:input_size, :hidden_size] = a

        # Convert vectors to Fortran order
        v_f = cp.asfortranarray(v, dtype=cp.float64)

        # Create output array matching Fortran expectations
        c_f = cp.empty((n, batch_size), dtype=cp.float64, order='F')

        print(f"Padded dimensions:")
        print(f"a_padded: {a_padded.shape} - (n x n)")
        print(f"v_f: {v_f.shape} - (n x batch_size)")
        print(f"c_f: {c_f.shape} - (n x batch_size)")

        try:
            self.lib.py_batched_vector_multiply(
                ctypes.c_void_p(v_f.data.ptr),
                ctypes.c_void_p(a_padded.data.ptr),
                ctypes.c_void_p(c_f.data.ptr),
                ctypes.c_int(n),
                ctypes.c_int(batch_size)
            )

            # Extract the relevant part of the result and reshape
            result = c_f[:hidden_size, :].T  # Transpose to get (batch_size x hidden_size)
            return cp.asfortranarray(result)

        except Exception as e:
            print(f"Error in batched_vector_matmul: {e}")
            print(f"Padded matrix:\n{a_padded}")
            print(f"Input vectors:\n{v_f}")
            raise

    def batched_matmul(self, a, b):
        """Compute batched matrix multiplication.

        Args:
            a: Batch of matrices (..., m, k)
            b: Batch of matrices (..., k, n)

        Returns:
            Batch of matrix products (..., m, n)
        """
        if a.shape[:-2] != b.shape[:-2]:
            raise ValueError("Batch dimensions must match")

        batch_size = np.prod(a.shape[:-2])
        m, k = a.shape[-2:]
        n = b.shape[-1]

        a_gpu = cp.asarray(a.reshape(batch_size, m, k))
        b_gpu = cp.asarray(b.reshape(batch_size, k, n))
        c_gpu = cp.empty((batch_size, m, n), dtype=cp.float64)

        self.lib.py_batched_matmul(
            ctypes.c_void_p(a_gpu.data.ptr),
            ctypes.c_void_p(b_gpu.data.ptr),
            ctypes.c_void_p(c_gpu.data.ptr),
            m, k, n, batch_size
        )

        result = cp.asnumpy(c_gpu) if isinstance(a, np.ndarray) else c_gpu
        return result.reshape((*a.shape[:-2], m, n))

    def vector_matmul(self, v, a):
        """Compute vector-matrix multiplication (v*A).

        Args:
            v: Input vector (n,)
            a: Input matrix (n x n)

        Returns:
            Result vector (n,)
        """
        v_gpu = cp.asarray(v, dtype=cp.float64)
        a_gpu = cp.asarray(a, dtype=cp.float64)

        if v_gpu.shape[0] != a_gpu.shape[0]:
            raise ValueError("Dimensions must match")

        n = v_gpu.shape[0]
        c_gpu = cp.empty(n, dtype=cp.float64)

        self.lib.py_vector_matrix_multiply(
            ctypes.c_void_p(v_gpu.data.ptr),
            ctypes.c_void_p(a_gpu.data.ptr),
            ctypes.c_void_p(c_gpu.data.ptr),
            n
        )

        return cp.asnumpy(c_gpu) if isinstance(v, np.ndarray) else c_gpu

    def matmul_vector(self, a, v):
        """Compute matrix-vector multiplication (A*v).

        Args:
            a: Input matrix (n x n)
            v: Input vector (n,)

        Returns:
            Result vector (n,)
        """
        a_gpu = cp.asarray(a, dtype=cp.float64)
        v_gpu = cp.asarray(v, dtype=cp.float64)

        if a_gpu.shape[1] != v_gpu.shape[0]:
            raise ValueError("Dimensions must match")

        n = v_gpu.shape[0]
        c_gpu = cp.empty(n, dtype=cp.float64)

        self.lib.py_matrix_vector_multiply(
            ctypes.c_void_p(a_gpu.data.ptr),
            ctypes.c_void_p(v_gpu.data.ptr),
            ctypes.c_void_p(c_gpu.data.ptr),
            n
        )

        return cp.asnumpy(c_gpu) if isinstance(a, np.ndarray) else c_gpu

    def strided_batch_matmul(self, a, b, m, k, n, batch_size):
        """Compute strided batch matrix multiplication.

        Args:
            a: Input matrices in strided format
            b: Input matrices in strided format
            m, k, n: Matrix dimensions
            batch_size: Number of matrices in batch

        Returns:
            Batch of matrix products in strided format
        """
        a_gpu = cp.asarray(a, dtype=cp.float64)
        b_gpu = cp.asarray(b, dtype=cp.float64)

        stride_a = m * k
        stride_b = k * n
        stride_c = m * n

        c_gpu = cp.empty(batch_size * stride_c, dtype=cp.float64)

        self.lib.py_strided_batch_multiply(
            ctypes.c_void_p(a_gpu.data.ptr),
            ctypes.c_void_p(b_gpu.data.ptr),
            ctypes.c_void_p(c_gpu.data.ptr),
            m, k, n, batch_size,
            stride_a, stride_b, stride_c
        )

        result = cp.asnumpy(c_gpu) if isinstance(a, np.ndarray) else c_gpu
        return result.reshape(batch_size, m, n)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # Cleanup handled by CUDA Fortran

    def __del__(self):
        pass # cleanup handled by fortran
