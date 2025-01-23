# fortran_tensor_ops.py; save this file in the same directory to make it run
# in the entire Jupyter notebook
# # At the start of your notebook
#       from fortran_tensor_ops import FortranTensorOps
# # Create global instance
#       tensor_ops = FortranTensorOps()
#
# python wrapper to be more pythonic with cuBLAS kernel:
import ctypes
from ctypes import c_int, c_void_p, c_double
import cupy as cp
import numpy as np
from numpy.ctypeslib import ndpointer  # This is the critical import

class FortranTensorOps:
    """High-level interface for Fortran-based tensor operations on GPU."""

    def __init__(self, lib_path='./libcudamatmul_tensor8.so'):
        """Initialize the tensor operations interface.

        Args:
            lib_path: Path to the compiled Fortran library
        """
        self.lib = ctypes.CDLL(lib_path)
        self._setup_function_signatures()

    def _setup_function_signatures(self):
        """Setup C function signatures for the Fortran interface."""
        # Setup exactly like original working version
        double_2d = ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')

        self.lib.py_matrix_dot1.argtypes = [
            double_2d,  # Input array
            double_2d,  # Output array
            c_int,      # Size
            c_int       # Iterations
        ]
        self.lib.py_matrix_dot1.restype = None
        self.lib.py_matrix_dot2.argtypes = [
            double_2d,  # Input array
            double_2d,  # Output array
            c_int,      # Size
            c_int       # Iterations
        ]
        self.lib.py_matrix_dot2.restype = None
        self.lib.py_matrix_dot.argtypes = [
            double_2d,  # Input array
            double_2d,  # Output array
            c_int,      # Size
            c_int       # Iterations
        ]
        self.lib.py_matrix_dot.restype = None


        # GPU memory pointer based operations
        self.lib.py_tensor_matrix_multiply.argtypes = [
            c_void_p,    # a ptr
            c_void_p,    # b ptr
            c_void_p,    # c ptr
            c_int,       # m
            c_int,       # k
            c_int        # n
        ]
        self.lib.py_tensor_matrix_multiply.restype = c_int  # Return error code

        self.lib.py_batched_vector_matrix_dot.argtypes = [
            c_void_p,    # vec ptr
            c_void_p,    # mat ptr
            c_void_p,    # result ptr
            c_int,       # batch_size
            c_int,       # vec_len
            c_int        # mat_cols
        ]
        self.lib.py_batched_vector_matrix_dot.restype = c_int  # Return error code

        self.lib.py_vector_matrix_dot.argtypes = [
            c_void_p,    # vec ptr
            c_void_p,    # mat ptr
            c_void_p,    # result ptr
            c_int,       # vec_len
            c_int        # mat_cols
        ]
        self.lib.py_vector_matrix_dot.restype = c_int  # Return error code

    def _enhanced_prepare_array(self, arr, dtype=None, expected_shape=None, transpose=False):
        """Enhanced array preparation with comprehensive safety checks.

        Args:
            arr: Input array
            dtype: Target data type
            expected_shape: Optional tuple for shape validation
            transpose: Whether to transpose the array
        """
        # Basic type conversion
        if not isinstance(arr, cp.ndarray):
            arr = cp.asarray(arr)

        # Data type conversion
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype)

        # Handle NaN and Inf values
        if cp.any(cp.isnan(arr)) or cp.any(cp.isinf(arr)):
            print(f"WARNING: Input array contains NaN/Inf values. Cleaning...")
            arr = cp.nan_to_num(arr, nan=0.0, posinf=1e300, neginf=-1e300)

        # Shape validation
        if expected_shape is not None:
            if arr.shape != expected_shape:
                raise ValueError(f"Array shape mismatch. Expected {expected_shape}, got {arr.shape}")

        # Transposition if needed
        if transpose:
            arr = arr.T

        # Ensure Fortran contiguous memory layout
        if not arr.flags.f_contiguous:
            arr = cp.asfortranarray(arr)

        return arr

    def _prepare_array(self, arr, dtype=None):
        """Prepare array for Fortran kernel.

        Args:
            arr: CuPy array to prepare
            dtype: Optional dtype to cast to

        Returns:
            CuPy array in Fortran order with specified dtype
        """
        if not isinstance(arr, cp.ndarray):
            arr = cp.asarray(arr)

        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype)

        if not arr.flags.f_contiguous:
            arr = cp.asfortranarray(arr)

        return arr

    def prepare_array_for_kernel(arr, transpose=False, expected_shape=None):
        """
        Prepare a CuPy array for Fortran kernel processing with numerical checks
        """
        # Check for NaNs or Infs in input
        if cp.any(cp.isnan(arr)) or cp.any(cp.isinf(arr)):
            print(f"WARNING: Input array contains NaN/Inf before preparation")
            print(f"Min: {float(cp.nanmin(arr))}, Max: {float(cp.nanmax(arr))}")
            # Replace NaNs with zeros and clip Infs
            arr = cp.nan_to_num(arr, nan=0.0, posinf=1e300, neginf=-1e300)

        if transpose:
            arr = arr.T

        if expected_shape:
            arr = arr.reshape(expected_shape, order='F')

        if not arr.flags.f_contiguous:
            arr = cp.asfortranarray(arr)

        # Verify array after preparation
        if cp.any(cp.isnan(arr)) or cp.any(cp.isinf(arr)):
            print(f"WARNING: Array contains NaN/Inf after preparation")

        #print(f"Prepared array: shape={arr.shape}, f_contiguous={arr.flags.f_contiguous}")
        #print(f"Range: [{float(cp.min(arr))}, {float(cp.max(arr))}]")
        return arr


    def matrix_dot1(self, a, iterations=1):
        """Simplified matrix dot implementation"""
        # Convert CuPy to NumPy if needed
        if isinstance(a, cp.ndarray):
            a = cp.asnumpy(a)

        # Basic validation
        if not isinstance(a, np.ndarray):
            raise TypeError("Input must be a numpy array")

        # Ensure correct format
        a = np.ascontiguousarray(a, dtype=np.float64)

        if a.shape[0] != a.shape[1]:
            raise ValueError("Matrix must be square")

        n = a.shape[0]
        result = np.empty_like(a, dtype=np.float64, order='C')

        try:
            # Call Fortran function
            self.lib.py_matrix_dot1(a, result, n, iterations)
        except Exception as e:
            print(f"Error calling matrix_dot1: {e}")
            print(f"Input array info:")
            print(f"Shape: {a.shape}")
            print(f"dtype: {a.dtype}")
            print(f"C_CONTIGUOUS: {a.flags['C_CONTIGUOUS']}")
            raise

        return result

    def matrix_dot2(self, a, iterations=1):
        """Simplified matrix dot implementation"""
        # Convert CuPy to NumPy if needed
        if isinstance(a, cp.ndarray):
            a = cp.asnumpy(a)

        # Basic validation
        if not isinstance(a, np.ndarray):
            raise TypeError("Input must be a numpy array")

        # Ensure correct format
        a = np.ascontiguousarray(a, dtype=np.float64)

        if a.shape[0] != a.shape[1]:
            raise ValueError("Matrix must be square")

        n = a.shape[0]
        result = np.empty_like(a, dtype=np.float64, order='C')

        try:
            # Call Fortran function
            self.lib.py_matrix_dot2(a, result, n, iterations)
        except Exception as e:
            print(f"Error calling matrix_dot1: {e}")
            print(f"Input array info:")
            print(f"Shape: {a.shape}")
            print(f"dtype: {a.dtype}")
            print(f"C_CONTIGUOUS: {a.flags['C_CONTIGUOUS']}")
            raise

        return result

    def matrix_dot(self, a, iterations=1):
        """Simplified matrix dot implementation"""
        # Convert CuPy to NumPy if needed
        if isinstance(a, cp.ndarray):
            a = cp.asnumpy(a)

        # Basic validation
        if not isinstance(a, np.ndarray):
            raise TypeError("Input must be a numpy array")

        # Ensure correct format
        a = np.ascontiguousarray(a, dtype=np.float64)

        if a.shape[0] != a.shape[1]:
            raise ValueError("Matrix must be square")

        n = a.shape[0]
        result = np.empty_like(a, dtype=np.float64, order='C')

        try:
            # Call Fortran function
            self.lib.py_matrix_dot(a, result, n, iterations)
        except Exception as e:
            print(f"Error calling matrix_dot1: {e}")
            print(f"Input array info:")
            print(f"Shape: {a.shape}")
            print(f"dtype: {a.dtype}")
            print(f"C_CONTIGUOUS: {a.flags['C_CONTIGUOUS']}")
            raise

        return result


    def matmul(self, a, b, out=None):
        """Enhanced matrix multiplication with improved stability.

        Args:
            a: First input matrix (m, k)
            b: Second input matrix (k, n)
            out: Optional output array (m, n)
        """
        # Input validation
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"Inputs must be 2D arrays. Got shapes {a.shape} and {b.shape}")

        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Incompatible matrix shapes for multiplication: {a.shape} and {b.shape}")

        # Prepare input arrays with safety checks
        try:
            a = self._enhanced_prepare_array(a, dtype=cp.float64)
            b = self._enhanced_prepare_array(b, dtype=cp.float64)
        except Exception as e:
            raise ValueError(f"Error preparing input arrays: {str(e)}")

        # Create or validate output array
        m, k = a.shape
        _, n = b.shape

        if out is None:
            try:
                out = cp.empty((m, n), dtype=cp.float64, order='F')
            except Exception as e:
                raise RuntimeError(f"Failed to allocate output array: {str(e)}")
        else:
            if not isinstance(out, cp.ndarray):
                raise TypeError("Output must be a CuPy array")
            if out.shape != (m, n):
                raise ValueError(f"Output shape mismatch. Expected ({m}, {n}), got {out.shape}")
            out = self._enhanced_prepare_array(out, dtype=cp.float64)

        # Synchronize before kernel call
        cp.cuda.Stream.null.synchronize()

        try:
            # Call Fortran kernel
            self.lib.py_tensor_matrix_multiply(
                a.data.ptr,
                b.data.ptr,
                out.data.ptr,
                m,
                k,
                n
            )
        except Exception as e:
            raise RuntimeError(f"CUDA kernel execution failed: {str(e)}")

        # Synchronize after kernel call
        cp.cuda.Stream.null.synchronize()

        return out

    def vector_matmul(self, vector, matrix, out=None):
        """Enhanced vector-matrix multiplication with improved stability.

        Args:
            vector: Input vector (vec_len,)
            matrix: Input matrix (vec_len, mat_cols)
            out: Optional output array (mat_cols,)
        """
        # Input validation
        if vector.ndim != 1:
            raise ValueError(f"Vector must be 1D array. Got shape {vector.shape}")
        if matrix.ndim != 2:
            raise ValueError(f"Matrix must be 2D array. Got shape {matrix.shape}")
        if vector.shape[0] != matrix.shape[0]:
            raise ValueError(f"Incompatible shapes: vector {vector.shape} and matrix {matrix.shape}")

        # Prepare input arrays with safety checks
        try:
            vector = self._enhanced_prepare_array(vector, dtype=cp.float64)
            matrix = self._enhanced_prepare_array(matrix, dtype=cp.float64)
        except Exception as e:
            raise ValueError(f"Error preparing input arrays: {str(e)}")

        vec_len = vector.shape[0]
        mat_cols = matrix.shape[1]

        # Create or validate output array
        if out is None:
            try:
                out = cp.empty(mat_cols, dtype=cp.float64, order='F')
            except Exception as e:
                raise RuntimeError(f"Failed to allocate output array: {str(e)}")
        else:
            if not isinstance(out, cp.ndarray):
                raise TypeError("Output must be a CuPy array")
            if out.shape != (mat_cols,):
                raise ValueError(f"Output shape mismatch. Expected ({mat_cols},), got {out.shape}")
            out = self._enhanced_prepare_array(out, dtype=cp.float64)

        # Synchronize before kernel call
        cp.cuda.Stream.null.synchronize()

        try:
            # Call Fortran kernel
            self.lib.py_vector_matrix_dot(
                vector.data.ptr,
                matrix.data.ptr,
                out.data.ptr,
                vec_len,
                mat_cols
            )
        except Exception as e:
            raise RuntimeError(f"CUDA kernel execution failed: {str(e)}")

        # Synchronize after kernel call
        cp.cuda.Stream.null.synchronize()

        return out

    def vector_matrix_dot(self, vec, mat, result=None):
        """
        Wrapper for vector-matrix multiplication using tensor ops.
        Args:
            vec: CuPy array (vec_len) in Fortran order
            mat: CuPy array (vec_len, mat_cols) in Fortran order
            result: Optional CuPy array (mat_cols) in Fortran order

        Returns:
            Result of vector-matrix multiplication
        """
        vec = cp.asfortranarray(vec)
        mat = cp.asfortranarray(mat)

        # If no result is provided, create an empty output array
        if result is None:
            result = cp.empty(mat.shape[1], dtype=cp.float64, order='F')
        else:
            result = cp.asfortranarray(result)

        # Call the vector-matrix multiplication method
        self.vector_matmul(
            vec,           # (vec_len)
            mat,           # (vec_len, mat_cols)
            out=result     # (mat_cols)
        )

        return result

    def batched_matrix_multiply(self, a, b, out=None):
        """Sequential batch matrix multiplication processing one batch at a time.

        Args:
            a: Input batched matrices (m, k, batch_size)
            b: Weight matrix (k, n)
            out: Optional output array (m, n, batch_size)
        """
        # Input validation
        if a.ndim != 3:
            raise ValueError(f"Batched input 'a' must be 3D array. Got shape {a.shape}")
        if b.ndim != 2:
            raise ValueError(f"Matrix 'b' must be 2D array. Got shape {b.shape}")

        # Extract dimensions
        m, k, batch_size = a.shape
        if b.shape[0] != k:
            raise ValueError(f"Incompatible shapes: batched matrices {a.shape} and matrix {b.shape}")
        n = b.shape[1]

        # Create output array
        if out is None:
            out = cp.empty((m, n, batch_size), dtype=cp.float64, order='F')
        else:
            if not isinstance(out, cp.ndarray):
                raise TypeError("Output must be a CuPy array")
            if out.shape != (m, n, batch_size):
                raise ValueError(f"Output shape mismatch")
            out = cp.asfortranarray(out.astype(cp.float64))

        # Process each batch individually
        b_aligned = cp.asfortranarray(b.astype(cp.float64))

        for i in range(batch_size):
            # Get current batch and ensure alignment
            a_batch = cp.asfortranarray(a[:, :, i].astype(cp.float64))

            try:
                # Use tensor_matrix_multiply for each batch
                err = self.lib.py_tensor_matrix_multiply(
                    a_batch.data.ptr,      # Current batch
                    b_aligned.data.ptr,    # Weight matrix
                    out[:, :, i].data.ptr, # Current output batch
                    m,
                    k,
                    n
                )

                if err != 0:
                    raise RuntimeError(f"CUDA kernel returned error code: {err} for batch {i}")

            except Exception as e:
                print(f"Error processing batch {i}: {str(e)}")
                raise

            # Ensure batch is complete before moving to next
            cp.cuda.Stream.null.synchronize()

        return out

    # stable
    def batched_vector_matmul(self, vectors, matrix, out=None):
        """Batched vector-matrix multiplication.

        Args:
            vectors: Input vectors of shape (vec_len, batch_size)
            matrix: Input matrix of shape (vec_len, mat_cols)
            out: Optional output array

        Returns:
            Result of shape (batch_size, mat_cols)
        """
        # Validate input shapes
        if vectors.shape[0] != matrix.shape[0]:
            raise ValueError(f"Incompatible shapes: {vectors.shape} and {matrix.shape}")

        # Prepare inputs
        vectors = self._prepare_array(vectors, cp.float64)
        matrix = self._prepare_array(matrix, cp.float64)

        batch_size = vectors.shape[1]
        vec_len = vectors.shape[0]
        mat_cols = matrix.shape[1]

        # Create or validate output array
        if out is None:
            out = cp.empty((batch_size, mat_cols), dtype=cp.float64, order='F')
        else:
            if not isinstance(out, cp.ndarray):
                raise TypeError("Output array must be a CuPy array")
            if out.shape != (batch_size, mat_cols):
                raise ValueError(f"Output shape mismatch: expected {(batch_size, mat_cols)}, got {out.shape}")
            out = self._prepare_array(out, cp.float64)

        # Call Fortran kernel
        self.lib.py_batched_vector_matrix_dot(
            vectors.data.ptr,
            matrix.data.ptr,
            out.data.ptr,
            batch_size,
            vec_len,
            mat_cols
        )

        return out
# this version 1 now has all kernels working (even if there are kludges); accuracy is good
# we will fix things and then use CUDA streams to avoid 1000 kernel launches and just use 1 launch 1000 times!
