import numpy as np
import cupy as cp
import ctypes
import time

# Define module-level variables for the singleton pattern
_CUDA_INITIALIZED = False
_LIB_INSTANCE = None

# Add this method after the imports but before the TensorMatrixOps class
def _get_singleton_library(lib_path):
    """Get or initialize the singleton library instance."""
    global _CUDA_INITIALIZED, _LIB_INSTANCE

    # If already initialized, return the existing instance
    if _CUDA_INITIALIZED and _LIB_INSTANCE is not None:
        return _LIB_INSTANCE

    # Otherwise, load the library
    try:
        print(f"Loading library: {lib_path}")
        _LIB_INSTANCE = ctypes.CDLL(lib_path)
        return _LIB_INSTANCE
    except Exception as e:
        print(f"Error loading library: {e}")
        raise

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


    # Then modify your __init__ method to use the function:
    def __init__(self, lib_path='./cuda_matlib.so', tensor_cores=None):
        """Initialize with proper device capability detection and tensor core configuration.

        Args:
            lib_path: Path to the compiled CUDA Fortran library
            tensor_cores: Number of tensor cores to use. None means use default.
        """
        global _CUDA_INITIALIZED

        try:
            # Initialize CUDA first
            self._init_cuda()

            # Get the singleton library instance
            self.lib = _get_singleton_library(lib_path)

            # Set up function signatures first
            self._setup_functions()

            # Initialize CUDA and cuBLAS first - but only once across all instances
            if not _CUDA_INITIALIZED:
                print("Initializing CUDA resources (one-time operation)...")
                self.lib.py_initialize_cuda_resources()
                _CUDA_INITIALIZED = True
                print("CUDA resources initialized")

        except Exception as e:
            print(f"Initialization error: {e}")
            raise

    def _init_cuda(self):
        """Initialize CUDA runtime and memory pools."""
        print("Initializing CUDA...")
        self.device = cp.cuda.Device(0)
        self.device.use()
        # Initialize memory pools
        self.mempool = cp.get_default_memory_pool()
        self.pinned_mempool = cp.get_default_pinned_memory_pool()
        self.stream = cp.cuda.Stream()
        print("CUDA initialization complete")

        # Initialize cuBLAS and CUDA resources done by fortran

    def _setup_functions(self):
        """Configure library function signatures."""
        # Core device functions
        self.lib.py_initialize_cuda_resources.argtypes = []
        self.lib.py_initialize_cuda_resources.restype = None

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

        # 4D tensor operation
        self.lib.py_tensor_4d_matmul.argtypes = [
            ctypes.c_void_p,    # Input tensor A
            ctypes.c_void_p,    # Input tensor B (weight)
            ctypes.c_void_p,    # Output tensor C
            ctypes.c_int,       # batch_size
            ctypes.c_int        # n (Matrix dimension)
        ]

        # 5D tensor operation - corrected to match Fortran
        self.lib.py_tensor_5d_matmul.argtypes = [
            ctypes.c_void_p,    # Input tensor A
            ctypes.c_void_p,    # Input tensor B
            ctypes.c_void_p,    # Output tensor C
            ctypes.c_int,       # batch_size
            ctypes.c_int,       # channels
            ctypes.c_int,       # depth
            ctypes.c_int,       # height
            ctypes.c_int,       # width
            ctypes.c_int        # new_width  <-- Added this argument
        ]
        self.lib.py_tensor_5d_matmul.restype = None  # Explicitly set return type

        print("Function signatures configured")



    def matrix_dot(self, a, power=1):
        """Compute matrix power A^n."""
        if not isinstance(a, (np.ndarray, cp.ndarray)):
            raise TypeError("Input must be numpy or cupy array")

        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("Input must be square matrix")

        # Convert to GPU array in Fortran order
        a_gpu = cp.asfortranarray(cp.asarray(a, dtype=cp.float64))
        c_gpu = cp.zeros_like(a_gpu, order='F')

        # Call Fortran function
        self.lib.py_matrix_dot(
            ctypes.c_void_p(a_gpu.data.ptr),
            ctypes.c_void_p(c_gpu.data.ptr),
            ctypes.c_int(a_gpu.shape[0]),
            ctypes.c_int(power)
        )

        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(c_gpu) if isinstance(a, np.ndarray) else c_gpu

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

    def batched_vector_matmul(self, v, a):
        """Compute batched vector-matrix multiplication matching Fortran dimensions.
        Args:
            v: Batch of vectors (input_size x batch_size)
            a: Matrix (input_size x hidden_size)
        """
        input_size, batch_size = v.shape
        _, hidden_size = a.shape

        #print(f"DEBUG batched dimensions:")
        #print(f"v: {v.shape} - (input_size x batch_size)")
        #print(f"a: {a.shape} - (input_size x hidden_size)")

        # Need to pad matrix 'a' to be square (n x n) as expected by Fortran
        n = max(input_size, hidden_size)
        a_padded = cp.zeros((n, n), dtype=cp.float64, order='F')
        a_padded[:input_size, :hidden_size] = a

        # Convert vectors to Fortran order
        v_f = cp.asfortranarray(v, dtype=cp.float64)

        # Create output array matching Fortran expectations
        c_f = cp.empty((n, batch_size), dtype=cp.float64, order='F')

        #print(f"Padded dimensions:")
        #print(f"a_padded: {a_padded.shape} - (n x n)")
        #print(f"v_f: {v_f.shape} - (n x batch_size)")
        #print(f"c_f: {c_f.shape} - (n x batch_size)")

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

    def strided_batch_matmul(self, m, k, n, batch_size, a, b):
        """Compute strided batch matrix multiplication."""
        # Convert dimensions to integers
        m = int(m)
        k = int(k)
        n = int(n)
        batch_size = int(batch_size)

        # Calculate strides as integers
        stride_a = int(m * k)
        stride_b = int(k * n)
        stride_c = int(m * n)

        # Create output array with correct shape
        c_gpu = cp.empty((batch_size * m, n), dtype=cp.float64)

        # Call Fortran function with correct type conversions
        self.lib.py_strided_batch_multiply(
            ctypes.c_void_p(a.data.ptr),
            ctypes.c_void_p(b.data.ptr),
            ctypes.c_void_p(c_gpu.data.ptr),
            ctypes.c_int(m),
            ctypes.c_int(k),
            ctypes.c_int(n),
            ctypes.c_int(batch_size),
            ctypes.c_int(stride_a),   # Changed from c_long to c_int
            ctypes.c_int(stride_b),   # Changed from c_long to c_int
            ctypes.c_int(stride_c)    # Changed from c_long to c_int
        )

        return c_gpu.reshape(batch_size, m, n)

    # fortran column major, 1-base
    def tensor_4d_matmul(self, a, b):
        """
        Compute 4D tensor multiplication using cuBLAS.

        Handles both input formats:
        - Format 1: (batch1, batch2, m, n) - PyTorch/TF style
        - Format 2: (n, n, batch, 1) - Fortran style
        """
        import numpy as np
        import cupy as cp

        # Convert to CuPy if needed
        a_gpu = cp.asarray(a) if isinstance(a, np.ndarray) else a
        b_gpu = cp.asarray(b) if isinstance(b, np.ndarray) else b

        # Check if input is already in Fortran format (n,n,batch,1)
        if a_gpu.shape[-1] == 1:  # Already in Fortran format
            n = a_gpu.shape[0]
            batch_size = a_gpu.shape[2]
            c_gpu = cp.empty_like(a_gpu)
            input_was_fortran = True
        else:  # Need to convert from (batch1,batch2,m,n)
            batch1, batch2, m, n = a_gpu.shape
            total_batch = batch1 * batch2
            a_gpu = cp.asfortranarray(a_gpu.transpose(2,3,0,1).reshape(m, n, total_batch, 1))
            b_gpu = cp.asfortranarray(b_gpu.transpose(2,3,0,1).reshape(n, n, total_batch, 1))
            c_gpu = cp.empty_like(a_gpu)
            batch_size = total_batch
            input_was_fortran = False

        # Call Fortran
        self.lib.py_tensor_4d_matmul(
            ctypes.c_void_p(a_gpu.data.ptr),
            ctypes.c_void_p(b_gpu.data.ptr),
            ctypes.c_void_p(c_gpu.data.ptr),
            ctypes.c_int(batch_size),
            ctypes.c_int(n)
        )

        # Convert back if needed
        if not input_was_fortran:
            result = c_gpu.reshape(m, n, batch1, batch2).transpose(2,3,0,1)
        else:
            result = c_gpu

        return cp.asnumpy(result) if isinstance(a, np.ndarray) else result

    def tensor_5d_matmul(self, a, b):
        """
        Compute 5D tensor multiplication using the Fortran tensor_5d_matmul_py_b.
        """
        # Convert to GPU arrays and ensure Fortran order.
        a_gpu = cp.asarray(a, dtype=cp.float64, order='F')
        b_gpu = cp.asarray(b, dtype=cp.float64, order='F')

        # Check dimensions
        if a_gpu.ndim != 5 or b_gpu.ndim != 5:
            raise ValueError("Input tensors must be 5D")

        # Get dimensions from Fortran-ordered arrays *correctly*
        batch, channels, depth, height, width = a_gpu.shape
        batch_b, channels_b, depth_b, width_b, new_width = b_gpu.shape

        # Check dimension compatibility
        if (batch, channels, depth, width) != (batch_b, channels_b, depth_b, width_b):
            raise ValueError("Incompatible shapes for tensor multiplication")

        # Convert to correct layout
        a_gpu = cp.asfortranarray(a_gpu.transpose(3,4,2,1,0))  # (h,w,d,c,b)
        b_gpu = cp.asfortranarray(b_gpu.transpose(3,4,2,1,0))  # (w,new_w,d,c,b)

        # Create output array in Fortran order.
        c_gpu = cp.zeros((batch, channels, depth, height, new_width),
                            dtype=cp.float64, order='F')


        # Call the Fortran subroutine DIRECTLY.  This is the key change.
        self.lib.py_tensor_5d_matmul(
            ctypes.c_void_p(a_gpu.data.ptr),
            ctypes.c_void_p(b_gpu.data.ptr),
            ctypes.c_void_p(c_gpu.data.ptr),
            ctypes.c_int32(batch),
            ctypes.c_int32(channels),
            ctypes.c_int32(depth),
            ctypes.c_int32(height),
            ctypes.c_int32(width),
            ctypes.c_int32(new_width)
        )

        # Ensure computation is complete
        cp.cuda.runtime.deviceSynchronize()

        # Convert result back to original layout
        result = cp.ascontiguousarray(c_gpu.transpose(4,3,2,0,1))

        # Return based on input type, converting to C order on return.
        return cp.asnumpy(c_gpu, order='C') if isinstance(a, np.ndarray) else c_gpu.copy()

    # python wrapper is running py_matmul, not tensor_5d_matmul; the answers are correct to e-5
    def parallel_tensor_5d_matmul(self, a, b, max_workers=4):
        """
        Parallel implementation of tensor_5d_matmul using task parallelism.

        Args:
            a: 5D tensor (batch, channels, depth, height, width)
            b: 5D tensor (batch, channels, depth, width, new_width)
            max_workers: Maximum number of concurrent workers

        Returns:
            5D tensor result of matrix multiplication
        """
        # Convert inputs to cupy arrays if needed
        a_gpu = cp.asarray(a) if isinstance(a, np.ndarray) else a
        b_gpu = cp.asarray(b) if isinstance(b, np.ndarray) else b

        # Get dimensions
        batch_size, channels, depth, height, width = a_gpu.shape
        new_width = b_gpu.shape[4]

        # Create output tensor
        c_gpu = cp.zeros((batch_size, channels, depth, height, new_width), dtype=a_gpu.dtype)

        # Create a list of all slice coordinates
        slice_coords = []
        for batch_idx in range(batch_size):
            for chan_idx in range(channels):
                for depth_idx in range(depth):
                    slice_coords.append((batch_idx, chan_idx, depth_idx))

        # Function to process a single slice
        def process_slice(coords):
            batch_idx, chan_idx, depth_idx = coords

            # Extract slices
            a_slice = a_gpu[batch_idx, chan_idx, depth_idx]
            b_slice = b_gpu[batch_idx, chan_idx, depth_idx]

            # Call the high-performance tensor_matrix_multiply
            c_slice = self.matmul(a_slice, b_slice)

            # Store result directly in output tensor
            c_gpu[batch_idx, chan_idx, depth_idx] = c_slice

            return f"Processed slice {coords}"

        # Use thread pool to process slices in parallel
        import time
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all slice processing tasks
            futures = [executor.submit(process_slice, coords) for coords in slice_coords]

            # Wait for all tasks to complete
            concurrent.futures.wait(futures)

        end_time = time.time()
        elapsed = end_time - start_time

        # Calculate GFLOPS
        total_flops = 2.0 * batch_size * channels * depth * height * width * new_width
        gflops = total_flops / (elapsed * 1e9)
        print(f"Parallel tensor_5d_matmul completed in {elapsed:.4f}s ({gflops:.2f} GFLOPS)")

        return cp.asnumpy(c_gpu) if isinstance(a, np.ndarray) else c_gpu

    def compare_implementations(self, a, b, debug=True):
        """
        Compare original and optimized implementations with debug output.
        """
        import time

        try:
            if debug:
                print("\nStarting comparison test...")
                print(f"Input shapes: A={a.shape}, B={b.shape}")
                print("Memory before operations:", cp.cuda.runtime.memGetInfo())

            # Run original implementation
            if debug:
                print("\nRunning original implementation...")
            start = time.perf_counter()
            result_orig = self.tensor_5d_matmul(a, b)
            time_orig = time.perf_counter() - start

            if debug:
                print("Original implementation complete")
                print("Memory after original:", cp.cuda.runtime.memGetInfo())

            # Run optimized implementation
            if debug:
                print("\nRunning optimized implementation...")
            start = time.perf_counter()
            result_opt = self.tensor_5d_matmul_optimized(a, b)
            time_opt = time.perf_counter() - start

            if debug:
                print("Optimized implementation complete")
                print("Memory after optimized:", cp.cuda.runtime.memGetInfo())

            # Calculate metrics
            batch, channels, depth, height, width = a.shape
            flops = 2.0 * batch * channels * depth * height * width * width
            gflops_orig = flops / (time_orig * 1e9)
            gflops_opt = flops / (time_opt * 1e9)

            # Compare results
            if isinstance(a, np.ndarray):
                max_diff = np.max(np.abs(result_orig - result_opt))
            else:
                max_diff = cp.max(cp.abs(result_orig - result_opt)).get()

            results = {
                'original_time': time_orig,
                'optimized_time': time_opt,
                'speedup': time_orig / time_opt,
                'original_gflops': gflops_orig,
                'optimized_gflops': gflops_opt,
                'max_difference': max_diff,
                'shapes': {
                    'input': a.shape,
                    'weights': b.shape,
                    'output': result_opt.shape
                }
            }

            if debug:
                print("\nResults:")
                for k, v in results.items():
                    print(f"{k}: {v}")

            return results

        except Exception as e:
            print(f"Error in comparison: {str(e)}")
            print("GPU Memory Info:")
            print(cp.cuda.runtime.memGetInfo())
            raise



    def _ensure_fortran_layout(self, arr):
        """Ensure array is in Fortran-order memory layout."""
        if not isinstance(arr, (np.ndarray, cp.ndarray)):
            raise TypeError("Input must be numpy or cupy array")
        if not arr.flags.f_contiguous:
            return cp.asfortranarray(arr)
        return arr


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # Cleanup handled by CUDA Fortran

    def __del__(self):
        pass # cleanup handled by fortran

# added optimised vector matrix operations, removed image convolution (to separate wrapper and kernel)
