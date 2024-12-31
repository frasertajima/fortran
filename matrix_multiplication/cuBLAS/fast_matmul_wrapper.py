import ctypes
import numpy as np
import cupy as cp

# Load the Fortran shared library
_lib = ctypes.CDLL('./fast_matmul.so')

# Define the function prototype
_lib.tf32_matmul.argtypes = [
    ctypes.c_void_p,  # A
    ctypes.c_void_p,  # B
    ctypes.c_void_p,  # C
    ctypes.c_int,     # M
    ctypes.c_int,     # N
    ctypes.c_int      # K
]

def gpu_matmul(a, b):
    """
    Fast matrix multiplication using cuBLAS TF32.
    """
    # Convert inputs to float32 if needed
    if isinstance(a, np.ndarray):
        a = cp.asarray(a, dtype=np.float32)
    if isinstance(b, np.ndarray):
        b = cp.asarray(b, dtype=np.float32)
    
    # Ensure float32
    a = cp.asarray(a, dtype=np.float32)
    b = cp.asarray(b, dtype=np.float32)
    
    # Get dimensions
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Inner dimensions must match"
    
    # Create output array
    c = cp.empty((M, N), dtype=np.float32)
    
    # Call Fortran function
    _lib.tf32_matmul(
        ctypes.c_void_p(a.data.ptr),
        ctypes.c_void_p(b.data.ptr),
        ctypes.c_void_p(c.data.ptr),
        M, N, K
    )
    
    return c