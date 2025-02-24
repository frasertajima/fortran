![tensor cores at work](https://github.com/frasertajima/fortran/blob/main/dot_matrix/tensor10/pixel_studio_20250123_48025364.jpg)
# CUDA Tensor Core Engine Matrix Multiplication Library Architecture Overview

## Key Architectural Features

1. **Split Precision Strategy**
 - This library implements a sophisticated split precision approach to maintain numerical accuracy while leveraging high-performance tensor cores.

2. **Tensor Core Optimization**
- Utilizes CUDA Tensor Cores for accelerated matrix operations (50x CuPy performance in many operations)
- Configures cuBLAS to use Tensor Core math mode

3. **Stream Management**
- Creates multiple CUDA streams for parallel computation
- Initializes cuBLAS handles with specific streams
- Enables concurrent kernel execution

4. **Flexible Tensor Representation**
- Implements a `tensor_5d` type that supports multi-dimensional tensor operations
- Provides initialization and cleanup methods for tensors
- Supports batched and strided tensor operations

5. Memory efficient
- enables larger tensor_5d matrix operations that CuPy cannot execute due to CuPy's massive memory requirements (1024x1024 tensor_5d runs in RTX 4060's 8GB but requires over 30GB in CuPy)

---

## Matrix Multiplication Subroutine Glossary

### 1. `matrix_dot(a, c, n, iterations)`
- **Purpose**: Compute matrix power A^n
- **Inputs**: 
  - `a`: Input square matrix
  - `n`: Matrix dimension
  - `iterations`: Number of matrix multiplications
- **Output**: Matrix raised to the power of `iterations`
- **Python Example**:
```python
from tensor_matrix_ops import TensorMatrixOps

ops = TensorMatrixOps()
result = ops.matrix_dot(square_matrix, power=3)
```

### 2. `tensor_matrix_multiply(a, b, c, m, k, n, iterations)`
- **Purpose**: General matrix multiplication with split precision
- **Inputs**:
  - `a`: First input matrix (m×k)
  - `b`: Second input matrix (k×n)
  - `m, k, n`: Matrix dimensions
  - `iterations`: Number of multiplications
- **Output**: Resultant matrix
- **Python Example**:
```python
ops = TensorMatrixOps()
result = ops.matmul(matrix_a, matrix_b)
```

### 3. `batched_matmul(a, b, c, m, k, n, batch_size)`
- **Purpose**: Perform matrix multiplication across multiple batches
- **Inputs**:
  - `a`: Batched input matrices (m×k×batch_size)
  - `b`: Batched input matrices (k×n×batch_size)
  - `m, k, n`: Matrix dimensions
  - `batch_size`: Number of matrices in batch
- **Output**: Batched resultant matrices
- **Python Example**:
```python
ops = TensorMatrixOps()
batch_result = ops.batched_matmul(batch_a, batch_b)
```

### 4. `vector_matrix_multiply(v, a, c, n)` & `matrix_vector_multiply(a, v, c, n)`
- **Purpose**: Vector-matrix and matrix-vector multiplication
- **Inputs**:
  - `v`: Input vector
  - `a`: Input matrix
  - `n`: Dimension
- **Output**: Resultant vector
- **Python Example**:
```python
ops = TensorMatrixOps()
vec_result = ops.vector_matmul(vector, matrix)
mat_vec_result = ops.matmul_vector(matrix, vector)
```

### 5. `tensor_4d_matmul(a, b, c, batch_size, n)` & `tensor_5d_matmul(a, b, c, ...)`
- **Purpose**: Multi-dimensional tensor multiplication
- **Supports**: 4D and 5D tensor operations
- **Python Example**:
```python
ops = TensorMatrixOps()
result_4d = ops.tensor_4d_matmul(tensor_a, tensor_b)
result_5d = ops.tensor_5d_matmul(tensor_a, tensor_b)
```

---

## Performance Optimisation details

1. **Tensor Core Utilisation**
- Uses `CUBLAS_GEMM_DEFAULT_TENSOR_OP` for optimal performance
- Supports mixed-precision computations
- trades .9999 precision for 50x faster than CuPy performance (fine for most machine learning applications)

2. **Memory Management**
- Uses device memory efficiently
- Implements custom copy and padding kernels
- Minimises memory transfers

3. **Parallel Processing**
- Utilises multiple CUDA streams
- Implements batched operations with parallel kernels

## Error Handling and Debugging

- Includes `debug_mode` flag for detailed logging
- Uses `debug_print()` for controlled output
- Implements safety checks

## Recommended Usage

1. Initialise the `TensorMatrixOps` class by copying the `tensor_matrix_ops.py` wrapper and `cuda_matlib.so` Fortran kernel in the same directory as the Jupyter notebook calling the tensor core engine.
2. Choose the appropriate matrix multiplication method
3. Ensure input tensors are in correct shape and precision

## Future Improvements

1. Support for more exotic tensor operations; prototype specialised cublasGemm operations for possible speed increase (some require square matrices)
2. Optimisations
3. Use in machine learning Jupyter notebooks (beyond Mnist example)

This implementation provides a high-performance framework for GPU-accelerated matrix operations while maintaining more than sufficient accuracy for most applications.

Claude Sonnet 3.7 seems excited: https://www.threads.net/@mean.absolute.error/post/DGeUA5Qy2jW?xmt=AQGz81zjqJKGKohdM8-aE_9uRuHkkXEOND5dyoslkn_BqA
