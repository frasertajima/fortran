import numpy as np
import cupy as cp
import ctypes
import time

class OptimizedImageOps:
    """GPU-accelerated image operations using N-body optimizations."""

    def __init__(self, lib_path='./cuda_image4.so'):
        """Initialize the image operations.

        Args:
            lib_path: Path to the compiled CUDA Fortran library
        """
        try:
            # Initialize CUDA first
            self._init_cuda()

            # Load the library
            self.lib = ctypes.CDLL(lib_path)

            # Set up function signatures
            self._setup_functions()

            # Initialize resources
            self.lib.init_image_ops()
            print("Image operations initialized")

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

    def _setup_functions(self):
        """Configure library function signatures."""
        # Core device functions
        self.lib.init_image_ops.argtypes = []
        self.lib.init_image_ops.restype = None

        self.lib.cleanup_image_ops.argtypes = []
        self.lib.cleanup_image_ops.restype = None

        # Blur operation
        self.lib.apply_blur_shared_memory.argtypes = [
            ctypes.c_void_p,    # Input image
            ctypes.c_void_p,    # Output result
            ctypes.c_int,       # height
            ctypes.c_int,       # width
            ctypes.c_int        # kernel_size
        ]
        self.lib.apply_blur_shared_memory.restype = None

        # Edge detection operation
        self.lib.apply_edge_detection.argtypes = [
            ctypes.c_void_p,    # Input image
            ctypes.c_void_p,    # Output result
            ctypes.c_int,       # height
            ctypes.c_int        # width
        ]
        self.lib.apply_edge_detection.restype = None

        # Emboss operation
        self.lib.apply_emboss.argtypes = [
            ctypes.c_void_p,    # Input image
            ctypes.c_void_p,    # Output result
            ctypes.c_int,       # height
            ctypes.c_int        # width
        ]
        self.lib.apply_emboss.restype = None

        # Sharpen operation
        self.lib.apply_sharpen.argtypes = [
            ctypes.c_void_p,    # Input image
            ctypes.c_void_p,    # Output result
            ctypes.c_int,       # height
            ctypes.c_int        # width
        ]
        self.lib.apply_sharpen.restype = None

        # Custom filter operation
        self.lib.apply_custom_filter.argtypes = [
            ctypes.c_void_p,    # Input image
            ctypes.c_void_p,    # Output result
            ctypes.c_int,       # height
            ctypes.c_int,       # width
            ctypes.c_void_p,    # filter
            ctypes.c_int        # filter_size
        ]
        self.lib.apply_custom_filter.restype = None

        # Tensor-based edge detection operation
        self.lib.apply_edge_detection_tensor.argtypes = [
            ctypes.c_void_p,    # Input image
            ctypes.c_void_p,    # Output result
            ctypes.c_int,       # height
            ctypes.c_int        # width
        ]
        self.lib.apply_edge_detection_tensor.restype = None

        print("Function signatures configured")

    def blur(self, image, kernel_size=3):
        """Apply optimized blur using N-body shared memory approach.

        Args:
            image: 2D numpy or cupy array
            kernel_size: Blur kernel size (odd number: 3, 5, 7, etc.)

        Returns:
            Blurred image
        """
        # Ensure kernel size is odd and valid
        if kernel_size % 2 == 0:
            kernel_size += 1

        if kernel_size < 3:
            kernel_size = 3

        # Convert to device array if needed
        image_d = cp.asarray(image, dtype=cp.float64)

        # Get dimensions
        height, width = image_d.shape

        # Create output array
        result_d = cp.empty_like(image_d)

        # Call the library function
        self.lib.apply_blur_shared_memory(
            ctypes.c_void_p(image_d.data.ptr),
            ctypes.c_void_p(result_d.data.ptr),
            ctypes.c_int(height),
            ctypes.c_int(width),
            ctypes.c_int(kernel_size)
        )

        # Return in same format as input
        return cp.asnumpy(result_d) if isinstance(image, np.ndarray) else result_d

    def edge_detection(self, image):
        """Apply optimized edge detection (Sobel) using N-body shared memory approach.

        Args:
            image: 2D numpy or cupy array

        Returns:
            Edge detection result
        """
        # Convert to device array if needed
        image_d = cp.asarray(image, dtype=cp.float64)

        # Get dimensions
        height, width = image_d.shape

        # Create output array
        result_d = cp.empty_like(image_d)

        # Call the library function
        self.lib.apply_edge_detection(
            ctypes.c_void_p(image_d.data.ptr),
            ctypes.c_void_p(result_d.data.ptr),
            ctypes.c_int(height),
            ctypes.c_int(width)
        )

        # Return in same format as input
        return cp.asnumpy(result_d) if isinstance(image, np.ndarray) else result_d

    def edge_detection_tensor(self, image):
        """Apply tensor-optimized edge detection (Sobel) using im2col and tensor cores.

        Args:
            image: 2D numpy or cupy array

        Returns:
            Edge detection result
        """
        # Convert to device array if needed
        image_d = cp.asarray(image, dtype=cp.float64)

        # Get dimensions
        height, width = image_d.shape

        # Create output array
        result_d = cp.empty_like(image_d)

        # Call the library function
        self.lib.apply_edge_detection_tensor(
            ctypes.c_void_p(image_d.data.ptr),
            ctypes.c_void_p(result_d.data.ptr),
            ctypes.c_int(height),
            ctypes.c_int(width)
        )

        # Return in same format as input
        return cp.asnumpy(result_d) if isinstance(image, np.ndarray) else result_d

    def emboss(self, image):
        """Apply optimized emboss filter using N-body shared memory approach.

        Args:
            image: 2D numpy or cupy array

        Returns:
            Embossed image
        """
        # Convert to device array if needed
        image_d = cp.asarray(image, dtype=cp.float64)

        # Get dimensions
        height, width = image_d.shape

        # Create output array
        result_d = cp.empty_like(image_d)

        # Call the library function
        self.lib.apply_emboss(
            ctypes.c_void_p(image_d.data.ptr),
            ctypes.c_void_p(result_d.data.ptr),
            ctypes.c_int(height),
            ctypes.c_int(width)
        )

        # Return in same format as input
        return cp.asnumpy(result_d) if isinstance(image, np.ndarray) else result_d

    def sharpen(self, image):
        """Apply optimized sharpen filter using N-body shared memory approach.

        Args:
            image: 2D numpy or cupy array

        Returns:
            Sharpened image
        """
        # Convert to device array if needed
        image_d = cp.asarray(image, dtype=cp.float64)

        # Get dimensions
        height, width = image_d.shape

        # Create output array
        result_d = cp.empty_like(image_d)

        # Call the library function
        self.lib.apply_sharpen(
            ctypes.c_void_p(image_d.data.ptr),
            ctypes.c_void_p(result_d.data.ptr),
            ctypes.c_int(height),
            ctypes.c_int(width)
        )

        # Return in same format as input
        return cp.asnumpy(result_d) if isinstance(image, np.ndarray) else result_d

    def custom_filter(self, image, filter_kernel):
        """Apply custom filter using N-body shared memory approach.

        Args:
            image: 2D numpy or cupy array
            filter_kernel: 2D numpy or cupy array representing the filter kernel

        Returns:
            Filtered image
        """
        # Convert to device arrays if needed
        image_d = cp.asarray(image, dtype=cp.float64)
        filter_d = cp.asarray(filter_kernel, dtype=cp.float64)

        # Get dimensions
        height, width = image_d.shape
        filter_size = filter_d.shape[0]  # Assuming square filter

        # Verify filter is square
        if filter_d.shape[0] != filter_d.shape[1]:
            raise ValueError("Filter kernel must be square")

        # Create output array
        result_d = cp.empty_like(image_d)

        # Call the library function
        self.lib.apply_custom_filter(
            ctypes.c_void_p(image_d.data.ptr),
            ctypes.c_void_p(result_d.data.ptr),
            ctypes.c_int(height),
            ctypes.c_int(width),
            ctypes.c_void_p(filter_d.data.ptr),
            ctypes.c_int(filter_size)
        )

        # Return in same format as input
        return cp.asnumpy(result_d) if isinstance(image, np.ndarray) else result_d

    def benchmark_compare(self, image, operations=['blur', 'edge', 'emboss', 'sharpen'], num_trials=3):
         """Compare our optimized operations with CuPy implementations.

         Args:
             image: 2D numpy array
             operations: List of operations to benchmark
             num_trials: Number of trials for each benchmark

         Returns:
             Dictionary with benchmark results
         """
         # Convert to device array
         image_d = cp.asarray(image, dtype=cp.float64)
         height, width = image_d.shape

         results = {}

        # Define CuPy kernels for comparison
        cupy_kernels = {
            'blur': cp.ElementwiseKernel(
                'raw T input, int32 width, int32 height, int32 ksize',
                'T output',
                '''
                int x = i % width;
                int y = i / width;
                int half_ksize = ksize / 2;
                double sum = 0;
                int count = 0;

                for (int ky = -half_ksize; ky <= half_ksize; ky++) {
                    int py = y + ky;
                    if (py < 0) py = 0;
                    if (py >= height) py = height - 1;

                    for (int kx = -half_ksize; kx <= half_ksize; kx++) {
                        int px = x + kx;
                        if (px < 0) px = 0;
                        if (px >= width) px = width - 1;

                        sum += input[py * width + px];
                        count++;
                    }
                }

                output = sum / count;
                ''',
                'blur_kernel'
            ),
            'edge': cp.ElementwiseKernel(
                'raw T input, int32 width, int32 height',
                'T output',
                '''
                int x = i % width;
                int y = i / width;

                if (x == 0 || x == width-1 || y == 0 || y == height-1) {
                    output = 0;
                    return;
                }

                double gx = -input[(y-1)*width + (x-1)] + input[(y-1)*width + (x+1)]
                        -2*input[y*width + (x-1)]     + 2*input[y*width + (x+1)]
                        -input[(y+1)*width + (x-1)]   + input[(y+1)*width + (x+1)];

                double gy = -input[(y-1)*width + (x-1)]  - 2*input[(y-1)*width + x]  - input[(y-1)*width + (x+1)]
                            +input[(y+1)*width + (x-1)]  + 2*input[(y+1)*width + x]  + input[(y+1)*width + (x+1)];

                output = sqrt(gx*gx + gy*gy);
                ''',
                'edge_kernel'
            ),
            'emboss': cp.ElementwiseKernel(
                'raw T input, int32 width, int32 height',
                'T output',
                '''
                int x = i % width;
                int y = i / width;

                if (x == 0 || x == width-1 || y == 0 || y == height-1) {
                    output = input[i];
                    return;
                }

                output = -2*input[(y-1)*width + (x-1)] - input[(y-1)*width + x]
                        -input[y*width + (x-1)]       + input[y*width + x]   + input[y*width + (x+1)]
                                                    + input[(y+1)*width + x] + 2*input[(y+1)*width + (x+1)];

                // Scale to [0,1] range
                output = (output + 1) / 2;
                ''',
                'emboss_kernel'
            ),
            'sharpen': cp.ElementwiseKernel(
                'raw T input, int32 width, int32 height',
                'T output',
                '''
                int x = i % width;
                int y = i / width;

                if (x == 0 || x == width-1 || y == 0 || y == height-1) {
                    output = input[i];
                    return;
                }

                output = -input[(y-1)*width + x]
                        -input[y*width + (x-1)] + 5*input[y*width + x] - input[y*width + (x+1)]
                        -input[(y+1)*width + x];

                // Clamp to [0,1] range
                if (output < 0) output = 0;
                if (output > 1) output = 1;
                ''',
                'sharpen_kernel'
            )
        }

         # Benchmarking loop
         for op in operations:
             results[op] = {}

             # Warm-up
             if op == 'blur':
                 _ = self.blur(image_d, 5)
                 cupy_output = cp.zeros_like(image_d)
                 cupy_kernels[op](image_d, width, height, 5, cupy_output)
             elif op == 'edge':
                 _ = self.edge_detection(image_d)  # N-body shared memory version
                 _ = self.edge_detection_tensor(image_d)  # Tensor core version
                 cupy_output = cp.zeros_like(image_d)
                 cupy_kernels[op](image_d, width, height, cupy_output)
             # ... other operations ...

             # Benchmark our N-body optimized implementation
             nbody_times = []
             for _ in range(num_trials):
                 cp.cuda.Stream.null.synchronize()
                 start = time.time()

                 if op == 'blur':
                     _ = self.blur(image_d, 5)
                 elif op == 'edge':
                     _ = self.edge_detection(image_d)
                 elif op == 'emboss':
                     _ = self.emboss(image_d)
                 elif op == 'sharpen':
                     _ = self.sharpen(image_d)

                 cp.cuda.Stream.null.synchronize()
                 nbody_times.append(time.time() - start)

             # For edge detection, also benchmark tensor core version
             if op == 'edge':
                 tensor_times = []
                 for _ in range(num_trials):
                     cp.cuda.Stream.null.synchronize()
                     start = time.time()
                     _ = self.edge_detection_tensor(image_d)
                     cp.cuda.Stream.null.synchronize()
                     tensor_times.append(time.time() - start)

             # Benchmark CuPy implementation
             cupy_times = []
             for _ in range(num_trials):
                 cupy_output = cp.zeros_like(image_d)
                 cp.cuda.Stream.null.synchronize()
                 start = time.time()

                 if op == 'blur':
                     cupy_kernels[op](image_d, width, height, 5, cupy_output)
                 else:
                     cupy_kernels[op](image_d, width, height, cupy_output)

                 cp.cuda.Stream.null.synchronize()
                 cupy_times.append(time.time() - start)

             # Calculate statistics
             nbody_avg = sum(nbody_times) / num_trials
             cupy_avg = sum(cupy_times) / num_trials
             nbody_mpix = (height * width) / (nbody_avg * 1000000.0)
             cupy_mpix = (height * width) / (cupy_avg * 1000000.0)
             nbody_speedup = cupy_avg / nbody_avg

             results[op] = {
                 'nbody_avg_time': nbody_avg,
                 'cupy_avg_time': cupy_avg,
                 'nbody_mpix_per_second': nbody_mpix,
                 'cupy_mpix_per_second': cupy_mpix,
                 'speedup': nbody_speedup
             }

             # Add tensor core results for edge detection
             if op == 'edge':
                 tensor_avg = sum(tensor_times) / num_trials
                 tensor_mpix = (height * width) / (tensor_avg * 1000000.0)
                 tensor_speedup = cupy_avg / tensor_avg

                 results[op]['tensor_avg_time'] = tensor_avg
                 results[op]['tensor_mpix_per_second'] = tensor_mpix
                 results[op]['tensor_speedup'] = tensor_speedup

         return results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lib.cleanup_image_ops()
        print("Image operations resources cleaned up")

    def __del__(self):
        try:
            self.lib.cleanup_image_ops()
        except:
            pass
