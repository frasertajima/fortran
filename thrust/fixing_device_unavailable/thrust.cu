#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <stdio.h>

struct matrix_multiply_functor {
    const int M, N, K;
    const double* A;
    const double* B;

    matrix_multiply_functor(int m, int n, int k, const double* a, const double* b)
        : M(m), N(n), K(k), A(a), B(b) {}

    __host__ __device__
    double operator()(const int idx) const {
        const int i = idx / N;
        const int j = idx % N;

        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[i * K + k] * B[k * N + j];
        }
        return sum;
    }
};

extern "C" {
    void thrust_matrix_multiply(double* A, double* B, double* C, int M, int N, int K) {
        // Simple initialization of CUDA
        cudaFree(0);

        try {
            // Use thrust's transform with raw iterators
            thrust::transform(
                thrust::device,
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(M * N),
                thrust::device_pointer_cast(C),
                matrix_multiply_functor(M, N, K, A, B)
            );

            // Make sure the operation is complete
            cudaDeviceSynchronize();
        }
        catch(thrust::system_error &e) {
            fprintf(stderr, "Thrust error in matrix multiply: %s\n", e.what());
            cudaError_t error = cudaGetLastError();
            fprintf(stderr, "CUDA error state: %s\n", cudaGetErrorString(error));
        }
    }

    void thrust_copy_device_to_host(double* device_data, double* host_data, int size) {
        try {
            thrust::copy(thrust::device_pointer_cast(device_data), thrust::device_pointer_cast(device_data) + size, host_data);
        }
        catch(thrust::system_error &e) {
            fprintf(stderr, "Thrust error in copy: %s\n", e.what());
        }
    }

    void thrust_test() {
        cudaFree(0);  // Initialize CUDA context
        try {
            thrust::device_vector<int> d_vec(10);
            thrust::sequence(d_vec.begin(), d_vec.end());
            printf("Thrust test successful. Device vector: ");
            for (int i = 0; i < d_vec.size(); ++i) {
                printf("%d ", d_vec[i]);
            }
            printf("\n");
        }
        catch(thrust::system_error &e) {
            fprintf(stderr, "Thrust test failed: %s\n", e.what());
        }
    }
}

// compile with: nvcc -c thrust.cu -o thrust.C.o
