// thrust.cu
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

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

static bool matrix_info_printed = false;

extern "C" {
    void thrust_print_matrix_info(int M, int N, int K) {
        if (!matrix_info_printed) {
            printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
            matrix_info_printed = true;
        }
    }

    void thrust_test() {
        try {
            thrust::device_vector<double> d_test(10, 1.0);
            printf("Thrust test successful\n");
        }
        catch(thrust::system_error &e) {
            fprintf(stderr, "Thrust test failed: %s\n", e.what());
        }
    }

    void thrust_matrix_multiply(double* A, double* B, double* C, int M, int N, int K) {
        try {
            // Create device pointers
            thrust::device_ptr<const double> dev_A(A);
            thrust::device_ptr<const double> dev_B(B);
            thrust::device_ptr<double> dev_C(C);

            // Create temporary device vector for output
            thrust::device_vector<double> d_output(M * N);

            // Create counting iterator
            thrust::counting_iterator<int> begin(0);
            thrust::counting_iterator<int> end = begin + (M * N);

            // Perform multiplication into temporary vector
            thrust::transform(
                thrust::device,
                begin, end,
                d_output.begin(),
                matrix_multiply_functor(M, N, K,
                    thrust::raw_pointer_cast(dev_A),
                    thrust::raw_pointer_cast(dev_B))
            );

            // Copy result back to C
            thrust::copy(d_output.begin(), d_output.end(), dev_C);

            cudaDeviceSynchronize();
        }
        catch(thrust::system_error &e) {
            fprintf(stderr, "Thrust matrix multiply failed: %s\n", e.what());
            cudaError_t error = cudaGetLastError();
            fprintf(stderr, "CUDA error state: %s\n", cudaGetErrorString(error));
        }
    }
}

// nvcc -O3 -arch=sm_86 -c thrust.cu -o thrust.C.o
