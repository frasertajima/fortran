// Updated thrust.cu with pointer-based functions
// Compile with: nvcc -c thrust.cu -o thrust.C.o --compiler-options -fPIC

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <stdint.h>

extern "C" {
  // Original functions
  
  // Sort for int arrays
  void sort_int_wrapper(int *data, int N) {
    thrust::device_ptr<int> dev_ptr(data);
    thrust::sort(dev_ptr, dev_ptr + N);
  }

  // Sort for float arrays
  void sort_float_wrapper(float *data, int N) {
    thrust::device_ptr<float> dev_ptr(data);
    thrust::sort(dev_ptr, dev_ptr + N);
  }

  // Sort for double arrays
  void sort_double_wrapper(double *data, int N) {
    thrust::device_ptr<double> dev_ptr(data);
    thrust::sort(dev_ptr, dev_ptr + N);
  }

  // Reduce for int arrays
  int reduce_int_wrapper(int *data, int N) {
    thrust::device_ptr<int> dev_ptr(data);
    return thrust::reduce(dev_ptr, dev_ptr + N);
  }

  // Reduce for float arrays
  float reduce_float_wrapper(float *data, int N) {
    thrust::device_ptr<float> dev_ptr(data);
    return thrust::reduce(dev_ptr, dev_ptr + N);
  }

  // Reduce for double arrays
  double reduce_double_wrapper(double *data, int N) {
    thrust::device_ptr<double> dev_ptr(data);
    return thrust::reduce(dev_ptr, dev_ptr + N);
  }

  // Transform for int arrays (square each element)
  void transform_int_wrapper(int *data, int N) {
    thrust::device_ptr<int> dev_ptr(data);
    thrust::transform(dev_ptr, dev_ptr + N, dev_ptr, thrust::placeholders::_1 * thrust::placeholders::_1);
  }

  // Transform for float arrays (square each element)
  void transform_float_wrapper(float *data, int N) {
    thrust::device_ptr<float> dev_ptr(data);
    thrust::transform(dev_ptr, dev_ptr + N, dev_ptr, thrust::placeholders::_1 * thrust::placeholders::_1);
  }

  // Transform for double arrays (square each element)
  void transform_double_wrapper(double *data, int N) {
    thrust::device_ptr<double> dev_ptr(data);
    thrust::transform(dev_ptr, dev_ptr + N, dev_ptr, thrust::placeholders::_1 * thrust::placeholders::_1);
  }
  
  // New pointer-based functions for memory optimization
  
  // Sort float array using raw device pointer
  void sort_float_ptr(void* device_ptr, int N) {
      thrust::device_ptr<float> dev_ptr(static_cast<float*>(device_ptr));
      thrust::sort(dev_ptr, dev_ptr + N);
  }
  
  // Sort double array using raw device pointer
  void sort_double_ptr(void* device_ptr, int N) {
      thrust::device_ptr<double> dev_ptr(static_cast<double*>(device_ptr));
      thrust::sort(dev_ptr, dev_ptr + N);
  }
  
  // Sort int array using raw device pointer
  void sort_int_ptr(void* device_ptr, int N) {
      thrust::device_ptr<int> dev_ptr(static_cast<int*>(device_ptr));
      thrust::sort(dev_ptr, dev_ptr + N);
  }
  
  // Reduce float array using raw device pointer
  float reduce_float_ptr(void* device_ptr, int N) {
      thrust::device_ptr<float> dev_ptr(static_cast<float*>(device_ptr));
      return thrust::reduce(dev_ptr, dev_ptr + N);
  }
  
  // Reduce double array using raw device pointer
  double reduce_double_ptr(void* device_ptr, int N) {
      thrust::device_ptr<double> dev_ptr(static_cast<double*>(device_ptr));
      return thrust::reduce(dev_ptr, dev_ptr + N);
  }
  
  // Reduce int array using raw device pointer
  int reduce_int_ptr(void* device_ptr, int N) {
      thrust::device_ptr<int> dev_ptr(static_cast<int*>(device_ptr));
      return thrust::reduce(dev_ptr, dev_ptr + N);
  }
  
  // Transform float array using raw device pointer (square each element)
  void transform_float_ptr(void* device_ptr, int N) {
      thrust::device_ptr<float> dev_ptr(static_cast<float*>(device_ptr));
      thrust::transform(dev_ptr, dev_ptr + N, dev_ptr, thrust::placeholders::_1 * thrust::placeholders::_1);
  }
  
  // Transform double array using raw device pointer (square each element)
  void transform_double_ptr(void* device_ptr, int N) {
      thrust::device_ptr<double> dev_ptr(static_cast<double*>(device_ptr));
      thrust::transform(dev_ptr, dev_ptr + N, dev_ptr, thrust::placeholders::_1 * thrust::placeholders::_1);
  }
  
  // Transform int array using raw device pointer (square each element)
  void transform_int_ptr(void* device_ptr, int N) {
      thrust::device_ptr<int> dev_ptr(static_cast<int*>(device_ptr));
      thrust::transform(dev_ptr, dev_ptr + N, dev_ptr, thrust::placeholders::_1 * thrust::placeholders::_1);
  }
}