// generic thrust.cu for calling any thrust function in fortran
// nvcc -c thrust.cu -o thrust.C.o
// keep the thrust.C.o instead of thrust.o to avoid conflicts with the fortran object file
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

extern "C" {
  // Sort for int arrays
  void sort_int_wrapper(int *data, int N)
  {
    thrust::device_ptr<int> dev_ptr(data);
    thrust::sort(dev_ptr, dev_ptr + N);
  }

  // Sort for float arrays
  void sort_float_wrapper(float *data, int N)
  {
    thrust::device_ptr<float> dev_ptr(data);
    thrust::sort(dev_ptr, dev_ptr + N);
  }

  // Sort for double arrays
  void sort_double_wrapper(double *data, int N)
  {
    thrust::device_ptr<double> dev_ptr(data);
    thrust::sort(dev_ptr, dev_ptr + N);
  }

  // Reduce for int arrays
  int reduce_int_wrapper(int *data, int N)
  {
    thrust::device_ptr<int> dev_ptr(data);
    return thrust::reduce(dev_ptr, dev_ptr + N);
  }

  // Reduce for float arrays
  float reduce_float_wrapper(float *data, int N)
  {
    thrust::device_ptr<float> dev_ptr(data);
    return thrust::reduce(dev_ptr, dev_ptr + N);
  }

  // Reduce for double arrays
  double reduce_double_wrapper(double *data, int N)
  {
    thrust::device_ptr<double> dev_ptr(data);
    return thrust::reduce(dev_ptr, dev_ptr + N);
  }

  // Transform for int arrays (example: square each element)
  void transform_int_wrapper(int *data, int N)
  {
    thrust::device_ptr<int> dev_ptr(data);
    thrust::transform(dev_ptr, dev_ptr + N, dev_ptr, thrust::placeholders::_1 * thrust::placeholders::_1);
  }

  // Transform for float arrays (example: square each element)
  void transform_float_wrapper(float *data, int N)
  {
    thrust::device_ptr<float> dev_ptr(data);
    thrust::transform(dev_ptr, dev_ptr + N, dev_ptr, thrust::placeholders::_1 * thrust::placeholders::_1);
  }

  // Transform for double arrays (example: square each element)
  void transform_double_wrapper(double *data, int N)
  {
    thrust::device_ptr<double> dev_ptr(data);
    thrust::transform(dev_ptr, dev_ptr + N, dev_ptr, thrust::placeholders::_1 * thrust::placeholders::_1);
  }
}
