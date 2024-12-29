I came across a device unavailable error when using thrust kernels and then using my custom CUDA kernels in Fortran. 
One way to solve it was to run the thrust kernel first. This is a more systematic solution.
Here are the key steps we took to resolve the device unavailable error and ensure that the Thrust kernels work seamlessly with custom CUDA kernels:

### Key Steps to Resolve Device Unavailable Error

1. **Ensure Proper Device Context Management**:
   - **Set Device**: Before running any CUDA kernel or Thrust operation, ensure the correct CUDA device is set using `cudaSetDevice`.
   - **Reset Device**: ==Use `cudaDeviceReset()` to reset the device if any errors occur, ensuring a clean state.==

2. **Correct Memory Allocation and Initialization**:
   - **Allocate Device Memory**: Properly allocate device memory for matrices and other necessary data.
   - **Initialize Data**: Initialize matrices with appropriate values before operations.

3. **Use Temporary Variables**:
   - **Avoid Multiple References**: Use temporary variables to avoid multiple references to device-resident objects in assignment statements.

4. **Explicitly Handle Errors**:
   - **Check for Errors**: After each CUDA operation or kernel launch, check for errors using `cudaGetLastError()` and handle them appropriately.
   - **Error Handling in Thrust**: Catch and handle `thrust::system_error` exceptions to identify and manage Thrust-specific errors.

5. **Using Thrust Copy for Data Transfer**:
   - **Use `thrust::copy`**: Instead of `cudaMemcpy`, use `thrust::copy` to transfer data between device and host. Ensure that pointers and sizes passed are correct.
   - This step was because Claude could never get cudamemcpy right. Using the native thrust function should be faster and more reliable.
