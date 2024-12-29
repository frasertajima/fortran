I came across a `device unavailable` error when using thrust kernels and then using my custom CUDA kernels in Fortran. 
One way to solve it was to run the thrust kernel first. This is a more flexible solution.
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

Note on using thrust with Fedora Silverblue 41 (or gcc 14 distros):
Fedora Silverblue 41 uses the newest gcc 14 C++ compiler, which thrust refuses to use (it wants gcc 13). I am assuming you already have the nvidia hpc toolkit with thrust installed already. The easiest way I found to enable thrust to create a Fedora 39 distrobox with nvidia support:

1. create fedora39 distrobox:
`distrobox create --nvidia --image quay.io/fedora/fedora:39 --name fedora39`
2. enter into fedora 39 distrobox and install gcc 13 (not 14 which is installed in Fedora 41): 
`sudo dnf groupinstall "Development Tools"`
`sudo dnf install gcc-c++`
3. create symbolic link instead fedora39 distrobox to redirect any gcc 14 calls to gcc 13:
`cd /usr/lib/gcc/x86_64-redhat-linux`
`sudo ln -s 13 14`
