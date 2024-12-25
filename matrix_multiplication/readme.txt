Exploration as to why the machine learning matmul functions are so slow compared to CUDA Fortran (200Gflops vs 2,700Gflops). It turns out thrust matrix multiplication is super slow.
To compile thrust libraries and use them in CUDA Fortran:
1. Compile the C++ thrust.cu file first (update for your architecture):     `nvcc -O3 -arch=sm_86 -c thrust.cu -o thrust.C.o`
2. make sure thrust.cuf is in the same directory to enable C++ bindings in Fortran
3. Compile Fortran with the C++ bindings for thrust:                        `nvfortran -o3 thrust.cuf matrix_multiplication_thrust.cuf thrust.C.o -c++libs -o matrix_multiplication_thrust`
4. run './matrix_multiplication_thrust`
