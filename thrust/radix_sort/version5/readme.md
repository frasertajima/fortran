Revisited thrust sort with managed memory handling for handling of very large arrays, such as sorting 33GB arrays on a 4GB GPU (with 80GB RAM). The RAM size, not the GPU RAM is now the limitation with chunking and managed memory.

Thrust still tests as equal to CuPy in terms of accuracy but a bit slower (except for smaller arrays). Still, CuPy will not be able to handle a 33GB array on a 4GB GPU.

Compile with:
`nvcc -c -o thrust.C.o thrust.cu -fPIC`
`nvfortran cuda_batch_state2.o thrust.cuf testsort5.cuf thrust.C.o -c++libs -cudalib=cublas -o testsort5`

for the fortran kernel to use in python and Jupyter notebooks:
`nvfortran -cuda -shared cuda_batch_state2.o thrust.cuf thrust.C.o -o libthrust_fortran.so -c++libs -ldl -fPIC`
