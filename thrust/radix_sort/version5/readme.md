Revisited thrust sort with managed memory handling for handling of very large arrays, such as sorting 33GB arrays on a 4GB GPU (with 80GB RAM). The RAM size, not the GPU RAM is now the limitation with chunking and managed memory.

Thrust still tests as equal to CuPy in terms of accuracy but a bit slower (except for smaller arrays). Still, CuPy will not be able to handle a 33GB array on a 4GB GPU.

Compile with:

`nvcc -c -o thrust.C.o thrust.cu -fPIC`

`nvfortran cuda_batch_state2.o thrust.cuf testsort5.cuf thrust.C.o -c++libs -cudalib=cublas -o testsort5`

for the fortran kernel to use in python and Jupyter notebooks:

`nvfortran -cuda -shared cuda_batch_state2.o thrust.cuf thrust.C.o -o libthrust_fortran.so -c++libs -ldl -fPIC`

Example run with 800,000,000 elements:

<img width="748" height="1662" alt="Screenshot From 2025-11-01 08-37-44" src="https://github.com/user-attachments/assets/86ab283b-ac47-4b0f-9fc4-e1ebbce021f4" />

For extremely large arrays, say 33GB, you will need about 80GB of RAM. It runs but the fortran random number generator falters (the sorting is fine, just the sample input generation runs out of numbers with very large arrays). cuRAND might fix it but actual data would be more practical rather than bothering with random numbers.

Also, while thrust is very fast, it also has issues with very large arrays (thus the chunking solution). On a related note, trying to sort a large array that completely fills the GPU memory resulted in 66s run times vs 6s run times for the same array that was chunked into 2 pieces. Chunking the array left room for the GPU to work properly without strain. Chunking is fast, merging the pieces together is slow so we want to cut the array into the largest pieces that the GPU can handle without strain. The current chunk size is for a 4GB A1000 GPU. It could be increased for larger GPUs without hurting performance.

---
testsort4 on an RTX4060 with 8GB RAM and 48GB system memory runs while testsort5 will crash as 48GB system memory is insufficient:

<img width="494" height="938" alt="Screenshot From 2025-11-01 21-16-13" src="https://github.com/user-attachments/assets/1a0b0601-13a0-4732-acc7-4aeb908808b2" />

On smaller arrays, the thrust sort can get very good performance:
<img width="538" height="942" alt="image" src="https://github.com/user-attachments/assets/e96adbd3-6c2c-4f70-82a7-b0fc43ca8da8" />
