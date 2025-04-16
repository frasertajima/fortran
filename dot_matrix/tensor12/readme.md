Completely refactored vector matrix functions, replacing tensor core operations with cuBLAS kernels as vector matrix does not fit tensor core strengths at all. Result is consistently better performance and much higher accuracy. The performance uplift is much more modest than for tensor core (only 5x vs 50x) but given the workflow, tensor core does not offer consistent performance advantages (and the cuBLAS kernel offers much higher performance for similar speed).

Also updated tensor core engine deploying cross terms for slightly greater accuracy in other matrix operations (slight improvement only).

Compiled for compute 86 (A1000 and above).

### Vector Matrix:
Note that the missing accuracy points represent 100% accuracy:
![Screenshot From 2025-03-28 10-49-31](https://github.com/user-attachments/assets/27e1d1f6-6f2d-4700-9124-5873e807948a)

Smaller matrices trade accuracy for speed:
![Screenshot From 2025-03-28 10-55-16](https://github.com/user-attachments/assets/03708893-5adb-4af5-9d2d-46a528c6a336)

### Matrix Vector:
![Screenshot From 2025-03-28 10-49-41](https://github.com/user-attachments/assets/626ba151-7884-4766-a279-8afde62a11d4)

Speed and accuracy:
![Screenshot From 2025-03-28 10-58-06](https://github.com/user-attachments/assets/86aa258e-2132-4011-9819-f4da0a45b0c9)

### Batched Vector:
Hybrid approach:
![Screenshot From 2025-03-28 10-49-53](https://github.com/user-attachments/assets/4f62800e-36f5-4eef-8716-0ffc63c9917c)

Speed for accuracy tradeoff:
![image](https://github.com/user-attachments/assets/f378703c-bccf-4188-bf9e-4102ff02cc57)

The end result is a Fortran CUDA kernel that outperforms CuPy on all workflows with 50x-5x in ideal conditions depending on the nature of the matrix multiplication.

NOTE: As of April 15, 2025, with the release of Fedora Silverblue 42 (using gcc 15 instead of gcc 14), it may be necessary to run the CUDA Fortran examples within a distrobox that has gcc 14.2 or lower because gcc 15 enforces tigther security that prevents the nvidia hpc library from loading. Specifically it cannot load a library file (ignore the pygame message): ![Screenshot From 2025-04-15 20-08-31](https://github.com/user-attachments/assets/7ccc0c08-ffb4-497e-9ad6-ff5c3994ed0f). Running VSCode within a Fedora 41 distrobox retains the working state for now.

