# fortran
examples while I learn Fortran:

1. random_number_generator7.f90: https://github.com/frasertajima/random_number_yubikey/blob/main/random_number_generator7.f90
2. random_password_generator4.f90: https://github.com/frasertajima/random_number_yubikey/blob/main/random_password_generator4.f90
3. montecarlo.cuf: see https://felixquinihildebet.wordpress.com/2024/07/01/cuda-fortran-example-calculating-european-options-using-the-monte-carlo-method/
4. random_number_generator_0to1.f90: basis for replacing curand_m.cuf module as it uses YubiKey to generate random numbers from 0 to 1.0
5. ran1 (folder): medfit example: Both Fortran 77 and refactored modern Fortran from *Numerical Recipes in Fortran* chapter 15. Added csv handling and fixed a ran1.f error in Fortran 77 that occured using the nvfortran compiler; refactoring ran1.f fixed the error (https://felixquinihildebet.wordpress.com/2024/07/09/misadventures-in-fortran-77/).
6. mandelbrot_color3.cuf: generated using Claude Sonnet 3.5 (one of the few LLMs to successfully code a Mandelbrot plot using CUDA Fortran)--not perfect, but impressive (outdated now)
7. mandelbrot_llama31.cuf: generated using LlaMA3.1-405B. The *fastest* Mandelbrot plot by a wide margin: 0.34ms vs 8.81ms for mandelbrot_color3.cuf or 184ms for Nvidia's CUDA code sample in Numba and Python. (https://felixquinihildebet.wordpress.com/2024/07/24/llama-3-1-and-the-mandelbrot-plots-in-cuda-fortran-revisited/)
8. gauss-jordan (folder): https://felixquinihildebet.wordpress.com/2024/07/19/solving-linear-equations-with-gauss-jordan-elimination-in-fortran/
9. replace_hard_line_breaks.f90: replaces hard line breaks created by llama3.1 when proofreading text with spaces for ease of editing. Asks for input markdown file name and output markdown file name.
10. replace_clipboard_linebreaks.f90: replaces the linux system clipboard text instead of a markdown file; this should make it easier: you copy the proofread output from llama3.1, run `./replace_clipboard_linebreaks` and then paste the clipboard into Obsidian or another editor (without needing to save the proofread text into a file and copy the processed text back).
11. cuda_device_query: small CUDA utility to display your GPU CUDA capabilities (useful for fitting programs to your GPU's abilities).
12. radix_sort: a massive 17 week project to get it running and try to optimise it; upended by thrust version (under thrust folder) which completely outperformed it (still working on optimising custom kernel)
13. laplace3D Oxford: C++ kernel was unbeatable in CUDA Fortran but host-device and device-host transfers were improved
14. matrix multiplication: custom kernel went from 300Gflops to 3,000Gflops after interesting optimisations in CUDA; cuBLAS blew this out of the water with 12,000Gflops
15. matrix_dot using tensor core: https://github.com/frasertajima/fortran/tree/main/dot_matrix/tensor12 newest kernel now hits 35.5TFLOPS vs 300GFLOPS for CuPy for matrix_dot, 18TFLOPS vs 200GFLOPS for matmul, 12TFLOPS vs 193 for batched matrix, and 1.5TFLOPS vs 139GFLOPS for tennsor 4d for example. Other applications using the tensor core engine are also included in the directories under tennsor11, etc.
![Screenshot From 2025-04-29 08-25-39](https://github.com/user-attachments/assets/38cf6350-0d86-4e5c-9985-91cb8879bb22)
