Great CUDA course: https://people.maths.ox.ac.uk/gilesm/cuda/index.html has a laplace3d version in C++. Performance was good so the question
was whether Fortran or thrust could improve this custom kernel. Claude helped with a fortran version of the C++ laplace program in exercise 3 of the course.
laplace3d6 is the CUDA Fortran version while laplace3d is the course C++ version in the screenshot.

Observations:
1. the main laplace kernel is highly optimised and given the size of the problem, there was not much improvement possible
2. C++ is faster than Fortran, but not by a massive amount
3. oddly, the host to device and device to host transfer in C++ was very slow compared to Fortran
4. I tried various thrust functions to try to speed up or simplify the code; generally, it was slower
5. I thought thrust copy might be faster than the native Fortran assignment, but I didn't notice much (not worth the bother really)
6. nsys profile shows about 1GB of host to device and device to host transfer going on with these copies taking up most of the time
7. pinned memory really helps with host to device and device to host transfers (about 40% improvement)
