An attempt to program a work efficient radix sort (which should work well when parallelised in CUDA) using Claude and Llama 3.1-405B. See https://felixquinihildebet.wordpress.com/2024/08/01/my-terrible-radix-sort-attempt-sends-claude-and-llama-3-1-into-a-doom-loop/ for discussion of attempt #1.

https://github.com/frasertajima/fortran/blob/main/radix_sort/radix_sort.f90 and notebook https://github.com/frasertajima/fortran/blob/main/radix_sort/radix_sort_python.ipynb show good performance for simple CPU based radix sort. Original python sort code was from locally run nemotron llm (which is impressive). A more complex CPU based radix sort closer to the GPU CUDA radix sort in development is slightly faster, at 1.60s per 20M elements vs 1.67s, but this python derived sort is much simpler and thus more suitable for use in a Jupyter notebook while retaining essentially the same performance. I tested it with 1 trillion elements (consuming 40GB plus in RAM). 2T should be possible but the python version would likely require over 160GB of ram.

cpu_radix_sort2.cuf is the current performance leader in my radix sort experiments (outperforming numpy); despite the .cuf extension it is not using CUDA but the GPU module can be added when its performance is satisfactory.

testSort3.cuf using thrust now superceeds custom kernel at 0.0066s for 2M element sort (see thrust folder)
