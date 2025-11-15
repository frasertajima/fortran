Following Oxford Flowers 102, version 27 breaks the training into two parts: 

1) python program loads CIFAR-10 data, transposes data for Fortran, and saves it in a binary format,
2) trains in cuDNN Fortran as before. The training results are the same but this structure makes it easier for adopting all machine learning workflows as data loading, processing and saving can quickly be adopted for any training workflow.

Only 2 lines to go from zero to training:

`uv run prepare_cifar10_v27.py`

`bash ./compile_cifar10_v27.sh && ./cifar10_cudnn_v27`
