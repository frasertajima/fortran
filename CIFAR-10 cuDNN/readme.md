CIFAR-10 using cuDNN instrinics matching PyTorch in accuracy and execution speed (prior to fortran optimisations such as found in MNIST and nbody project).
[cifar10_cudnn26.cuf](https://github.com/frasertajima/fortran/blob/main/CIFAR-10%20cuDNN/cifar10_cudnn26.cuf) is the baseline cuDNN version that achieves parity with PyTorch (79.3% accuracy and 3.5s/epoch) on an unaugmented dataset. As the baseline version it has not been optimised for performance.


![IMG_2618](https://github.com/user-attachments/assets/d2fc37e6-fb55-415d-9949-15d572707220)
cuDNN functions that were attempted

Update: 
Use the 2 line version 28 workflow to go from zero to training. Now optimised for 4x PyTorch speed and modular design. 

![`v28baseline_plus_export`](https://github.com/frasertajima/fortran/tree/main/CIFAR-10%20cuDNN/v28_baseline_plus_export) enables further inference and testing using Jupyter notebooks and python.

*Version 28b (in testing) provides managed memory (enabling for example 34GB datasets to be processed on an 8GB GPU without crashing--unlike CuPy or PyTorch). Version 28c has cuDNN warp shuffle, used in the GPU diagnostics for example, which cuts 2s off the total training time in CIFAR-10. Version 28d is refining managed memory to enable async batch loading of the dataset to handle exteremely large datasets limited only by the SSD, not main RAM. With the ability to handle a 4TB dataset, after v28d is tested and working, scientific computing projects should be accessible: ![any ideas for scientific computing projects on large datasets?](https://felixquinihildebet.wordpress.com/2025/11/20/can-someone-suggest-a-scientific-project-involving-a-large-dataset-that-is-difficult-for-pytorch/)*

![v28d_streaming](https://github.com/frasertajima/fortran/tree/main/CIFAR-10%20cuDNN/v28d_streaming) can now handle datasets of unlimited size (just add SSD storage). Obviously slower than the full RAM version, which should be used for normal datasets that can fit in your system RAM (not just GPU RAM). Aimed for scientific computing datasets that are larger than system RAM (such as 500GB or above). Even then, CIFAR-10 streaming at 52.5s training is still faster than PyTorch (but not as fast as the normal 29.8s run).

<img width="1325" height="973" alt="Screenshot From 2025-11-19 13-19-33" src="https://github.com/user-attachments/assets/01b9a115-a3e7-4441-b0a3-b0b30de1b318" />
