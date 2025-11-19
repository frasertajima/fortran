CIFAR-10 using cuDNN instrinics matching PyTorch in accuracy and execution speed (prior to fortran optimisations such as found in MNIST and nbody project).
[cifar10_cudnn26.cuf](https://github.com/frasertajima/fortran/blob/main/CIFAR-10%20cuDNN/cifar10_cudnn26.cuf) is the baseline cuDNN version that achieves parity with PyTorch (79.3% accuracy and 3.5s/epoch) on an unaugmented dataset. As the baseline version it has not been optimised for performance.


![IMG_2618](https://github.com/user-attachments/assets/d2fc37e6-fb55-415d-9949-15d572707220)
cuDNN functions that were attempted

Update: 
Use the 2 line version 28 workflow to go from zero to training. Now optimised for 4x PyTorch speed and modular design. ![`v28baseline_plus_export`](https://github.com/frasertajima/fortran/tree/main/CIFAR-10%20cuDNN/v28_baseline_plus_export) enables further inference and testing using Jupyter notebooks and python.

<img width="1325" height="973" alt="Screenshot From 2025-11-19 13-19-33" src="https://github.com/user-attachments/assets/01b9a115-a3e7-4441-b0a3-b0b30de1b318" />
