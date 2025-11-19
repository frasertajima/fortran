"""
Key insight: cuDNN with CUDNN_TENSOR_NCHW expects data in a specific memory layout.

When Fortran declares: real(4) :: weights(32, 3, 3, 3)
And tells cuDNN: NCHW format with dims (32, 3, 3, 3)

cuDNN interprets the memory as:
- N (filters) = 32
- C (channels) = 3  
- H (height) = 3
- W (width) = 3

But Fortran stores in column-major (F-order), so the FIRST index varies fastest in memory.
This means in memory, we have: weights(1,1,1,1), weights(2,1,1,1), weights(3,1,1,1), ..., weights(32,1,1,1), weights(1,2,1,1), ...

For cuDNN NCHW, it expects (in C-order memory): filter0_ch0_row0_col0, filter0_ch0_row0_col1, ...

The key is: does cuDNN know about F-order, or does it always assume C-order?
Answer: cuDNN always assumes C-order memory layout!

So when we export from Fortran and load in Python, we need to account for this.
"""

import numpy as np

# Load weights
conv1_w_data = np.fromfile("datasets/cifar10/saved_models/cifar10/conv1_weights.bin", dtype=np.float32)

print("="*70)
print("Understanding the memory layout")
print("="*70)

print("\nFortran allocates: weights(32, 3, 3, 3)")
print("Fortran writes to file in F-order (column-major)")
print("This means memory order is: [filter_idx varies fastest]")
print()

# When Fortran writes (32,3,3,3) in F-order to a file:
# Memory: w[0,0,0,0], w[1,0,0,0], w[2,0,0,0], ..., w[31,0,0,0], w[0,1,0,0], ...
# where indices are (filter, height, width, channel) in Fortran's indexing

# cuDNN expects NCHW in C-order memory:
# Memory: w[0,0,0,0], w[0,0,0,1], w[0,0,0,2], ..., w[0,0,0,31], w[0,0,1,0], ...
# where indices are (filter, channel, height, width) in PyTorch's indexing

print("cuDNN NCHW descriptor tells cuDNN:")
print("  dims = (32, 3, 3, 3) means (N=32 filters, C=3 channels, H=3, W=3)")
print()

print("But cuDNN ALWAYS reads memory in C-order!")
print("So it reads the Fortran F-order data incorrectly.")
print()

print("The question is: What shape did Fortran actually write?")
print(f"File has {conv1_w_data.size} elements = {32*3*3*3}")
print()

# Let's think about this differently:
# If Fortran has weights(32,3,3,3) and cuDNN works correctly in Fortran,
# then cuDNN must be interpreting the F-order memory correctly.
# This means cuDNN's NCHW descriptor with Fortran arrays expects:
#   - First Fortran dimension (32) = N (filters)
#   - Second Fortran dimension (3) = C (channels)  
#   - Third Fortran dimension (3) = H (height)
#   - Fourth Fortran dimension (3) = W (width)

# When written to file in F-order and read in Python:
# We need to reshape with order='F' to get the same logical array
weights_f_order = conv1_w_data.reshape((32, 3, 3, 3), order='F')

# But PyTorch expects (filters, channels, height, width) in C-order
# The Fortran array IS (filters, channels, height, width) but in F-order memory
# So we just need to convert F-order to C-order:
weights_c_order = np.ascontiguousarray(weights_f_order)

print("HYPOTHESIS: Just reshape with order='F' and make C-contiguous")
print(f"Shape: {weights_c_order.shape}")
print(f"First filter[0], channel[0], 3x3:")
print(weights_c_order[0, 0, :, :])
