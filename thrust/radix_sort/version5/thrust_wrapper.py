#!/usr/bin/env python3
"""
Python wrapper for the optimized Fortran Thrust kernel.
This wrapper directly interfaces with the device pointers for maximum performance.
"""

import numpy as np
import cupy as cp
import ctypes
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from ctypes import c_int, c_float, c_double, c_void_p, POINTER

class ThrustWrapper:
    """Wrapper for the Fortran Thrust library optimized for GPU operations."""
    
    def __init__(self, lib_path='./libthrust_fortran.so'):
        """
        Initialize the wrapper with the Thrust shared library.
        
        Parameters:
        -----------
        lib_path : str
            Path to the compiled Thrust Fortran shared library
        """
        # Load the shared library
        try:
            self.lib = ctypes.CDLL(lib_path)
            print(f"Successfully loaded Thrust library from {lib_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load library: {e}")
        
        # Set up function signatures for the *_ptr functions that work with raw device pointers
        self._setup_functions()
    
    def _setup_functions(self):
        """Set up function signatures for direct device pointer operations."""
        # Sort functions
        self.lib.sort_float_ptr.argtypes = [c_void_p, c_int]
        self.lib.sort_double_ptr.argtypes = [c_void_p, c_int]
        self.lib.sort_int_ptr.argtypes = [c_void_p, c_int]
        
        # Reduce functions
        self.lib.reduce_float_ptr.argtypes = [c_void_p, c_int]
        self.lib.reduce_float_ptr.restype = c_float
        self.lib.reduce_double_ptr.argtypes = [c_void_p, c_int]
        self.lib.reduce_double_ptr.restype = c_double
        self.lib.reduce_int_ptr.argtypes = [c_void_p, c_int]
        self.lib.reduce_int_ptr.restype = c_int
        
        # Transform functions
        self.lib.transform_float_ptr.argtypes = [c_void_p, c_int]
        self.lib.transform_double_ptr.argtypes = [c_void_p, c_int]
        self.lib.transform_int_ptr.argtypes = [c_void_p, c_int]
    
    def sort(self, array):
        """Sort a CuPy array using Thrust."""
        if not isinstance(array, cp.ndarray):
            raise TypeError("Input must be a CuPy array")
        
        # Create a temporary copy for sorting
        temp = cp.copy(array)
        
        # Get raw device pointer
        dev_ptr = temp.data.ptr
        
        # Call Thrust sort function
        if array.dtype == cp.float32:
            self.lib.sort_float_ptr(c_void_p(dev_ptr), c_int(temp.size))
        elif array.dtype == cp.float64:
            self.lib.sort_double_ptr(c_void_p(dev_ptr), c_int(temp.size))
        elif array.dtype == cp.int32:
            self.lib.sort_int_ptr(c_void_p(dev_ptr), c_int(temp.size))
        else:
            raise TypeError(f"Unsupported dtype: {array.dtype}")
        
        # Ensure operation completes
        cp.cuda.runtime.deviceSynchronize()
        
        # Copy sorted data back to the original array
        array[:] = temp
        
        return array
    
    def reduce(self, array):
        """
        Reduce (sum) a CuPy array using Thrust.
        
        Parameters:
        -----------
        array : cupy.ndarray
            Array to reduce
            
        Returns:
        --------
        float or int
            Sum of all elements in the array
        """
        if not isinstance(array, cp.ndarray):
            raise TypeError("Input must be a CuPy array")
        
        # Ensure array is contiguous
        if not array.flags.c_contiguous:
            array = cp.ascontiguousarray(array)
        
        # Get raw device pointer
        dev_ptr = array.data.ptr
        
        # Call the appropriate function based on dtype
        if array.dtype == cp.float32:
            result = self.lib.reduce_float_ptr(c_void_p(dev_ptr), c_int(array.size))
        elif array.dtype == cp.float64:
            result = self.lib.reduce_double_ptr(c_void_p(dev_ptr), c_int(array.size))
        elif array.dtype == cp.int32:
            result = self.lib.reduce_int_ptr(c_void_p(dev_ptr), c_int(array.size))
        else:
            raise TypeError(f"Unsupported dtype: {array.dtype}")
        
        # Ensure operation completes
        cp.cuda.runtime.deviceSynchronize()
        return result
    
    def transform(self, array):
        """
        Transform (square) a CuPy array in-place using Thrust.
        
        Parameters:
        -----------
        array : cupy.ndarray
            Array to transform
        """
        if not isinstance(array, cp.ndarray):
            raise TypeError("Input must be a CuPy array")
        
        # Ensure array is contiguous
        if not array.flags.c_contiguous:
            array = cp.ascontiguousarray(array)
        
        # Get raw device pointer
        dev_ptr = array.data.ptr
        
        # Call the appropriate function based on dtype
        if array.dtype == cp.float32:
            self.lib.transform_float_ptr(c_void_p(dev_ptr), c_int(array.size))
        elif array.dtype == cp.float64:
            self.lib.transform_double_ptr(c_void_p(dev_ptr), c_int(array.size))
        elif array.dtype == cp.int32:
            self.lib.transform_int_ptr(c_void_p(dev_ptr), c_int(array.size))
        else:
            raise TypeError(f"Unsupported dtype: {array.dtype}")
        
        # Ensure operation completes
        cp.cuda.runtime.deviceSynchronize()

def verify_sort_with_sequential_data():
    """
    Verify sorting with a sequential dataset to ensure both libraries work correctly.
    """
    print("\n--- SORTING VERIFICATION TEST ---\n")
    
    # Initialize the wrapper
    thrust = ThrustWrapper()
    
    # Create a reversed sequential array for easy verification
    size = 20
    sequential_data = np.arange(size, 0, -1, dtype=np.int32)  # [20, 19, 18, ..., 1]
    expected_result = np.arange(1, size+1, dtype=np.int32)    # [1, 2, 3, ..., 20]
    
    print(f"Original data:  {sequential_data}")
    print(f"Expected result: {expected_result}")
    
    # Test CuPy sort
    cupy_data = cp.array(sequential_data)
    print("\nCuPy sort:")
    print("  Before: ", cupy_data.get())
    # Use explicit assignment since we know in-place doesn't work
    cupy_data = cp.sort(cupy_data)
    print("  After:  ", cupy_data.get())
    print(f"  Correct: {np.array_equal(cupy_data.get(), expected_result)}")
    
    # Test Thrust sort
    thrust_data = cp.array(sequential_data)
    print("\nThrust sort:")
    print("  Before: ", thrust_data.get())
    thrust.sort(thrust_data)
    print("  After:  ", thrust_data.get())
    print(f"  Correct: {np.array_equal(thrust_data.get(), expected_result)}")
    
    # Compare results
    print("\nResults match:", np.array_equal(cupy_data.get(), thrust_data.get()))
    
    print("\n--- END VERIFICATION TEST ---\n")

def run_benchmark(sizes=[10000, 100000, 1000000], dtypes=[np.float32, np.float64, np.int32], num_trials=3):
    """
    Run benchmarks comparing Thrust and CuPy performance.
    
    Parameters:
    -----------
    sizes : list
        List of array sizes to test
    dtypes : list
        List of data types to test
    num_trials : int
        Number of trials for each test
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing benchmark results
    """
    # Initialize Thrust wrapper
    thrust = ThrustWrapper()
    
    results = []
    
    for dtype in dtypes:
        cp_dtype = getattr(cp, dtype.__name__)
        
        for size in sizes:
            print(f"Testing {dtype.__name__} with size {size}")
            
            for trial in range(num_trials):
                # Generate random test data on GPU
                if np.issubdtype(dtype, np.integer):
                    data = cp.random.randint(0, 1000, size=size, dtype=cp_dtype)
                else:
                    data = cp.random.random(size=size).astype(cp_dtype)
                
                # Create copies for each operation
                sort_data_cupy = cp.copy(data)
                sort_data_thrust = cp.copy(data)
                reduce_data_cupy = cp.copy(data)
                reduce_data_thrust = cp.copy(data)
                transform_data_cupy = cp.copy(data)
                transform_data_thrust = cp.copy(data)
                
                # Sort benchmark
                # CuPy
                cp.cuda.runtime.deviceSynchronize()
                start = time.time()
                sort_data_cupy = cp.sort(sort_data_cupy)  # Use explicit assignment
                cp.cuda.runtime.deviceSynchronize()
                cupy_sort_time = time.time() - start
                
                # Thrust
                cp.cuda.runtime.deviceSynchronize()
                start = time.time()
                thrust.sort(sort_data_thrust)
                cp.cuda.runtime.deviceSynchronize()
                thrust_sort_time = time.time() - start
                
                # Verify sort results match
                sort_match = cp.allclose(sort_data_cupy, sort_data_thrust, rtol=1e-5, atol=1e-8)
                
                # Reduce benchmark
                # CuPy
                cp.cuda.runtime.deviceSynchronize()
                start = time.time()
                cupy_sum = float(cp.sum(reduce_data_cupy))
                cp.cuda.runtime.deviceSynchronize()
                cupy_reduce_time = time.time() - start
                
                # Thrust
                cp.cuda.runtime.deviceSynchronize()
                start = time.time()
                thrust_sum = float(thrust.reduce(reduce_data_thrust))
                cp.cuda.runtime.deviceSynchronize()
                thrust_reduce_time = time.time() - start
                
                # Verify reduce results match
                if np.issubdtype(dtype, np.floating):
                    reduce_match = abs(cupy_sum - thrust_sum) / max(abs(cupy_sum), 1e-10) < 1e-5
                else:
                    reduce_match = cupy_sum == thrust_sum
                
                # Transform benchmark
                # CuPy
                cp.cuda.runtime.deviceSynchronize()
                start = time.time()
                transform_data_cupy = transform_data_cupy * transform_data_cupy
                cp.cuda.runtime.deviceSynchronize()
                cupy_transform_time = time.time() - start
                
                # Thrust
                cp.cuda.runtime.deviceSynchronize()
                start = time.time()
                thrust.transform(transform_data_thrust)
                cp.cuda.runtime.deviceSynchronize()
                thrust_transform_time = time.time() - start
                
                # Verify transform results match
                transform_match = cp.allclose(transform_data_cupy, transform_data_thrust, rtol=1e-5, atol=1e-8)
                
                # Store results
                results.append({
                    'size': size,
                    'dtype': dtype.__name__,
                    'trial': trial,
                    'cupy_sort_time': cupy_sort_time,
                    'thrust_sort_time': thrust_sort_time,
                    'sort_match': sort_match,
                    'cupy_reduce_time': cupy_reduce_time,
                    'thrust_reduce_time': thrust_reduce_time,
                    'reduce_match': reduce_match,
                    'cupy_transform_time': cupy_transform_time,
                    'thrust_transform_time': thrust_transform_time,
                    'transform_match': transform_match
                })
    
    return pd.DataFrame(results)

def verify_accuracy(dtype=np.float32, size=10000, thrust=None):
    """
    Verify that Thrust and CuPy produce identical results.
    
    Parameters:
    -----------
    dtype : numpy.dtype
        Data type to test
    size : int
        Size of test array
    thrust : ThrustWrapper, optional
        An existing ThrustWrapper instance to use
        
    Returns:
    --------
    dict
        Dictionary with accuracy verification results
    """
    # Initialize Thrust wrapper if not provided
    if thrust is None:
        thrust = ThrustWrapper()
    
    # Convert numpy dtype to CuPy dtype
    cp_dtype = getattr(cp, dtype.__name__)
    
    # Create random data
    if np.issubdtype(dtype, np.integer):
        data = cp.random.randint(0, 1000, size=size, dtype=cp_dtype)
    else:
        data = cp.random.random(size=size).astype(cp_dtype)
    
    # Test sort
    sort_data_cupy = cp.copy(data)
    sort_data_thrust = cp.copy(data)
    
    # Sort with both libraries
    sort_data_cupy = cp.sort(sort_data_cupy)  # Make sure to use assignment!
    thrust.sort(sort_data_thrust)

    # In verify_accuracy function
    if np.issubdtype(dtype, np.integer):
        min_val = cp.min(sort_data_cupy).get()
        max_val = cp.max(sort_data_cupy).get()
        print(f"\n\n****** Range of values in {dtype.__name__} array: {min_val} to {max_val} ******")
    
    # Debug output - add this
    print(f"\nSort comparison (first 10 elements):")
    for i in range(min(10, size)):
        print(f"Index {i}: CuPy={float(sort_data_cupy[i]):.8f}, Thrust={float(sort_data_thrust[i]):.8f}, " +
              f"Equal={abs(float(sort_data_cupy[i] - sort_data_thrust[i])) < 1e-5}")
    
    sort_match = cp.allclose(sort_data_cupy, sort_data_thrust, rtol=1e-5, atol=1e-8)
    
    # Test reduce
    reduce_data_cupy = cp.copy(data)
    reduce_data_thrust = cp.copy(data)
    
    cupy_sum = float(cp.sum(reduce_data_cupy))
    thrust_sum = float(thrust.reduce(reduce_data_thrust))
    
    if np.issubdtype(dtype, np.floating):
        reduce_match = abs(cupy_sum - thrust_sum) / max(abs(cupy_sum), 1e-10) < 1e-5
    else:
        reduce_match = cupy_sum == thrust_sum
    
    # Test transform
    transform_data_cupy = cp.copy(data)
    transform_data_thrust = cp.copy(data)
    
    transform_data_cupy = transform_data_cupy * transform_data_cupy
    thrust.transform(transform_data_thrust)
    
    transform_match = cp.allclose(transform_data_cupy, transform_data_thrust, rtol=1e-5, atol=1e-8)
    
    return {
        'sort': sort_match,
        'reduce': reduce_match,
        'transform': transform_match
    }

def debug_sort_mismatch(benchmark_data, size=100):
    """Debug sort mismatch with actual benchmark data."""
    thrust = ThrustWrapper()
    
    # Make copies for testing
    cupy_data = cp.copy(benchmark_data)
    thrust_data = cp.copy(benchmark_data)
    
    # Sort with both libraries
    cupy_data = cp.sort(cupy_data)  # Explicit assignment
    thrust.sort(thrust_data)
    
    # Check if results match
    match = cp.allclose(cupy_data, thrust_data, rtol=1e-5, atol=1e-8)
    print(f"Sort results match: {match}")
    
    # Check if arrays are actually sorted
    cupy_sorted = all(float(cupy_data[i]) <= float(cupy_data[i+1]) for i in range(len(cupy_data)-1))
    thrust_sorted = all(float(thrust_data[i]) <= float(thrust_data[i+1]) for i in range(len(thrust_data)-1))
    
    print(f"CuPy array is sorted: {cupy_sorted}")
    print(f"Thrust array is sorted: {thrust_sorted}")
    
    # If they don't match but both are sorted, show samples
    if not match and cupy_sorted and thrust_sorted:
        print("\nBoth arrays are correctly sorted but don't match. Sampling:")
        
        # Show first few elements
        print("\nFirst 10 elements:")
        for i in range(min(10, len(cupy_data))):
            print(f"{i:4d}: CuPy={float(cupy_data[i]):.8f}, Thrust={float(thrust_data[i]):.8f}, " +
                  f"Diff={float(cupy_data[i] - thrust_data[i]):.8f}")
        
        # Show elements where differences are largest
        diffs = np.abs(cupy_data.get() - thrust_data.get())
        largest_diff_indices = np.argsort(diffs)[-10:]
        
        print("\nElements with largest differences:")
        for i in largest_diff_indices:
            print(f"{i:4d}: CuPy={float(cupy_data[i]):.8f}, Thrust={float(thrust_data[i]):.8f}, " +
                  f"Diff={float(cupy_data[i] - thrust_data[i]):.8f}")

def calculate_speedups(df):
    """
    Calculate speedup ratios between CuPy and Thrust.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with benchmark results
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with speedup ratios
    """
    # Group by size and dtype and calculate means
    summary = df.groupby(['size', 'dtype']).agg({
        'cupy_sort_time': 'mean',
        'thrust_sort_time': 'mean',
        'cupy_reduce_time': 'mean',
        'thrust_reduce_time': 'mean',
        'cupy_transform_time': 'mean',
        'thrust_transform_time': 'mean',
        'sort_match': 'all',
        'reduce_match': 'all',
        'transform_match': 'all'
    }).reset_index()
    
    # Calculate speedups (ratio of CuPy time to Thrust time)
    # Values > 1 mean Thrust is faster, < 1 mean CuPy is faster
    summary['sort_speedup'] = summary['cupy_sort_time'] / summary['thrust_sort_time']
    summary['reduce_speedup'] = summary['cupy_reduce_time'] / summary['thrust_reduce_time']
    summary['transform_speedup'] = summary['cupy_transform_time'] / summary['thrust_transform_time']
    
    return summary

def plot_benchmark_results(df, output_dir='gpu_benchmark_results'):
    """
    Create plots of benchmark results.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with benchmark results
    output_dir : str
        Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate speedups
    speedups = calculate_speedups(df)
    
    # Save to CSV
    df.to_csv(f'{output_dir}/raw_results.csv', index=False)
    speedups.to_csv(f'{output_dir}/speedup_summary.csv', index=False)
    
    # Plot execution times
    for op_type in ['sort', 'reduce', 'transform']:
        plt.figure(figsize=(10, 6))
        
        for dtype in df['dtype'].unique():
            dtype_data = speedups[speedups['dtype'] == dtype]
            
            # Plot CuPy times
            plt.plot(dtype_data['size'], 
                     dtype_data[f'cupy_{op_type}_time'], 
                     'o-', label=f'CuPy {dtype}')
            
            # Plot Thrust times
            plt.plot(dtype_data['size'], 
                     dtype_data[f'thrust_{op_type}_time'], 
                     's--', label=f'Thrust {dtype}')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Array Size')
        plt.ylabel('Time (seconds)')
        plt.title(f'{op_type.capitalize()} Performance: CuPy vs Thrust')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{op_type}_performance.png', dpi=300)
        plt.close()
    
    # Plot speedups
    plt.figure(figsize=(10, 6))
    
    for dtype in speedups['dtype'].unique():
        dtype_data = speedups[speedups['dtype'] == dtype]
        
        for op_type in ['sort', 'reduce', 'transform']:
            plt.plot(dtype_data['size'], 
                     dtype_data[f'{op_type}_speedup'], 
                     'o-', label=f'{op_type.capitalize()} ({dtype})')
    
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Array Size')
    plt.ylabel('Speedup (CuPy time / Thrust time)')
    plt.title('Performance Comparison: Values > 1 mean Thrust is faster')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/speedup_comparison.png', dpi=300)
    plt.close()

def display_sample_data(dtype=np.float32, size=20):
    """
    Display sample data to compare CuPy and Thrust results.
    
    Parameters:
    -----------
    dtype : numpy.dtype
        Data type to use
    size : int
        Number of elements to display
    """
    # Initialize Thrust wrapper
    thrust = ThrustWrapper()
    
    # Convert numpy dtype to CuPy dtype
    cp_dtype = getattr(cp, dtype.__name__)
    
    # Create random data
    if np.issubdtype(dtype, np.integer):
        data = cp.random.randint(0, 1000, size=size, dtype=cp_dtype)
    else:
        data = cp.random.random(size=size).astype(cp_dtype)
    
    print(f"Original data ({dtype.__name__}):")
    print(data)
    print("\n" + "-"*70 + "\n")
    
    # Test sort
    sort_data_cupy = cp.copy(data)
    sort_data_thrust = cp.copy(data)
    
    cp.sort(sort_data_cupy)
    thrust.sort(sort_data_thrust)
    
    print("Sorted results:")
    print("CuPy:   ", sort_data_cupy)
    print("Thrust: ", sort_data_thrust)
    print("Match:  ", cp.allclose(sort_data_cupy, sort_data_thrust, rtol=1e-5, atol=1e-8))
    print("\n" + "-"*70 + "\n")
    
    # Test transform
    transform_data_cupy = cp.copy(data)
    transform_data_thrust = cp.copy(data)
    
    transform_data_cupy = transform_data_cupy * transform_data_cupy
    thrust.transform(transform_data_thrust)
    
    print("Transform results (square):")
    print("CuPy:   ", transform_data_cupy)
    print("Thrust: ", transform_data_thrust)
    print("Match:  ", cp.allclose(transform_data_cupy, transform_data_thrust, rtol=1e-5, atol=1e-8))
    print("\n" + "-"*70 + "\n")
    
    # Test reduce
    reduce_data_cupy = cp.copy(data)
    reduce_data_thrust = cp.copy(data)
    
    cupy_sum = float(cp.sum(reduce_data_cupy))
    thrust_sum = float(thrust.reduce(reduce_data_thrust))
    
    print("Reduce results (sum):")
    print("CuPy:   ", cupy_sum)
    print("Thrust: ", thrust_sum)
    
    if np.issubdtype(dtype, np.floating):
        rel_diff = abs(cupy_sum - thrust_sum) / max(abs(cupy_sum), 1e-10)
        print(f"Relative difference: {rel_diff:.2e}")
        reduce_match = rel_diff < 1e-5
    else:
        reduce_match = cupy_sum == thrust_sum
    
    print("Match:  ", reduce_match)

def main():
    """Run the benchmark with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU Thrust vs CuPy Benchmark')
    parser.add_argument('--sizes', type=int, nargs='+', 
                        default=[10000, 100000, 1000000],
                        help='Sizes of arrays to benchmark')
    parser.add_argument('--trials', type=int, default=3,
                        help='Number of trials for each test')
    parser.add_argument('--output-dir', type=str, default='gpu_benchmark_results',
                        help='Directory to save results')
    parser.add_argument('--lib-path', type=str, default='./libthrust_fortran.so',
                        help='Path to Thrust shared library')
    parser.add_argument('--sample', action='store_true',
                        help='Display sample data for verification')
    # Add the debug flag
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output for sort mismatch')
    args = parser.parse_args()
    
    # Create a single instance
    thrust = ThrustWrapper()
    print("Verifying accuracy...")
    for dtype in [np.float32, np.float64, np.int32]:
        print(f"\nAccuracy Results for {dtype.__name__}:")
        accuracy = verify_accuracy(dtype=dtype, thrust=thrust)

    # If sample flag is set, just display sample data
    if args.sample:
        print("Displaying sample results for verification:")
        for dtype in [np.float32, np.float64, np.int32]:
            display_sample_data(dtype=dtype, size=20)
            print("\n")
        return
    
    # First run verification test
    verify_sort_with_sequential_data()

    # Run benchmarks with debugging if requested
    if args.debug:
        print("Running benchmark with debugging enabled")
        # Generate test data for each type and debug it
        debug_size = min(10000, args.sizes[0])  # Use the smallest requested size
        
        for dtype in [np.float32, np.float64, np.int32]:
            print(f"\n--- Debugging sort for {dtype.__name__} with size {debug_size} ---")
            
            # Generate consistent test data for debugging
            cp_dtype = getattr(cp, dtype.__name__)
            if np.issubdtype(dtype, np.integer):
                data = cp.random.randint(0, 1000, size=debug_size, dtype=cp_dtype)
            else:
                data = cp.random.random(size=debug_size).astype(cp_dtype)
                
            # Call debug function
            debug_sort_mismatch(data, size=debug_size)
            
        # Ask if user wants to continue with full benchmark
        response = input("\nContinue with full benchmark? (y/n): ")
        if response.lower() != 'y':
            return

    # Run benchmarks
    print(f"Running benchmark with sizes: {args.sizes}")
    results = run_benchmark(
        sizes=args.sizes,
        dtypes=[np.float32, np.float64, np.int32],
        num_trials=args.trials
    )
    
    # Plot and save results
    plot_benchmark_results(results, args.output_dir)
    
    # Print summary
    speedups = calculate_speedups(results)
    print("\nSpeedup ratios (CuPy time / Thrust time):")
    print("Values > 1 mean Thrust is faster, < 1 mean CuPy is faster")
    print(speedups[['size', 'dtype', 'sort_speedup', 'reduce_speedup', 'transform_speedup']])
    
    # Verify accuracy for all data types
    print("\nVerifying accuracy...")
    for dtype in [np.float32, np.float64, np.int32]:
        accuracy = verify_accuracy(dtype=dtype)
        print(f"Accuracy Results for {dtype.__name__}:")
        for op, match in accuracy.items():
            print(f"{op.capitalize()}: {'MATCH' if match else 'MISMATCH'}")

if __name__ == "__main__":
    main()