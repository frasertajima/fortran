#!/bin/bash
#
# Run all PyTorch reference benchmarks and compare to v28 Fortran
#
# Usage: ./run_all_benchmarks.sh
#

echo "======================================================================"
echo "v28 PyTorch Reference Benchmarks"
echo "======================================================================"
echo ""
echo "This script runs all PyTorch reference implementations and compares"
echo "them to the v28 CUDA Fortran baseline."
echo ""
echo "Expected total runtime: ~4-5 minutes on V100"
echo "======================================================================"
echo ""

# Create results directory
mkdir -p results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="results/benchmark_${TIMESTAMP}.txt"

echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Function to run benchmark and capture output
run_benchmark() {
    local script=$1
    local dataset=$2

    echo "======================================================================"
    echo "Running: $dataset"
    echo "======================================================================"

    python "$script" | tee -a "$RESULTS_FILE"

    echo ""
    echo "----------------------------------------------------------------------"
    echo ""
}

# Run all benchmarks
echo "Starting benchmarks at $(date)" | tee "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

run_benchmark "cifar10_reference.py" "CIFAR-10"
run_benchmark "fashion_mnist_reference.py" "Fashion-MNIST"
run_benchmark "cifar100_reference.py" "CIFAR-100"
run_benchmark "svhn_reference.py" "SVHN"

echo "======================================================================"
echo "All benchmarks completed at $(date)"
echo "======================================================================"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "Summary:"
echo "--------"
echo "Review the results file for detailed timing and accuracy comparisons."
echo "All implementations should show ~2× speedup for v28 Fortran vs PyTorch."
