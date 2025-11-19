#!/usr/bin/env python3
"""
v28 Baseline Performance Visualization
Creates publication-quality charts comparing PyTorch vs CUDA Fortran performance
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set up publication-quality plot style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Performance data (15 epochs)
datasets = ['CIFAR-10', 'Fashion-MNIST', 'CIFAR-100', 'SVHN']
pytorch_times = [147, 87, 144, 176]  # seconds
fortran_times = [31, 28, 35, 40]     # seconds

# Calculate speedups
speedups = [p / f for p, f in zip(pytorch_times, fortran_times)]
avg_speedup = np.mean(speedups)

# Accuracy data (if available, update these)
accuracies = {
    'CIFAR-10': {'pytorch': 78.5, 'fortran': 78.9},
    'Fashion-MNIST': {'pytorch': 92.0, 'fortran': 92.3},
    'CIFAR-100': {'pytorch': 52.0, 'fortran': 52.5},
    'SVHN': {'pytorch': 89.0, 'fortran': 89.2}
}

# Color scheme
color_pytorch = '#FF6B6B'    # Coral red
color_fortran = '#4ECDC4'    # Turquoise
color_speedup = '#45B7D1'    # Sky blue
color_grid = '#e0e0e0'

def create_training_time_comparison():
    """Create side-by-side bar chart comparing training times"""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(datasets))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, pytorch_times, width, label='PyTorch (CPU)',
                   color=color_pytorch, alpha=0.85, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, fortran_times, width, label='v28 Fortran (CUDA)',
                   color=color_fortran, alpha=0.85, edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}s',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Customize plot
    ax.set_xlabel('Dataset', fontweight='bold', fontsize=13)
    ax.set_ylabel('Training Time (seconds)', fontweight='bold', fontsize=13)
    ax.set_title('Training Time Comparison: PyTorch vs v28 CUDA Fortran\n(15 Epochs)',
                fontweight='bold', fontsize=15, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--', color=color_grid)
    ax.set_axisbelow(True)

    # Add average info box
    textstr = f'Average Speedup: {avg_speedup:.1f}×'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='orange', linewidth=2)
    ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', horizontalalignment='right', bbox=props,
           fontweight='bold')

    plt.tight_layout()
    return fig

def create_speedup_chart():
    """Create horizontal bar chart showing speedup factors"""
    fig, ax = plt.subplots(figsize=(12, 6))

    y_pos = np.arange(len(datasets))

    # Create horizontal bars
    bars = ax.barh(y_pos, speedups, color=color_speedup, alpha=0.85,
                   edgecolor='white', linewidth=2, height=0.6)

    # Add value labels
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        width = bar.get_width()
        ax.text(width + 0.15, bar.get_y() + bar.get_height()/2.,
               f'{speedup:.1f}× faster',
               ha='left', va='center', fontweight='bold', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor=color_speedup, alpha=0.9))

    # Add vertical line at average
    ax.axvline(avg_speedup, color='red', linestyle='--', linewidth=2,
              alpha=0.7, label=f'Average: {avg_speedup:.1f}×')

    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(datasets, fontweight='bold')
    ax.set_xlabel('Speedup Factor (×)', fontweight='bold', fontsize=13)
    ax.set_title('v28 CUDA Fortran Performance Advantage\n(Speedup over PyTorch CPU)',
                fontweight='bold', fontsize=15, pad=20)
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='gray')
    ax.grid(axis='x', alpha=0.3, linestyle='--', color=color_grid)
    ax.set_axisbelow(True)
    ax.set_xlim([0, max(speedups) + 1])

    plt.tight_layout()
    return fig

def create_combined_chart():
    """Create comprehensive chart with multiple metrics"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Training Time Comparison
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax1.bar(x - width/2, pytorch_times, width, label='PyTorch (CPU)',
                    color=color_pytorch, alpha=0.85, edgecolor='white', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, fortran_times, width, label='v28 Fortran (CUDA)',
                    color=color_fortran, alpha=0.85, edgecolor='white', linewidth=1.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}s',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax1.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax1.set_title('Training Time Comparison (15 Epochs)', fontweight='bold', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontweight='bold')
    ax1.legend(loc='upper left', framealpha=0.95)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # 2. Speedup Factors
    ax2 = fig.add_subplot(gs[1, 0])
    colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(datasets)))
    bars = ax2.barh(datasets, speedups, color=colors_gradient, alpha=0.85,
                    edgecolor='white', linewidth=2)

    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2.,
                f'{speedup:.1f}×',
                ha='left', va='center', fontweight='bold', fontsize=10)

    ax2.axvline(avg_speedup, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Avg: {avg_speedup:.1f}×')
    ax2.set_xlabel('Speedup Factor', fontweight='bold')
    ax2.set_title('Performance Speedup', fontweight='bold', fontsize=13)
    ax2.legend(framealpha=0.95)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # 3. Time Savings
    ax3 = fig.add_subplot(gs[1, 1])
    time_saved = [p - f for p, f in zip(pytorch_times, fortran_times)]
    bars = ax3.bar(datasets, time_saved, color='#2ECC71', alpha=0.85,
                   edgecolor='white', linewidth=2)

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}s',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax3.set_ylabel('Time Saved (seconds)', fontweight='bold')
    ax3.set_title('Time Savings per Training Run', fontweight='bold', fontsize=13)
    ax3.set_xticklabels(datasets, rotation=15, ha='right', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)

    # Add main title
    fig.suptitle('v28 CUDA Fortran Baseline Performance Summary',
                fontsize=18, fontweight='bold', y=0.98)

    # Add footer with stats
    footer_text = (f'Platform: NVIDIA GPU | Framework: v28 CUDA Fortran | '
                  f'Average Speedup: {avg_speedup:.1f}× | Total Time Saved: {sum(time_saved)}s per run')
    fig.text(0.5, 0.01, footer_text, ha='center', fontsize=9,
            style='italic', color='gray')

    plt.tight_layout()
    return fig

def save_all_charts(output_dir='performance_charts'):
    """Generate and save all charts"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("Generating performance charts...")
    print("=" * 70)

    # Generate charts
    charts = [
        ("training_time_comparison.png", create_training_time_comparison()),
        ("speedup_chart.png", create_speedup_chart()),
        ("combined_performance.png", create_combined_chart()),
    ]

    # Save charts
    for filename, fig in charts:
        filepath = output_path / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved: {filepath}")
        plt.close(fig)

    print("=" * 70)
    print(f"\nAll charts saved to: {output_path.absolute()}")
    print(f"\nPerformance Summary:")
    print(f"  Average Speedup: {avg_speedup:.1f}×")
    print(f"  Best Speedup: {max(speedups):.1f}× ({datasets[speedups.index(max(speedups))]})")
    print(f"  Total Time Saved: {sum([p-f for p,f in zip(pytorch_times, fortran_times)])}s per run")
    print()

if __name__ == '__main__':
    save_all_charts()
