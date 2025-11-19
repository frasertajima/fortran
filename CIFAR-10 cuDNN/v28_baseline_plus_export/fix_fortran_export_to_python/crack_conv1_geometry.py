"""
Brute-Force Conv1 Geometry Cracker
Systematically tests all valid permutations to find the correct weight loading.
"""

import torch
import torch.nn.functional as F
import numpy as np
import itertools
from pathlib import Path

def crack_conv1_geometry(debug_dir, model_dir):
    print("="*70)
    print("🔓 Starting Systematic Brute-Force on Conv1...")
    print("="*70)
    print()

    # 1. Load Artifacts
    debug_path = Path(debug_dir)
    model_path = Path(model_dir)
    
    # Load raw flat weights (Do not reshape yet)
    w_raw = np.fromfile(model_path / 'conv1_weights.bin', dtype=np.float32)
    
    # Load Bias
    bias_raw = np.fromfile(model_path / 'conv1_bias.bin', dtype=np.float32)
    bias = torch.from_numpy(bias_raw)
    
    # Load Input
    inp_raw = np.fromfile(debug_path / 'debug_input.bin', dtype=np.float32)
    # Reshape from Fortran (W,H,C,1) to Python (1,C,H,W)
    fort_whc = inp_raw.reshape((32, 32, 3, 1), order='F')
    inp_chw = np.zeros((1, 3, 32, 32), dtype=np.float32)
    for c in range(3):
        for h in range(32):
            for w in range(32):
                inp_chw[0, c, h, w] = fort_whc[w, h, c, 0]
    t_input = torch.from_numpy(inp_chw)
    
    # Load Golden Output
    golden_raw = np.fromfile(debug_path / 'debug_conv1.bin', dtype=np.float32)
    # Reshape from Fortran (W,H,C,1) to Python (1,C,H,W)
    fort_golden_whc = golden_raw.reshape((32, 32, 32, 1), order='F')
    golden_chw = np.zeros((1, 32, 32, 32), dtype=np.float32)
    for c in range(32):
        for h in range(32):
            for w in range(32):
                golden_chw[0, c, h, w] = fort_golden_whc[w, h, c, 0]
    
    flat_gold = golden_chw.flatten()
    
    print(f"Loaded:")
    print(f"  Weights: {len(w_raw)} elements")
    print(f"  Input: {t_input.shape}")
    print(f"  Golden: {golden_chw.shape}")
    print()
    
    # 2. Define The Search Space
    dims = [32, 3, 3, 3]
    
    # A. Load Orders
    orders = ['F', 'C']
    
    # B. Reshape Permutations
    shapes_to_try = set(itertools.permutations(dims))
    
    # C. Transpose Permutations
    axes_perms = list(itertools.permutations([0, 1, 2, 3]))
    
    # D. Spatial Flips
    flips = [None, [2, 3]]  # Flip H, W
    
    total_tests = len(orders) * len(shapes_to_try) * len(axes_perms) * len(flips)
    print(f"Testing ~{total_tests} combinations...")
    print()
    
    best_corr = -1.0
    best_config = None
    tests_run = 0

    # 3. Run The Cracker
    for order in orders:
        for shape in shapes_to_try:
            # Try reshaping the flat binary
            try:
                w_reshaped = w_raw.reshape(shape, order=order)
                w_tensor = torch.from_numpy(w_reshaped)
            except:
                continue
                
            for perm in axes_perms:
                # Permute into PyTorch Native (32, 3, 3, 3)
                try:
                    w_permuted = w_tensor.permute(perm)
                except:
                    continue
                
                # Constraint: Result must be (32, 3, 3, 3)
                if w_permuted.shape != (32, 3, 3, 3):
                    continue

                for flip in flips:
                    tests_run += 1
                    w_final = w_permuted.contiguous()
                    if flip:
                        w_final = torch.flip(w_final, flip)
                        
                    # 4. Test Inference
                    try:
                        out = F.conv2d(t_input, w_final, bias=bias, stride=1, padding=1)
                        out_np = out.detach().numpy()
                        
                        # Correlation Check
                        flat_out = out_np.flatten()
                        
                        # Quick check on subset first
                        corr_quick = np.corrcoef(flat_out[:1000], flat_gold[:1000])[0, 1]
                        
                        if corr_quick > 0.95:
                            # Full correlation check
                            corr = np.corrcoef(flat_out, flat_gold)[0, 1]
                            
                            if corr > best_corr:
                                best_corr = corr
                                best_config = {
                                    "Order": order,
                                    "Load Shape": shape,
                                    "Permute": perm,
                                    "Flip": "YES" if flip else "NO"
                                }
                                
                                # Early Exit on Perfect Match
                                if corr > 0.999:
                                    print("\n" + "="*70)
                                    print("✨ EUREKA! PERFECT CONFIGURATION FOUND ✨")
                                    print("="*70)
                                    print(f"Correlation: {corr:.6f}")
                                    print(f"Tests run: {tests_run}/{total_tests}")
                                    print()
                                    print(f"1. Reshape raw binary to: {shape} using order='{order}'")
                                    print(f"2. Permute dimensions by: {perm}")
                                    print(f"3. Flip Spatial: {best_config['Flip']}")
                                    print("="*70)
                                    print()
                                    print("Python code:")
                                    print(f"  w = np.fromfile(..., dtype=np.float32)")
                                    print(f"  w = w.reshape{shape}, order='{order}')")
                                    print(f"  w = w.transpose{perm}")
                                    if flip:
                                        print(f"  w = np.flip(w, axis={flip})")
                                    print("="*70)
                                    return best_config
                                    
                    except Exception as e:
                        pass
            
            # Progress indicator
            if tests_run % 100 == 0:
                print(f"  Progress: {tests_run}/{total_tests} tests, best corr: {best_corr:.4f}")

    print("\n" + "="*70)
    print("🏁 Search Complete.")
    print("="*70)
    print(f"Tests run: {tests_run}")
    print(f"Best Correlation: {best_corr:.6f}")
    if best_config:
        print()
        print("Best Config:")
        print(f"  Order: {best_config['Order']}")
        print(f"  Shape: {best_config['Load Shape']}")
        print(f"  Permute: {best_config['Permute']}")
        print(f"  Flip: {best_config['Flip']}")
    print("="*70)
    return best_config

# Run the cracker
if __name__ == "__main__":
    result = crack_conv1_geometry(
        'datasets/cifar10/saved_models/cifar10',
        'datasets/cifar10/saved_models/cifar10'
    )
