"""
Layer-by-Layer Forensics: Compare Fortran vs Python activations
"""

import sys
sys.path.insert(0, 'inference')

import torch
import numpy as np
from pathlib import Path
from model_loader import load_v28_model, load_fortran_binary

def compare_layers(model_dir, debug_dir):
    print("="*70)
    print("🕵️ Starting Layer-by-Layer Forensics")
    print("="*70)
    print(f"Model: {model_dir}")
    print(f"Debug: {debug_dir}")
    print()
    
    # 1. Load Model
    device = 'cpu'
    model = load_v28_model(model_dir, device=device)
    model.eval()
    
    # 2. Load Input (from Fortran export)
    debug_path = Path(debug_dir)
    try:
        # Fortran exports as (W,H,C,1) in F-order
        fortran_input_flat = np.fromfile(debug_path / 'debug_input.bin', dtype=np.float32)
        print(f"Loaded input: {len(fortran_input_flat)} values")
        
        # Reshape from Fortran's (W,H,C,1) to Python's (1,C,H,W)
        fort_whc = fortran_input_flat.reshape((32, 32, 3, 1), order='F')
        # Convert to (C,H,W)
        input_chw = np.zeros((3, 32, 32), dtype=np.float32)
        for c in range(3):
            for h in range(32):
                for w in range(32):
                    input_chw[c, h, w] = fort_whc[w, h, c, 0]
        
        input_tensor = torch.from_numpy(input_chw).unsqueeze(0).to(device)
        print(f"✅ Loaded debug input from Fortran export")
        print(f"   Shape: {input_tensor.shape}, Mean: {input_tensor.mean():.6f}")
    except Exception as e:
        print(f"❌ Error loading input: {e}")
        return
    
    print()
    
    # 3. Hook PyTorch Layers to capture output
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.bn1.register_forward_hook(get_activation('bn1'))
    model.pool1.register_forward_hook(get_activation('pool1'))
    model.conv2.register_forward_hook(get_activation('conv2'))
    model.bn2.register_forward_hook(get_activation('bn2'))
    model.pool2.register_forward_hook(get_activation('pool2'))
    
    # 4. Run Inference
    with torch.no_grad():
        output = model(input_tensor)
    
    print("="*70)
    print("Layer-by-Layer Comparison")
    print("="*70)
    print()
    
    # 5. Compare against Golden Files
    layers_to_check = [
        ('conv1', (32, 32, 32, 1)),  # (W,H,C,N) from Fortran
        ('bn1', (32, 32, 32, 1)),
        ('pool1', (16, 16, 32, 1)),
        ('conv2', (16, 16, 64, 1)),
        ('bn2', (16, 16, 64, 1)),
        ('pool2', (8, 8, 64, 1)),
    ]
    
    for layer_name, fort_shape in layers_to_check:
        bin_path = debug_path / f'debug_{layer_name}.bin'
        
        if not bin_path.exists():
            print(f"⚠️  Skipping {layer_name} - file not found")
            continue
        
        # Load Fortran Golden Vector
        try:
            golden_flat = np.fromfile(bin_path, dtype=np.float32)
            expected_size = np.prod(fort_shape)
            
            if len(golden_flat) != expected_size:
                print(f"❌ {layer_name}: Size mismatch - got {len(golden_flat)}, expected {expected_size}")
                continue
            
            # Reshape from Fortran (W,H,C,1)
            golden_whc = golden_flat.reshape(fort_shape, order='F')
            
            # Convert to Python (C,H,W) for comparison
            c, h, w = fort_shape[2], fort_shape[1], fort_shape[0]
            golden_chw = np.zeros((c, h, w), dtype=np.float32)
            for ci in range(c):
                for hi in range(h):
                    for wi in range(w):
                        golden_chw[ci, hi, wi] = golden_whc[wi, hi, ci, 0]
            
            golden_flat_chw = golden_chw.flatten()
            
        except Exception as e:
            print(f"❌ {layer_name}: Error loading - {e}")
            continue
        
        # Get Python output
        py_out = activations[layer_name][0].cpu().numpy().flatten()
        
        # Compare
        if golden_flat_chw.size != py_out.size:
            print(f"❌ {layer_name}: SIZE MISMATCH")
            print(f"   Fortran: {golden_flat_chw.size}, Python: {py_out.size}")
            break
        
        # Statistics
        corr = np.corrcoef(golden_flat_chw, py_out)[0, 1]
        mse = np.mean((golden_flat_chw - py_out)**2)
        max_diff = np.max(np.abs(golden_flat_chw - py_out))
        
        print(f"{layer_name:<15} | Corr: {corr:7.4f} | MSE: {mse:10.6f} | MaxDiff: {max_diff:8.4f}", end="")
        
        if corr < 0.9:
            print(f" 🚨 DIVERGENCE!")
            print(f"   Fortran first 10: {golden_flat_chw[:10]}")
            print(f"   Python first 10:  {py_out[:10]}")
            print(f"   Fortran stats: min={golden_flat_chw.min():.4f}, max={golden_flat_chw.max():.4f}, mean={golden_flat_chw.mean():.4f}")
            print(f"   Python stats:  min={py_out.min():.4f}, max={py_out.max():.4f}, mean={py_out.mean():.4f}")
            break
        else:
            print(" ✅")
    
    print()
    print("="*70)

if __name__ == "__main__":
    compare_layers(
        'datasets/cifar10/saved_models/cifar10/',
        'datasets/cifar10/saved_models/cifar10/'
    )
