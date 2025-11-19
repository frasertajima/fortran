"""
Brute-force test all possible FC1 weight geometries to find the correct one.
"""

import sys
sys.path.insert(0, 'inference')

import torch
import numpy as np
from pathlib import Path
from model_loader import load_v28_model, load_fortran_binary

def test_permutations(model_dir, test_data, test_labels, n_test=100):
    """
    Brute-force the correct FC1 weight geometry.
    """
    print("="*70)
    print(f"🔍 Diagnosing FC1 Geometry from: {model_dir}")
    print("="*70)
    print()
    
    # 1. Setup Model
    model = load_v28_model(model_dir, device='cpu')
    original_fc1 = model.fc1.weight.data.clone()
    
    # 2. Dimensions
    out_features = 512
    channels = 128
    height = 4
    width = 4
    
    # 3. Define Candidates
    candidates = []

    # Case A: Current baseline (what we have now - 12% accuracy)
    candidates.append({
        "name": "Baseline (Current)",
        "weight": original_fc1
    })

    # Case B: Transpose the entire weight matrix
    # Fortran might store (In, Out) instead of (Out, In)
    w_raw = np.fromfile(f"{model_dir}/fc1_weights.bin", dtype=np.float32)
    w_transposed_load = w_raw.reshape(2048, 512, order='F').T
    candidates.append({
        "name": "Transposed Load (2048,512).T",
        "weight": torch.from_numpy(w_transposed_load)
    })

    # Case C: Spatial First (H, W, C) -> (C, H, W)
    w_reshaped = original_fc1.view(out_features, height, width, channels) 
    w_permuted = w_reshaped.permute(0, 3, 1, 2).reshape(out_features, -1)
    candidates.append({
        "name": "Spatial First (H,W,C)->(C,H,W)",
        "weight": w_permuted
    })

    # Case D: F-order load, C-order flatten (what we just tried)
    w_f_order = w_raw.reshape((out_features, channels, height, width), order='F')
    w_f_fixed = w_f_order.reshape(out_features, -1, order='C')
    candidates.append({
        "name": "F-order Reconstruction",
        "weight": torch.from_numpy(w_f_fixed)
    })
    
    # Case E: Try (C, W, H) instead of (C, H, W)
    w_cwh = original_fc1.view(out_features, channels, width, height)
    w_cwh_fixed = w_cwh.permute(0, 1, 3, 2).reshape(out_features, -1)
    candidates.append({
        "name": "Swapped Spatial (C,W,H)->(C,H,W)",
        "weight": w_cwh_fixed
    })
    
    # Case F: Complete transpose with F-order
    w_f_transpose = w_raw.reshape(2048, 512, order='C').T
    candidates.append({
        "name": "C-order Transposed Load",
        "weight": torch.from_numpy(w_f_transpose)
    })

    # 4. Run Inference Test
    print("📊 Testing Candidates on", n_test, "images:")
    print("-" * 70)
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for cand in candidates:
            # Swap weights
            model.fc1.weight.data = cand['weight']
            
            # Test on multiple images
            correct = 0
            for i in range(n_test):
                img = test_data[i].reshape((3, 32, 32))
                img_tensor = torch.from_numpy(img).unsqueeze(0)
                output = model(img_tensor)
                pred = output.argmax(dim=1).item()
                if pred == test_labels[i]:
                    correct += 1
            
            accuracy = 100.0 * correct / n_test
            results.append((cand['name'], accuracy))
            
            status = "✅ WINNER!" if accuracy > 70 else ("🔶 Better" if accuracy > 15 else "❌ Failed")
            print(f"{cand['name']:<35} | Accuracy: {accuracy:5.1f}% | {status}")
    
    print()
    print("="*70)
    best = max(results, key=lambda x: x[1])
    print(f"🏆 Best Result: {best[0]} with {best[1]:.1f}% accuracy")
    print("="*70)
    
    return results

# --- USAGE ---
if __name__ == "__main__":
    # Load test data
    test_data = np.fromfile('datasets/cifar10/cifar10_data/images_test.bin', dtype=np.float32)
    test_data = test_data.reshape((3072, 10000), order='C').T
    test_labels = np.fromfile('datasets/cifar10/cifar10_data/labels_test.bin', dtype=np.int32)
    
    results = test_permutations("datasets/cifar10/saved_models/cifar10/", test_data, test_labels, n_test=1000)
