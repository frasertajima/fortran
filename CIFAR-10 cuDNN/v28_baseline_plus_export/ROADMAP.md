# v28 Baseline Plus Export - Project Roadmap

## Current Status (November 2025)

### ✅ Completed Features

**Core Framework:**
- ✅ Modular architecture (70% code reduction: 12K → 1.5K lines)
- ✅ 4 datasets fully integrated: CIFAR-10, CIFAR-100, Fashion-MNIST, SVHN
- ✅ **4× faster than PyTorch** (with persistent memory pools)
- ✅ Complete Fortran→Python export pipeline
- ✅ Jupyter notebook inference for all 4 datasets
- ✅ Comprehensive documentation (1,500+ lines)

**Performance:**
- ✅ GPU-only batch extraction (eliminates 75,000+ CPU↔GPU transfers)
- ✅ Blocking synchronization (5-10% CPU usage vs 100%)
- ✅ Persistent memory pools (achieved 4× speedup)
- ✅ 85-95% GPU utilization (compute-bound)

**Export System:**
- ✅ Binary weight export (19 files per model)
- ✅ PyTorch model loader with memory layout conversion
- ✅ Accuracy preservation (training: 79%, PyTorch inference: 81%)
- ✅ Dataset-specific inference notebooks with visualizations

### 🟡 Partially Complete

**Oxford Flowers 102:**
- ✅ Implemented in v28_baseline (dense layer only)
- ✅ 78.94% accuracy, 1.2s training time
- ❌ **Not yet integrated into v28_baseline_plus_export**
- ❌ **No export functionality**
- ❌ **No inference notebook**

---

## Missing Features & Improvements

### Priority 1: Production-Critical Features

#### 1. **Data Augmentation** 🎯
**Status:** Intentionally disabled for benchmarking
**Impact:** Could improve CIFAR-10 accuracy from ~79% to 85%+

**Implementation Plan:**
- Random crops, horizontal flips, rotations
- Color jittering for RGB datasets
- GPU-based augmentation (faster than CPU)
- Configurable per-dataset

**Effort:** 2-3 days
**Files to modify:**
- `common/data_augmentation.cuf` (new module)
- `datasets/*/config.cuf` (enable/disable flags)

**Trade-offs:**
- ✅ Better accuracy and generalization
- ✅ More realistic ML workflow
- ❌ Slightly slower training (minimal with GPU augmentation)
- ❌ Harder to compare with PyTorch baseline (need to match augmentations)

---

#### 2. **Model Checkpointing & Validation** 🎯
**Status:** Missing
**Impact:** Production necessity for real ML workflows

**Implementation Plan:**
- Save best model based on validation accuracy
- Early stopping to prevent overfitting
- Resume training from checkpoint
- Validation split (e.g., 10% of training data)

**Effort:** 3-4 days
**Files to create:**
- `common/checkpoint.cuf` - Checkpoint save/load
- `common/early_stopping.cuf` - Early stopping logic

**Files to modify:**
- `datasets/*/main.cuf` - Add validation loop

**Trade-offs:**
- ✅ Prevents overfitting
- ✅ Can resume long training runs
- ✅ Automatic best model selection
- ❌ Slightly more complex training loop
- ❌ Requires validation data split

---

#### 3. **Learning Rate Scheduling** 🎯
**Status:** Intentionally disabled for benchmarking
**Impact:** 2-5% accuracy improvement on all datasets

**Implementation Plan:**
- Cosine annealing with warmup
- Step decay (reduce LR every N epochs)
- Plateau detection (reduce on validation plateau)
- Configurable per-dataset

**Effort:** 2-3 days
**Files to create:**
- `common/lr_scheduler.cuf` - Scheduler implementations

**Files to modify:**
- `datasets/*/main.cuf` - Update learning rate each epoch

**Trade-offs:**
- ✅ Better convergence
- ✅ Higher final accuracy
- ✅ Industry standard practice
- ❌ More hyperparameters to tune
- ❌ Longer training time (more epochs needed)

---

### Priority 2: Usability & Developer Experience

#### 4. **YAML/JSON Configuration Files** 🚀
**Status:** Currently using Fortran modules
**Impact:** Much easier to modify hyperparameters

**Implementation Plan:**
- Replace `*_config.cuf` with `config.yaml`
- Parse YAML in Python, generate Fortran parameters
- Store configs alongside trained models
- Version control for reproducibility

**Example Config:**
```yaml
dataset: cifar10
model:
  conv_filters: [32, 64, 128]
  fc_units: [512, 256]
  dropout: 0.0
training:
  epochs: 15
  batch_size: 128
  learning_rate: 0.001
  optimizer: adam
  lr_schedule:
    type: cosine
    warmup_epochs: 2
augmentation:
  enabled: true
  horizontal_flip: true
  random_crop: true
  color_jitter: true
```

**Effort:** 4-5 days
**Files to create:**
- `common/config_parser.py` - Parse YAML → Fortran
- `configs/*.yaml` - Per-dataset configs

**Files to modify:**
- `datasets/*/compile_*.sh` - Generate config from YAML

**Trade-offs:**
- ✅ Much easier to experiment with hyperparameters
- ✅ Better reproducibility
- ✅ Config versioning with git
- ✅ No recompilation needed for hyperparameter changes
- ❌ Adds Python dependency to build process
- ❌ More complex build system

---

#### 5. **Generic Training Binary** 💡
**Status:** Phase 4 vision (not started)
**Impact:** Massive UX improvement

**Implementation Plan:**
```bash
# Current workflow (requires recompilation)
cd datasets/cifar10
nvfortran cifar10_main.cuf ...  # 30+ seconds
./cifar10_train                 # 31 seconds training

# Proposed workflow (no recompilation)
./train_cnn --dataset=cifar10 --epochs=15 --lr=0.001
./train_cnn --config=configs/cifar10_experiment.yaml
```

**Architecture:**
- Single executable for all datasets
- Runtime polymorphism (Fortran 2003 abstract types)
- Plugin system for dataset loaders
- Dynamic architecture configuration

**Effort:** 2-3 weeks (major refactor)
**Files to create:**
- `src/train_generic.cuf` - Main binary
- `common/dataset_interface.cuf` - Abstract dataset type
- `common/model_builder.cuf` - Dynamic model construction

---

### Priority 3: Advanced Features

#### 6. **ONNX Export** 🌐
**Status:** Currently exports to PyTorch only
**Impact:** Deploy to any framework (TensorFlow, ONNX Runtime, etc.)

**Implementation Plan:**
- Generate ONNX graph from trained weights
- Support Conv, FC, BatchNorm, LeakyReLU, MaxPool layers
- Validate against PyTorch ONNX export
- Inference benchmarks (ONNX Runtime vs PyTorch)

**Effort:** 4-5 days
**Files to create:**
- `inference/onnx_exporter.py` - Convert weights → ONNX

**Trade-offs:**
- ✅ Framework-agnostic deployment
- ✅ Optimized inference (ONNX Runtime)
- ✅ Mobile/edge deployment ready
- ❌ ONNX has limitations for custom ops
- ❌ Additional testing burden

---

#### 7. **Mixed Precision Training (FP16)** ⚡
**Status:** Currently FP32 only
**Impact:** 2-3× speedup, 50% memory reduction

**Implementation Plan:**
- Use NVIDIA Tensor Cores (FP16 compute, FP32 accumulation)
- Loss scaling to prevent underflow
- Dynamic loss scaling for stability
- Benchmark accuracy impact

**Effort:** 1 week
**Files to create:**
- `common/mixed_precision.cuf` - FP16 utilities

**Files to modify:**
- `datasets/*/main.cuf` - FP16 training loop

**Trade-offs:**
- ✅ 2-3× faster training
- ✅ 50% less GPU memory
- ✅ Can train larger models
- ❌ Numerical instability (need loss scaling)
- ❌ Accuracy might drop slightly (usually <0.5%)
- ❌ More complex to debug

---

#### 8. **Different CNN Architectures** 🏗️
**Status:** Fixed architecture (3 conv + 3 FC)
**Impact:** Support modern architectures

**Proposed Architectures:**
- ResNet-18/34 (residual connections)
- VGG-16 (deeper stacks)
- MobileNetV2 (depthwise separable convolutions)
- EfficientNet (compound scaling)

**Effort:** 2-3 weeks per architecture
**Files to create:**
- `common/resnet_layers.cuf`
- `common/mobilenet_layers.cuf`

**Trade-offs:**
- ✅ Better accuracy (ResNet: 92%+ on CIFAR-10)
- ✅ More flexible framework
- ✅ Educational value
- ❌ Much more complex codebase
- ❌ Harder to maintain modularity
- ❌ Longer compilation times

---

## Trade-Off Analysis: Generic Binary vs Current Structure

### Current Structure (Fortran Recompilation)

**Workflow:**
```bash
cd datasets/cifar10
nvfortran cifar10_main.cuf -o cifar10_train ...
./cifar10_train
```

**Advantages:**
- ✅ **Maximum performance:** Compiler optimizations per dataset
- ✅ **Full control:** Can customize training loop per dataset
- ✅ **Memory pools:** Can experiment with custom memory layouts
- ✅ **No runtime overhead:** Everything resolved at compile time
- ✅ **Easy to experiment:** Modify `.cuf` files directly
- ✅ **Debugging:** Easier to debug dataset-specific issues

**Disadvantages:**
- ❌ **30+ second recompilation** for every hyperparameter change
- ❌ **Separate binary per dataset** (disk space, maintenance)
- ❌ **Higher barrier to entry** (need Fortran knowledge)

---

### Generic Binary (Proposed)

**Workflow:**
```bash
./train_cnn --dataset=cifar10 --epochs=15 --lr=0.001
./train_cnn --config=configs/my_experiment.yaml
```

**Advantages:**
- ✅ **Zero recompilation:** Change hyperparameters instantly
- ✅ **Better UX:** Command-line interface
- ✅ **Easy experimentation:** Just edit YAML file
- ✅ **Single binary:** Easier deployment
- ✅ **Lower barrier to entry:** No Fortran knowledge needed

**Disadvantages:**
- ❌ **Less customization:** Harder to create dataset-specific workflows
- ❌ **Runtime overhead:** Virtual function calls, dynamic dispatch
- ❌ **Complex codebase:** Abstract types, polymorphism
- ❌ **Memory pool constraints:** Harder to experiment with custom layouts
- ❌ **Potential performance loss:** ~5-10% slower (estimate)

---

### **Recommendation: Do Both!** 🎯

**Hybrid Approach:**
1. **Keep current structure for research/experimentation**
   - Researchers can still modify `.cuf` files directly
   - Full control over training loops
   - Maximum performance

2. **Add generic binary for production/ease-of-use**
   - Casual users get easy command-line interface
   - Quick hyperparameter sweeps
   - Better for community contributions

**Implementation:**
- Generic binary reuses common modules
- Datasets remain as separate implementations
- User chooses which workflow suits their needs

**Analogy:**
- Current structure = **Compiling from source** (maximum control)
- Generic binary = **Pre-built executable** (maximum convenience)

---

## Dataset Expansion Plans

### Quick Wins (< 2 hours each)
- **MNIST:** 99%+ accuracy expected (copy Fashion-MNIST, change to 1-channel)
- **KMNIST:** Japanese characters (same as MNIST)
- **EMNIST:** Extended MNIST (letters + digits)

### Medium Effort (< 1 day)
- **STL-10:** 96×96 images (needs architecture adjustment)
- **Caltech-101/256:** Variable-size images (needs resizing)

### Large Projects (> 1 week)
- **ImageNet:** 224×224, 1000 classes (needs architecture redesign)
- **COCO:** Object detection (completely different architecture)

---

## Oxford Flowers 102 Integration

### Current Situation
- Implemented in `v28_baseline` (not `v28_baseline_plus_export`)
- Dense layer only (1280 → 102) on pre-extracted MobileNetV2 features
- 78.94% accuracy, 1.2s training time
- No export functionality, no inference notebook

### Proposed Approach (Choose One)

#### **Option A: Full Migration (Recommended)**
**Effort:** 2-3 days

1. Migrate to `v28_baseline_plus_export/datasets/oxford_flowers/`
2. Create modular structure:
   - `oxford_flowers_config.cuf` - Feature loading
   - `oxford_flowers_main.cuf` - Dense layer training
   - `prepare_oxford_flowers.py` - MobileNetV2 feature extraction
3. Add export functionality (simpler than CNN - just 2 files):
   - `fc_weights.bin` - (102, 1280)
   - `fc_bias.bin` - (102)
4. Create `model_loader.py` extension for dense layers
5. Create `oxford_flowers_inference.ipynb`

**Advantages:**
- ✅ Consistent with other datasets
- ✅ Complete export pipeline
- ✅ Jupyter notebook inference
- ✅ Documentation and examples

---

#### **Option B: Minimal Adapter**
**Effort:** 1 day

1. Keep Oxford Flowers in `v28_baseline`
2. Create simple export script in existing code
3. Create standalone inference notebook
4. Add cross-reference documentation

**Advantages:**
- ✅ Faster to implement
- ✅ Less code duplication

**Disadvantages:**
- ❌ Inconsistent structure
- ❌ Not part of main framework

---

#### **Option C: Document as Future Work**
**Effort:** 1 hour

1. Add Oxford Flowers to roadmap
2. Document current status
3. Focus on more impactful features first

---

## Community Contributions

### Ways to Encourage Contributions

1. **Dataset Challenges:**
   - "Add your favorite dataset in < 2 hours using our guide!"
   - Hall of fame for contributors
   - Benchmark leaderboard

2. **Architecture Zoo:**
   - Implement ResNet, VGG, MobileNet
   - Performance comparison matrix
   - Best practices documentation

3. **Optimization Contest:**
   - Can you beat our 4× PyTorch speedup?
   - Novel GPU kernel implementations
   - Memory optimization techniques

4. **Export Targets:**
   - TensorFlow export
   - ONNX improvements
   - TensorRT integration

### Infrastructure Needed
- Contribution guidelines (CONTRIBUTING.md)
- Issue templates
- Automated testing (compilation + accuracy checks)
- Code review checklist

---

## Implementation Timeline (Proposed)

### Phase 1: Production Features (4-6 weeks)
**Week 1-2:**
- ✅ Data augmentation
- ✅ Learning rate scheduling

**Week 3-4:**
- ✅ Model checkpointing & validation
- ✅ YAML configuration

**Week 5-6:**
- ✅ Oxford Flowers integration (Option A)
- ✅ Documentation updates

### Phase 2: Advanced Features (6-8 weeks)
**Week 7-10:**
- ✅ Generic training binary
- ✅ Mixed precision (FP16)

**Week 11-14:**
- ✅ ONNX export
- ✅ Additional datasets (MNIST, STL-10)

### Phase 3: Architecture Expansion (12+ weeks)
**Week 15+:**
- ✅ ResNet implementation
- ✅ MobileNet implementation
- ✅ Architecture comparison study

---

## Success Metrics

### Technical Metrics
- **Accuracy:** Match or exceed PyTorch baselines
- **Speed:** Maintain 2-4× speedup over PyTorch
- **Memory:** < 1GB GPU memory for all current datasets
- **Compilation:** < 60s compilation time per dataset

### Community Metrics
- **Dataset Coverage:** 10+ datasets by Q2 2026
- **Contributors:** 5+ external contributors
- **GitHub Stars:** 100+ stars
- **Documentation:** 95%+ code coverage in docs

---

## Questions to Resolve

1. **Oxford Flowers:** Which option (A/B/C) should we pursue?
2. **Generic Binary:** Priority level (high/medium/low)?
3. **Architectures:** Start with ResNet or other?
4. **FP16:** Is numerical instability acceptable for 2-3× speedup?
5. **Community:** Do we want to actively encourage external contributions?

---

## Summary

**Current Status:**
- ✅ 4 datasets with complete export pipeline
- ✅ 4× faster than PyTorch
- ✅ Production-ready modular framework

**Missing Production Features:**
- ❌ Data augmentation
- ❌ Checkpointing & validation
- ❌ Learning rate scheduling
- ❌ Oxford Flowers notebook

**Proposed Next Steps:**
1. Add production-critical features (augmentation, checkpointing, LR scheduling)
2. Integrate Oxford Flowers (Option A recommended)
3. Add YAML configs for better UX
4. Evaluate generic binary (can coexist with current structure)
5. Expand dataset coverage (MNIST, STL-10)
6. Explore advanced features (FP16, ONNX, new architectures)

**Philosophy:**
- **Keep what works:** Modular structure, performance optimizations
- **Add what's missing:** Production features, better UX
- **Enable innovation:** Let users choose their workflow (Fortran vs binary)
- **Encourage community:** Make it easy to contribute new datasets/features

---

**Last Updated:** 2025-11-20
**Next Review:** After implementing Phase 1 features
