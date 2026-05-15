# Architecture & Implementation Guide

## Overview

This document provides a detailed explanation of the system architecture, design decisions, and implementation details for the skin cancer classification pipeline.

## 1. System Architecture

### 1.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                      Configuration Layer                     │
│  (YAML configs + dataclass validation)                      │
└──────────┬──────────────────────────────────────────────────┘
           │
┌──────────v──────────────────────────────────────────────────┐
│                      Data Layer                              │
│  ├── Load metadata CSV                                       │
│  ├── Resolve image paths                                     │
│  ├── 5-fold stratified CV split                             │
│  └── Create PyTorch Datasets with augmentations             │
└──────────┬──────────────────────────────────────────────────┘
           │
┌──────────v──────────────────────────────────────────────────┐
│                      Model Layer                             │
│  ├── Build model (EfficientNet, ConvNeXt, etc.)            │
│  ├── Load pretrained ImageNet weights                       │
│  └── Configure for binary classification                    │
└──────────┬──────────────────────────────────────────────────┘
           │
┌──────────v──────────────────────────────────────────────────┐
│                      Loss & Optimization Layer               │
│  ├── Build loss function (BCE, Focal, CB-Focal, etc.)       │
│  ├── Create optimizer (AdamW)                               │
│  └── Setup LR scheduler (Cosine Annealing)                  │
└──────────┬──────────────────────────────────────────────────┘
           │
┌──────────v──────────────────────────────────────────────────┐
│                      Training Layer                          │
│  ├── Trainer class with:                                     │
│  │   - Forward/backward passes                              │
│  │   - Mixed precision (AMP)                                │
│  │   - Early stopping                                        │
│  │   - Checkpoint management                                │
│  └── Per-epoch metrics computation                          │
└──────────┬──────────────────────────────────────────────────┘
           │
┌──────────v──────────────────────────────────────────────────┐
│                      Evaluation Layer                        │
│  ├── Compute metrics on val/test sets                        │
│  ├── Threshold optimization (Youden, F1, Clinical)          │
│  ├── Error analysis (FP/FN)                                 │
│  └── Grad-CAM visualization                                 │
└──────────┬──────────────────────────────────────────────────┘
           │
└──────────v──────────────────────────────────────────────────┐
            Visualization & Output Layer
            ├── ROC/PR curves
            ├── Confusion matrices
            ├── Training curves
            ├── Error analysis plots
            └── LaTeX-ready tables
```

### 1.2 Modular Design Principles

1. **Separation of Concerns**: Each module handles a specific responsibility
2. **Composability**: Modules can be combined in different ways
3. **Extensibility**: Easy to add new models, losses, or methods
4. **Reusability**: Components are reusable across different experiments
5. **Testability**: Each module can be tested independently

## 2. Module Details

### 2.1 Configuration System (src/configs/)

**Files:**
- `config.py`: Dataclass-based configuration

**Key Features:**
- Hierarchical configuration with nested dataclasses
- YAML serialization/deserialization
- Type hints for all fields
- Automatic path resolution

**Usage:**

```python
from src.configs import Config

# Load from YAML
config = Config.from_yaml(Path("configs/config.yaml"))

# Or create programmatically
config = Config()
config.model.model_name = "efficientnet_b0"
config.training.num_epochs = 20

# Save back to YAML
config.to_yaml(Path("outputs/config_used.yaml"))
```

**Why Dataclasses?**
- Type safety
- IDE autocomplete support
- Easy validation
- Clean syntax
- Automatic `__init__` and `__repr__`

### 2.2 Dataset Module (src/datasets/)

**Files:**
- `dataset.py`: PyTorch Dataset class
- `cv_split.py`: Cross-validation utilities
- `imbalance_methods.py`: SMOTE, Mixup, CutMix

**Key Classes:**

```python
class SkinLesionDataset(Dataset):
    """Standard PyTorch Dataset implementation"""
    - Lazy image loading
    - Support for custom transforms
    - Returns dict with image, target, image_path

class MixupCollator:
    """Collate function for Mixup augmentation"""
    - Applied at batch level
    - Interpolates between samples
    
class CutMixCollator:
    """Collate function for CutMix augmentation"""
    - Swaps image regions between samples
    - Adjusts targets proportionally
```

**Cross-Validation:**

```python
folds = stratified_k_fold_split(df, n_splits=5)
# Returns: List[(train_df, val_df), ...]
# Each fold has balanced class distribution
```

**Why this design?**
- Lazy loading: Memory efficient
- Dict-based returns: Clear and extensible
- Stratified splits: Ensures balanced folds
- Separate collators: Clean separation of augmentation logic

### 2.3 Model Module (src/models/)

**Files:**
- `model_factory.py`: Model creation with timm library

**Supported Models:**

```
- EfficientNet: efficientnet_b0, efficientnet_b3
- ConvNeXt: convnext_tiny
- Vision Transformer: vit_tiny_patch16_224
- Swin Transformer: swin_tiny_patch4_window7_224
```

**Key Design:**

```python
def build_model(model_name: str, pretrained: bool = True, num_classes: int = 1):
    """Factory pattern for model creation"""
    - Maps friendly names to timm identifiers
    - Handles pretrained weight loading
    - Forces num_classes=1 for binary classification
    - Single logit output (sigmoid applied during inference)
```

**Why timm?**
- 1000+ pre-trained models
- Consistent API
- Active community
- Easy to add new models
- Well-tested implementations

### 2.4 Loss Functions Module (src/losses/)

**Implemented Losses:**

1. **BCE (Binary Cross-Entropy)**
   - `nn.BCEWithLogitsLoss()`
   - Simple baseline
   
2. **Weighted BCE**
   - Adjusts positive class weight
   - Formula: `pos_weight = num_negative / num_positive`
   - Emphasizes minority class
   
3. **Focal Loss**
   - Reduces weight of easy negatives
   - Parameters: `alpha` (class balance), `gamma` (focus)
   - Good for imbalance but can be unstable
   
4. **Class-Balanced Focal Loss**
   - Combines focal loss with effective sample re-weighting
   - Formula: `weight = (1 - beta) / (1 - beta^N)`
   - Most stable for severe imbalance

**Mathematical Details:**

```
BCE: L = -[y*log(p) + (1-y)*log(1-p)]

Weighted BCE: L = -pos_weight * y * log(p) + (1-y) * log(1-p)

Focal Loss: L = -alpha * (1-pt)^gamma * log(pt)
  where pt = p if y=1 else (1-p)

CB-Focal: Combines both with effective number weighting
```

**Factory Pattern:**

```python
def build_loss(loss_type, class_counts, config):
    """Factory for loss creation"""
    - Computes class counts from training data
    - Auto-calculates weights/parameters
    - Returns ready-to-use loss module
```

**Why separate modules?**
- Clear separation of concerns
- Easy to add new losses
- Reusable across experiments
- Well-tested implementations

### 2.5 Trainer Module (src/trainers/)

**Core Components:**

```python
class Trainer:
    def train_one_epoch(self):
        """Training loop with AMP support"""
        
    def validate(self):
        """Validation without gradients"""
        
    def should_stop_early(self, metrics):
        """Early stopping logic"""
        
    def save_checkpoint(self, epoch, metrics):
        """Checkpoint management"""
        
    def train(self):
        """Full training orchestration"""
```

**Key Features:**

1. **Mixed Precision (AMP)**
   - Reduces memory by ~50%
   - Speeds up training
   - Maintains accuracy with proper scaling

2. **Early Stopping**
   - Monitors validation metric (AUPRC preferred)
   - Saves best checkpoint automatically
   - Prevents overfitting

3. **Gradient Accumulation**
   - Simulates larger batch sizes
   - Useful for limited memory

4. **Gradient Clipping**
   - Prevents exploding gradients
   - Configurable max norm

5. **Learning Rate Scheduling**
   - Cosine annealing (smooth decay)
   - Warmup support
   - Integrates with PyTorch schedulers

**Training Loop Flow:**

```
For each epoch:
  1. Forward pass through batch
  2. Compute loss
  3. Backward pass (with AMP scaling)
  4. Gradient clipping
  5. Optimizer step
  6. Scheduler step
  
  Validate and check early stopping
  Log metrics
  Save checkpoint if improved
```

**Why AUPRC for Early Stopping?**
- ROC-AUC is too lenient for imbalanced data
- F1 can be noisy at low thresholds
- AUPRC directly reflects PR performance
- More clinically relevant

### 2.6 Evaluation Module (src/evaluators/)

**Functions:**

```python
def evaluate_epoch(model, data_loader, criterion, device, use_amp=True):
    """Evaluate model on full epoch"""
    - Returns loss, metrics, predictions
    - Handles AMP
    - No gradients
```

**Why separate module?**
- Reusable for val/test evaluation
- Consistent evaluation across scripts
- Single source of truth for metrics

### 2.7 Visualization Module (src/visualization/)

**Components:**

1. **Plotting** (`plotting.py`)
   - ROC curves
   - Precision-Recall curves
   - Confusion matrices
   - Training curves
   - Combined comparison plots

2. **Threshold Optimization** (`threshold_optimization.py`)
   - Youden index: `max(TPR + TNR - 1)`
   - F1-based: `max F1 score`
   - Clinical: `min(FN_cost × FN + FP)`
   - Generates comprehensive reports

3. **Error Analysis** (`error_analysis.py`)
   - Identifies false positives/negatives
   - Ranks by confidence
   - Saves to CSV for manual review

4. **Grad-CAM** (`gradcam.py`)
   - Generates attention maps
   - Denormalizes images for visualization
   - Saves overlay images
   - Supports custom thresholds

**Design Rationale:**

```
Separate files because:
- Each handles different visualization type
- Can be imported independently
- Easy to extend with new visualizations
- Clean namespace
```

### 2.8 Utils Module (src/utils/)

**Components:**

1. **Reproducibility** (`reproducibility.py`)
   ```python
   seed_everything(seed=42, cudnn_deterministic=True)
   - Sets seeds across all libraries
   - Ensures reproducible results
   - Configurable determinism (speed vs reproducibility)
   ```

2. **Logger** (`logger.py`)
   ```python
   logger = setup_logger(name, log_file=path)
   - Console + file logging
   - Consistent format
   - Timestamp included
   ```

3. **Metrics** (`metrics.py`)
   ```python
   def compute_binary_metrics(y_true, y_prob, threshold):
       """Comprehensive metrics computation"""
       - ROC-AUC, AUPRC
       - Sensitivity, Specificity, Precision, Recall
       - F1, Accuracy
       - Confusion matrix values
   
   def find_optimal_threshold(y_true, y_prob, method):
       """Find optimal classification threshold"""
       - Supports multiple methods
       - Returns threshold and score
   ```

## 3. Workflow Diagrams

### 3.1 Training Workflow

```
1. Load Config
   ↓
2. Setup Reproducibility (seed, device)
   ↓
3. Load Dataset
   ├─ Read CSV metadata
   ├─ Resolve image paths
   ├─ Clean targets (binary 0/1)
   └─ Validate: len(images) == len(targets)
   ↓
4. Create Folds (5-fold stratified CV)
   │
   ├─→ Fold 0:
   │   ├─ Split: train, val, test
   │   ├─ Build Model → Load pretrained
   │   ├─ Build Loss → Compute class weights
   │   ├─ Build Optimizer → AdamW
   │   ├─ Build Scheduler → CosineAnnealing
   │   ├─ Instantiate Trainer
   │   │
   │   ├─→ Training Loop (up to N epochs):
   │   │   ├─ Train epoch
   │   │   ├─ Validate epoch
   │   │   ├─ Compute metrics
   │   │   ├─ Check early stopping
   │   │   ├─ Log metrics
   │   │   └─ Save checkpoint if improved
   │   │
   │   ├─ Load best checkpoint
   │   ├─ Evaluate on test set
   │   ├─ Generate visualizations (ROC, PR, CM, curves)
   │   ├─ Threshold optimization
   │   ├─ Error analysis
   │   ├─ Save results to fold directory
   │   └─ Append fold results to summary
   │
   ├─→ Fold 1:
   │   └─ ... repeat ...
   │
   └─→ ... Fold 4
   ↓
5. Aggregate Results
   ├─ Compute mean ± std across folds
   ├─ Save CV summary metrics
   ├─ Create comparison plots
   └─ Print final report
```

### 3.2 Evaluation Workflow

```
1. Load checkpoint
   ↓
2. Recreate model architecture
   ↓
3. Load model weights
   ↓
4. Prepare test dataset
   ↓
5. Run inference
   ├─ Batch forward passes
   ├─ Collect predictions
   ├─ Compute metrics
   └─ Save predictions to CSV
   ↓
6. Generate visualizations
   ├─ ROC curve
   ├─ PR curve
   ├─ Confusion matrix
   └─ Threshold report
   ↓
7. Grad-CAM (if enabled)
   ├─ Select samples
   └─ Generate and save visualizations
   ↓
8. Error analysis (if enabled)
   ├─ Find FP/FN samples
   ├─ Rank by confidence
   └─ Save to CSV
```

## 4. Key Design Decisions

### 4.1 Why Single Logit Output?

```python
# Instead of: output shape (B, 2)
# We use: output shape (B, 1)

model_output = (B, 1)  # Single logit
prob = sigmoid(logit)  # ∈ [0, 1]
pred = prob >= threshold  # Binary decision
```

**Benefits:**
- More memory efficient
- Matches binary cross-entropy loss
- Natural for binary classification
- Simpler to interpret

### 4.2 Why YAML Configs?

```yaml
model:
  model_name: efficientnet_b0
  pretrained: true
training:
  batch_size: 32
  num_epochs: 20
```

**Alternatives Considered:**
- JSON: No comments, verbose
- TOML: Good but less popular
- Python files: Hard to validate

**Advantages:**
- Human readable
- Supports comments
- Easy to version control
- Integrates with dataclasses

### 4.3 Why 5-Fold Cross-Validation?

**Instead of single train/val/test split:**

- Reduces variance of performance estimates
- Better utilization of limited data
- Provides uncertainty (mean ± std)
- Standard in ML research
- Enables robust statistical testing

**Why 5 folds?**
- 10-fold often overkill
- 3-fold too coarse
- 5-fold is computational sweet spot
- Industry standard

### 4.4 Why Early Stopping on AUPRC?

**Metric Comparison for Imbalanced Data:**

```
Metric        | Advantages            | Disadvantages
──────────────┼──────────────────────┼─────────────────────────
ROC-AUC       | Threshold-independent | Insensitive to imbalance
F1            | Interpretable         | Noisy, varies with threshold
AUPRC         | ↑ Sensitive to FN     | Correlated with ROC-AUC
Sensitivity   | Clinical relevance    | Doesn't consider specificity
```

**AUPRC is best because:**
- Emphasizes minority class
- Reflects PR performance
- Clinically relevant (FN cost)
- Correlates with F1

### 4.5 Why Stratified K-Fold?

```
Regular K-fold:
Fold 0: [0,0,0,0,0,1,1,1,1,1]  # 50% pos, 50% neg
Fold 1: [0,0,0,0,0,1,1,1,1,1]  # 50% pos, 50% neg
...
Issue: May not reflect true distribution

Stratified K-fold:
Data: [0,0,0,0,0,0,0,0,0,1]  # 90% neg, 10% pos
Fold 0: [0,0,0,0,0,0,0,0,0,1]  # Same distribution!
Fold 1: [0,0,0,0,0,0,0,0,0,1]  # Same distribution!
```

**Why it matters:**
- Ensures representative folds
- Better metrics estimates
- Especially important for imbalanced data

## 5. Data Flow Examples

### 5.1 Batch Processing

```
Input: Batch of 32 images
       ├─ Shape: (32, 3, 224, 224)
       ├─ Dtype: float32
       ├─ Values: [-∞, +∞] (ImageNet normalized)
       ├─ Targets: (32,) with values {0, 1}
       └─ Paths: list[32]

Model Forward:
  ConvNeXt/EfficientNet
  → Feature extraction
  → Global average pooling
  → Classification head (1 output)
  → Output: (32, 1) logits

Loss Computation:
  Logits: (32, 1)
  Targets: (32, 1)
  Loss = BCEWithLogitsLoss(logits, targets)
  → Scalar loss value

Metrics:
  Probs = sigmoid(logits) → (32,) ∈ [0, 1]
  Preds = (probs >= 0.5).int() → (32,) ∈ {0, 1}
  ROC-AUC, F1, Sensitivity, etc.
```

### 5.2 Batch Augmentation with Mixup

```
Input Batch:
  images_a: (32, 3, 224, 224)
  targets_a: (32,)

Mixup:
  λ ~ Beta(α, α)  # Sample mixing coefficient
  
  For each sample i:
    idx = random permutation
    images_mixed[i] = λ * images_a[i] + (1-λ) * images_a[idx]
    targets_mixed[i] = λ * targets_a[i] + (1-λ) * targets_a[idx]

Output Batch:
  images: (32, 3, 224, 224)  # Mixed
  targets: (32,)  # Soft targets ∈ [0, 1]

Loss:
  Targets are now soft (between 0 and 1)
  BCE loss naturally handles this
  Encourages smoother decision boundaries
```

## 6. Performance Optimization

### 6.1 Memory Optimization

**Techniques Used:**
1. **Automatic Mixed Precision (AMP)**
   - FP32 for most ops, FP16 for others
   - ~50% memory savings
   - No accuracy loss

2. **Lazy Image Loading**
   - Load image only on `__getitem__`
   - Not all images in memory at once

3. **Single Logit Output**
   - (B, 1) instead of (B, 2)
   - Saves some GPU memory

4. **Gradient Accumulation** (optional)
   - Simulate larger batch with smaller batches
   - Useful for very large models

### 6.2 Speed Optimization

**Techniques Used:**
1. **Data Loader Parallelization**
   - `num_workers > 0`
   - Background loading while training

2. **Pin Memory**
   - `pin_memory=True`
   - Speeds up CPU→GPU transfer

3. **CudNN Optimization**
   - `torch.backends.cudnn.benchmark = True`
   - Auto-tunes algorithms

4. **Gradient Checkpointing** (optional)
   - Trade compute for memory
   - Useful for large models

## 7. Error Handling & Validation

### 7.1 Input Validation

```python
# In load_metadata():
✓ Check CSV file exists
✓ Check image directory exists
✓ Verify target column exists
✓ Validate binary targets (0 or 1)
✓ Ensure image paths resolve
✓ Drop samples with missing data

# In stratified_k_fold_split():
✓ Verify stratification is possible (both classes present)
✓ Check n_splits is valid
```

### 7.2 Error Propagation

```python
# Training loop:
try:
    fold_result = train_fold(...)
except Exception as e:
    logger.error(f"Fold {i} failed: {e}")
    if config.continue_on_error:
        continue
    else:
        raise
```

## 8. Extension Points

### 8.1 Adding New Models

```python
# 1. Edit src/models/model_factory.py
MODEL_NAME_MAP = {
    ...
    "my_model": "my_model_in_timm",  # Add this
}

# 2. Update config
config.model.model_name = "my_model"

# 3. Run training (rest is automatic)
```

### 8.2 Adding New Loss Functions

```python
# 1. Implement in src/losses/loss_factory.py
class MyLoss(nn.Module):
    def forward(self, logits, targets):
        ...

# 2. Add to build_loss()
if loss_type == "my_loss":
    return MyLoss(...)

# 3. Update config
config.loss.loss_type = "my_loss"
```

### 8.3 Adding New Metrics

```python
# 1. Implement in src/utils/metrics.py
def compute_my_metric(y_true, y_prob):
    ...

# 2. Call in trainer/evaluator
metrics = compute_binary_metrics(...)
metrics["my_metric"] = compute_my_metric(...)

# 3. Log and visualize
log_df["my_metric"] = metrics["my_metric"]
```

## 9. Testing Checklist

- [ ] Config loading/saving
- [ ] Dataset loading and augmentation
- [ ] Model building and parameter counting
- [ ] Loss function computation
- [ ] Training loop (one epoch)
- [ ] Validation loop
- [ ] Metrics computation
- [ ] Checkpoint saving/loading
- [ ] Threshold optimization
- [ ] Visualization generation
- [ ] Cross-validation split creation
- [ ] Error analysis

## 10. Deployment Considerations

### 10.1 Inference Script

```python
# Load checkpoint
checkpoint = torch.load(ckpt_path)
model = build_model(config.model.model_name, pretrained=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Prepare image
image = Image.open(image_path)
input_tensor = test_transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    logit = model(input_tensor)
    prob = torch.sigmoid(logit).item()
    pred = 1 if prob >= threshold else 0
```

### 10.2 Production Deployment

**Checklist:**
- [ ] Model quantization (float32 → float16 or int8)
- [ ] Model compression (pruning, distillation)
- [ ] Batch inference optimization
- [ ] ONNX export for cross-platform compatibility
- [ ] Monitoring and logging
- [ ] Confidence thresholding
- [ ] Uncertainty quantification
- [ ] Fairness auditing

---

**Document Version**: 1.0  
**Last Updated**: 2024
"""

with open("/home/22010759/Minhk16/como/ARCHITECTURE.md", "w") as f:
    f.write(readme_content)

print("Architecture documentation written!")
