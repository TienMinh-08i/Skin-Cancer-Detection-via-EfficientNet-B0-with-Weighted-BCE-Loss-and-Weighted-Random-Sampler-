"""Comprehensive README file for the project."""
readme_content = """
# Skin Cancer Classification with Class Imbalance Handling

This is a research-grade implementation of binary skin lesion classification (benign vs malignant) with comprehensive handling of class imbalance using multiple techniques and paper-ready evaluation metrics.

## Project Overview

**Key Features:**

- **Binary Classification**: Benign (0) vs Malignant (1)
- **Cross-Validation**: 5-fold stratified cross-validation for robust evaluation
- **Multiple Models**: EfficientNet-B0, ConvNeXt-Tiny (extensible to ViT, Swin Transformers)
- **Loss Functions**: BCE, Weighted BCE, Focal Loss, Class-Balanced Focal Loss
- **Imbalance Methods**: WeightedRandomSampler, SMOTE, Mixup, CutMix
- **Threshold Optimization**: Youden Index, F1-based, Clinical Cost-sensitive
- **Comprehensive Evaluation**: ROC-AUC, AUPRC, F1, Sensitivity, Specificity, Precision, Recall
- **Interpretability**: Grad-CAM visualization with quantitative metrics
- **Error Analysis**: Automatic false positive/negative analysis
- **Paper-Ready Outputs**: Publication-quality figures, tables, and metrics

## Project Structure

```
.
├── src/
│   ├── configs/              # Configuration system
│   │   ├── config.py
│   │   └── __init__.py
│   ├── datasets/             # Data loading and preprocessing
│   │   ├── dataset.py        # PyTorch Dataset class
│   │   ├── cv_split.py       # Cross-validation splits
│   │   ├── imbalance_methods.py  # SMOTE, Mixup, CutMix
│   │   └── __init__.py
│   ├── models/               # Model architectures
│   │   ├── model_factory.py  # Model creation (EfficientNet, ConvNeXt, etc.)
│   │   └── __init__.py
│   ├── losses/               # Loss functions
│   │   ├── loss_factory.py   # BCE, Weighted BCE, Focal, CB-Focal
│   │   └── __init__.py
│   ├── trainers/             # Training loop
│   │   ├── trainer.py        # Trainer class with early stopping
│   │   └── __init__.py
│   ├── evaluators/           # Evaluation utilities
│   │   ├── evaluator.py      # Evaluation epoch
│   │   └── __init__.py
│   ├── visualization/        # Plotting and analysis
│   │   ├── plotting.py       # ROC, PR, confusion matrix, training curves
│   │   ├── threshold_optimization.py  # Threshold search
│   │   ├── error_analysis.py # False positive/negative analysis
│   │   ├── gradcam.py        # Grad-CAM visualization
│   │   └── __init__.py
│   └── utils/                # Utilities
│       ├── reproducibility.py    # Seed everything
│       ├── logger.py             # Logging setup
│       ├── metrics.py            # Metrics computation
│       └── __init__.py
├── scripts/
│   ├── main_train.py         # Main training script (5-fold CV)
│   ├── main_eval.py          # Evaluation script (test set)
│   └── generate_config.py    # Generate default config
├── configs/
│   └── config_isic2019.yaml  # Example configuration
├── outputs/                  # Generated outputs
│   ├── checkpoints/          # Model checkpoints
│   ├── logs/                 # Training logs
│   ├── figures/              # Paper-ready figures
│   └── results/              # CSV results, predictions, error analysis
└── README.md                 # This file
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.13+
- torchvision
- scikit-learn
- pandas
- numpy
- matplotlib
- PyYAML
- timm (for model zoo)
- optionally: grad-cam, imbalanced-learn

### Setup

```bash
# Clone or navigate to project directory
cd skin-cancer-classification

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### ISIC 2019 Dataset

The code expects:

1. **Metadata CSV** with columns:
   - Image identifier (e.g., `image_name`, `image_id`)
   - Target column: binary labels (0 = benign, 1 = malignant)

2. **Image Directory** containing image files (JPG, PNG)

### Example Directory Structure

```
dataset/
├── train-metadata.csv
└── train-image/
    ├── image_001.jpg
    ├── image_002.jpg
    └── ...
```

## Configuration

All hyperparameters are managed via YAML config files. See `configs/config_isic2019.yaml` for a complete example.

### Key Configuration Options

**Dataset:**
- `cv_folds`: Number of folds for cross-validation (default: 5)
- `image_size`: Image size for resizing (default: 224)
- `max_samples`: Maximum samples for debugging (default: None)

**Model:**
- `model_name`: "efficientnet_b0", "convnext_tiny", "vit_tiny", "swin_tiny"
- `pretrained`: Use ImageNet pretrained weights

**Loss Function:**
- `loss_type`: "bce", "weighted_bce", "focal", "cb_focal"
- `focal_alpha`, `focal_gamma`: Focal loss parameters
- `cb_focal_beta`: Class-balanced focal loss parameter

**Imbalance Handling:**
- `weighted_sampler_enabled`: Use WeightedRandomSampler
- `mixup_enabled`: Apply Mixup augmentation
- `cutmix_enabled`: Apply CutMix augmentation
- `smote_enabled`: Apply SMOTE oversampling

**Training:**
- `batch_size`: Batch size (default: 32)
- `num_epochs`: Maximum epochs (default: 20)
- `early_stopping_patience`: Epochs without improvement before stopping
- `early_stopping_metric`: "auprc" (recommended), "roc_auc", or "f1"
- `use_amp`: Automatic mixed precision

**Evaluation:**
- `threshold`: Default classification threshold (default: 0.5)
- `compute_gradcam`: Generate Grad-CAM visualizations
- `compute_error_analysis`: Analyze misclassifications

**Threshold Optimization:**
- `methods`: ["youden", "f1", "clinical"]
- `clinical_fn_cost_ratio`: Cost ratio FN:FP for clinical threshold

## Training

### 5-Fold Cross-Validation

Train a model with 5-fold cross-validation:

```bash
python scripts/main_train.py \\
  --config configs/config_isic2019.yaml \\
  --device cuda
```

**Output Structure:**

```
outputs/
├── cv_results/
│   ├── fold_0/
│   │   ├── checkpoints/
│   │   │   └── fold_0_best.pt
│   │   ├── logs/
│   │   │   └── fold_0.log
│   │   │   └── fold_0_training_log.csv
│   │   ├── figures/
│   │   │   ├── roc_curve.png
│   │   │   ├── pr_curve.png
│   │   │   ├── confusion_matrix.png
│   │   │   ├── training_curves.png
│   │   │   └── threshold_report.csv
│   │   ├── error_analysis/
│   │   │   ├── false_positives.csv
│   │   │   └── false_negatives.csv
│   │   └── test_predictions.csv
│   ├── fold_1/
│   ├── ...
│   ├── cv_results_summary.csv      # Metrics for all folds
│   └── cv_summary_metrics.csv      # Mean ± std across folds
├── checkpoints/
├── logs/
├── figures/
└── results/
```

### Key Outputs

#### CSV Results

**cv_results_summary.csv** - Fold-wise metrics:

```
fold,test_loss,best_epoch,roc_auc,auprc,f1,sensitivity,specificity,...
0,0.1234,15,0.8956,0.8234,0.7892,0.8123,0.8456,...
1,0.1156,18,0.9012,0.8445,0.8034,0.8234,0.8567,...
...
```

**cv_summary_metrics.csv** - Aggregated metrics:

```
roc_auc,auprc,f1,sensitivity,specificity,accuracy
0.8923 ± 0.0145,0.8342 ± 0.0187,0.7945 ± 0.0156,...
```

#### Figures

- **roc_curve.png**: ROC curves per fold and combined
- **pr_curve.png**: Precision-Recall curves
- **confusion_matrix.png**: Confusion matrices
- **training_curves.png**: Loss and metric curves during training
- **gradcam/*.png**: Grad-CAM visualizations

## Evaluation

### Evaluate on Test Set

```bash
python scripts/main_eval.py \\
  --checkpoint outputs/cv_results/fold_0/checkpoints/fold_0_best.pt \\
  --test-csv outputs/cv_results/fold_0/test_predictions.csv \\
  --config configs/config_isic2019.yaml \\
  --output-dir outputs/eval_results \\
  --device cuda
```

### Threshold Optimization

Automatically generates reports for multiple thresholds:

**threshold_report.csv:**

```
threshold,sensitivity,specificity,precision,recall,f1,accuracy,roc_auc,auprc,tp,tn,fp,fn
0.40,0.92,0.75,0.60,0.92,0.73,0.78,...
0.50,0.85,0.82,0.68,0.85,0.76,0.83,...
0.60,0.78,0.88,0.75,0.78,0.76,0.84,...
```

### Error Analysis

Identifies and saves misclassifications:

**error_analysis/false_positives.csv:**

```
image_path,true_label,pred_label,confidence,error_type
/path/to/img_001.jpg,0,1,0.78,FP
/path/to/img_002.jpg,0,1,0.82,FP
...
```

### Grad-CAM Visualization

Generates interpretable heatmaps:

```
gradcam/
├── gradcam_000_true0_prob0.123.png
├── gradcam_001_true1_prob0.956.png
└── ...
```

## Metrics Explanation

### Binary Classification Metrics

- **ROC-AUC**: Area Under Receiver Operating Characteristic Curve
  - Range: [0, 1], higher is better
  - Threshold-independent
  
- **AUPRC**: Area Under Precision-Recall Curve
  - Range: [0, 1], higher is better
  - More appropriate for imbalanced datasets
  
- **Sensitivity (Recall)**: TP / (TP + FN)
  - Clinically important: ability to detect malignancy
  
- **Specificity**: TN / (TN + FP)
  - Ability to correctly identify benign cases
  
- **Precision**: TP / (TP + FP)
  - When predicting malignant, how often correct
  
- **F1 Score**: 2 * (Precision * Recall) / (Precision + Recall)
  - Harmonic mean of precision and recall
  
- **Accuracy**: (TP + TN) / Total
  - Overall correctness

### Imbalance-Aware Methods

**Weighted BCE**:
- Adjusts positive class weight: `pos_weight = num_negative / num_positive`
- Emphasizes minority class losses

**Focal Loss**:
- Reduces weight of easy negatives
- Parameters: `alpha` (class weighting), `gamma` (focus strength)

**Class-Balanced Focal Loss**:
- Combines focal loss with effective number of samples re-weighting
- Parameter: `beta` (typically 0.9999)

## Model Architectures

### EfficientNet-B0
- Pre-trained ImageNet weights available
- 5.3M parameters
- Efficient and accurate
- Recommended for production

### ConvNeXt-Tiny
- Modern architecture with depthwise separable convolutions
- 28M parameters
- Often outperforms EfficientNet
- Good balance of performance and efficiency

### Extensible Models
Code easily supports:
- **Vision Transformer (ViT)**: `model_name: vit_tiny`
- **Swin Transformer**: `model_name: swin_tiny`
- Custom models via timm library

## Hyperparameter Search

### Grid Search Script (Optional)

Create `scripts/grid_search.py` for systematic hyperparameter exploration:

```python
import itertools
from pathlib import Path
from src.configs import Config

loss_configs = ["bce", "weighted_bce", "focal", "cb_focal"]
model_configs = ["efficientnet_b0", "convnext_tiny"]
imbalance_configs = [
    {"weighted_sampler": True, "mixup": False},
    {"weighted_sampler": False, "mixup": True},
    {"weighted_sampler": True, "mixup": True},
]

for loss, model, imb in itertools.product(loss_configs, model_configs, imbalance_configs):
    config = Config()
    config.loss.loss_type = loss
    config.model.model_name = model
    # ... update imbalance settings ...
    config.to_yaml(Path(f"configs/grid_{loss}_{model}.yaml"))
    # Run training
```

## Reproducibility

**Seed Management:**

```python
from src.utils import seed_everything
seed_everything(seed=42, cudnn_deterministic=True)
```

- Ensures reproducibility across all libraries
- `cudnn_deterministic=True`: Reproducible but slightly slower
- `cudnn_deterministic=False`: Faster but non-deterministic (default for training)

**Recommendations for Paper:**

1. Set seed before all experiments
2. Report random seed in methods section
3. Use same seed across all folds
4. Save config files for reproducibility

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
- Reduce batch size: `batch_size: 16` or `8`
- Disable mixed precision: `use_amp: false`
- Reduce image size: `image_size: 192` or `160`

### Slow Training

**Solutions:**
- Increase batch size (if memory allows)
- Reduce number of workers: `num_workers: 2`
- Use `use_amp: true` for faster computation
- Consider using EfficientNet-B0 instead of larger models

### Poor Class Imbalance Handling

**Solutions:**
- Use `weighted_bce` or `cb_focal` loss
- Enable `weighted_sampler`
- Try combining with Mixup or CutMix
- Increase `early_stopping_patience` for more epochs

### GPU Not Detected

**Troubleshooting:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Paper Usage Guide

### Main Results Table

Use `cv_summary_metrics.csv` for the main results table:

**LaTeX Example:**

```latex
\\begin{table}[h]
\\centering
\\caption{5-Fold Cross-Validation Results (mean ± std)}
\\begin{tabular}{lcccccc}
\\toprule
Model & ROC-AUC & AUPRC & Sensitivity & Specificity & F1 & Accuracy \\\\
\\midrule
EfficientNet-B0 & 0.8923 ± 0.0145 & ... & ... & ... & ... & ... \\\\
ConvNeXt-Tiny & 0.9012 ± 0.0134 & ... & ... & ... & ... & ... \\\\
\\bottomrule
\\end{tabular}
\\end{table}
```

### Figures for Publication

**ROC Curves:**
- Use `combined_test_roc.png` for model comparison
- Include AUC values in figure caption

**Confusion Matrices:**
- Include for best-performing model
- Report TP, TN, FP, FN in text or caption

**Training Curves:**
- Show convergence and early stopping
- Include in supplementary if space limited

**Error Analysis:**
- Show representative false positives/negatives
- Use for discussion section

### Methods Section Template

```
We trained binary classifiers on the ISIC 2019 dataset using:

- Dataset: [number] benign and [number] malignant lesions
- Split: 5-fold stratified cross-validation
- Models: EfficientNet-B0 and ConvNeXt-Tiny
- Training loss: Class-balanced focal loss
- Imbalance handling: Weighted random sampling + mixup
- Early stopping: Validation AUPRC with patience=[value]
- Threshold optimization: Youden index + clinical cost-sensitivity
- Hyperparameters: [learning rate, batch size, etc. from config]

All experiments were conducted with seed=42 for reproducibility.
```

### Results Section Template

```
**Cross-Validation Results:**

Across 5 folds, [Model] achieved:
- ROC-AUC: [mean] ± [std]
- AUPRC: [mean] ± [std]
- Sensitivity: [mean] ± [std]
- Specificity: [mean] ± [std]
- F1 Score: [mean] ± [std]

**Threshold Optimization:**

Clinical threshold optimization using cost-sensitive approach
(FN cost = 2× FP cost) resulted in optimal threshold = [value]
with sensitivity = [value] and specificity = [value].

**Interpretability:**

Grad-CAM analysis (n=[number] samples) showed that the model
primarily focused on [describe features/regions].
```

## Contributing & Extensions

### Adding New Models

1. Add model to `MODEL_NAME_MAP` in `src/models/model_factory.py`
2. Ensure model has 1 output for binary classification
3. Models from `timm` are automatically supported

```python
MODEL_NAME_MAP = {
    "efficientnet_b0": "efficientnet_b0",
    "convnext_tiny": "convnext_tiny",
    "vit_tiny": "vit_tiny_patch16_224",  # New
    "my_custom_model": "my_custom_model_name_in_timm",  # New
}
```

### Adding New Loss Functions

1. Create loss class in `src/losses/loss_factory.py`
2. Add to `build_loss` function
3. Update config: `loss.loss_type: my_new_loss`

### Adding New Metrics

1. Add computation to `src/utils/metrics.py`
2. Update trainer logging in `src/trainers/trainer.py`
3. Update visualization code as needed

## Citation

If you use this code in your research, please cite:

```bibtex
@article{skin_cancer_classification_2024,
  title={Deep Learning for Binary Skin Cancer Classification with Class Imbalance Handling},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

[Specify license: MIT, Apache 2.0, etc.]

## Contact

For questions or issues, please open a GitHub issue or contact [email].

## References

- EfficientNet: Tan & Le (2019)
- ConvNeXt: Liu et al. (2022)
- Focal Loss: Lin et al. (2017)
- Class-Balanced Loss: Cui et al. (2019)
- Grad-CAM: Selvaraju et al. (2017)

---

**Last Updated**: 2024
**Version**: 1.0
"""

# Write to file
with open("README_FULL.md", "w") as f:
    f.write(readme_content)

print("README written successfully!")
