# Experiment Management System - Complete Guide

## Overview

This system automates the execution of 15 comprehensive ablation study experiments for skin cancer classification. It manages configuration generation, experiment execution, result collection, and automatic LaTeX table generation for papers.

## Architecture

### 1. Configuration Hierarchy

```
configs/
├── config_hpc.yaml          (Base config template)
├── experiments.yaml         (Experiment definitions)
└── experiments/
    ├── config_E1.yaml       (Generated: EfficientNet-B0 + BCE)
    ├── config_E2.yaml       (Generated: EfficientNet-B0 + Weighted BCE)
    ├── ...
    ├── config_E15.yaml      (Generated: Swin-Tiny + CB-Focal)
    └── experiment_index.yaml (Auto-generated index)
```

### 2. Configuration Inheritance

```
config_hpc.yaml (Base)
    ↓
experiments.yaml (Experiment definitions + defaults)
    ↓
generate_experiments.py (Merge & Generate)
    ↓
configs/experiments/config_E*.yaml (Final configs)
```

**How it works:**
1. `config_hpc.yaml` contains all default hyperparameters
2. `experiments.yaml` defines which parameters differ for each experiment
3. `generate_experiments.py` merges these to create experiment-specific configs
4. `run_all_experiments.py` executes each generated config

## Experiments Design (E1-E15)

### Models
- **EfficientNet-B0** (E1-E5): Lightweight, efficient
- **ConvNeXt-Tiny** (E6-E13): Modern CNN architecture
- **Vision Transformer** (E14): Transformer-based
- **Swin Transformer** (E15): Hierarchical transformer

### Loss Functions
- **BCE**: Binary Cross-Entropy (baseline)
- **Weighted BCE**: BCE with class weights
- **Focal Loss**: Down-weights easy examples
- **CB-Focal**: Class-balanced focal loss

### Imbalance Handling Techniques
- **WeightedRandomSampler**: Oversamples minority class
- **SMOTE**: Synthetic minority oversampling
- **Mixup**: Linear interpolation of samples
- **CutMix**: Random region mixing

## Quick Start

### 1. Generate All Experiment Configs

```bash
cd /home/22010759/Minhk16/como
python scripts/generate_experiments.py \
    --base-config configs/config_hpc.yaml \
    --output-dir configs/experiments
```

This creates 15 config files: `config_E1.yaml` through `config_E15.yaml`

### 2. Run All Experiments

```bash
# Run all experiments with resume capability
bash scripts/run_all.sh

# Or manually
python scripts/run_all_experiments.py --device cuda --resume

# Run with force re-run (ignore completed)
python scripts/run_all_experiments.py --device cuda --force
```

### 3. Run Specific Experiments

```bash
# Single experiment
python scripts/run_all_experiments.py --device cuda --experiments E1

# Multiple experiments
python scripts/run_all_experiments.py --device cuda --experiments E1 E5 E9

# Using shell scripts
bash scripts/run_e1.sh    # Runs only E1
bash scripts/run_e9.sh    # Runs only E9
```

### 4. Collect Results Only (Without Running)

```bash
python scripts/run_all_experiments.py --device cuda --collect-only
```

### 5. Generate LaTeX Tables

```bash
python scripts/generate_latex_tables.py \
    --results outputs/experiments/summary.csv
```

This generates:
- `table_main_results.tex` - All experiments with all metrics
- `table_model_comparison.tex` - Model comparison
- `table_loss_comparison.tex` - Loss function comparison
- `table_ablation_study.tex` - Ablation study impact
- `ranking_*.tex` - Rankings by each metric (AUC, AUPRC, F1, Sensitivity, Specificity)

## Output Structure

```
outputs/experiments/
├── run_log.json                    (Execution log)
├── summary.csv                     (Results CSV)
├── summary.json                    (Results JSON)
├── summary.tex                     (LaTeX table)
├── table_main_results.tex
├── table_model_comparison.tex
├── table_loss_comparison.tex
├── table_ablation_study.tex
├── ranking_roc_auc.tex
├── ranking_auprc.tex
├── ranking_f1.tex
├── ranking_sensitivity.tex
├── ranking_specificity.tex
├── E1/cv_results/                  (E1 results)
├── E2/cv_results/                  (E2 results)
└── ... (E3-E15)
```

## Resume & Recovery

The system automatically tracks experiment status in `run_log.json`:

```json
{
  "E1": {
    "status": "completed",
    "elapsed_time": 600.5,
    "name": "EfficientNet-B0 + BCE"
  },
  "E2": {
    "status": "failed",
    "return_code": 1,
    "name": "EfficientNet-B0 + Weighted BCE"
  }
}
```

**Resume Behavior:**
- If interrupted, next run continues from where it stopped
- Use `--force` flag to re-run all experiments
- Use `--experiments E5 E7` to retry failed experiments

## Adding New Experiments

### Step 1: Edit `configs/experiments.yaml`

```yaml
experiments:
  E16:
    name: "Custom Experiment"
    model_name: densenet121
    loss_type: cb_focal
    weighted_sampler: true
    smote: false
    mixup: false
    cutmix: false
    cv_folds: 2
    num_epochs: 20
```

### Step 2: Regenerate Configs

```bash
python scripts/generate_experiments.py
```

### Step 3: Run

```bash
python scripts/run_all_experiments.py --device cuda --experiments E16
```

## Customizing Hyperparameters

### For All Experiments

Edit `configs/experiments.yaml` `defaults` section:

```yaml
defaults:
  batch_size: 16          # Change for all
  learning_rate: 0.0001
  num_epochs: 30
```

### For Specific Experiments

Edit experiment definition in `configs/experiments.yaml`:

```yaml
E1:
  name: "Custom E1"
  model_name: efficientnet_b0
  loss_type: bce
  cv_folds: 5             # Override defaults
```

Then regenerate:
```bash
python scripts/generate_experiments.py
```

## HPC Cluster Execution

### Submit to Slurm

```bash
sbatch scripts/run_all_hpc.sh
```

This script:
1. Loads CUDA modules
2. Generates configs
3. Runs all experiments with 72-hour time limit
4. Logs to `logs/ablation_*.log`

### Monitor

```bash
# Check job status
squeue -u $USER

# Check logs
tail -f logs/ablation_*.log

# Get results while running
python scripts/run_all_experiments.py --device cuda --collect-only
```

## Reproduce Paper Results

### One-Command Reproduction

```bash
# Generate configs + Run all experiments + Create tables
bash scripts/run_all.sh && \
python scripts/generate_latex_tables.py --results outputs/experiments/summary.csv
```

### Step-by-Step

```bash
# 1. Generate experiment configs
python scripts/generate_experiments.py \
    --base-config configs/config_hpc.yaml \
    --output-dir configs/experiments

# 2. Run all experiments (takes ~24-48 hours depending on GPU)
python scripts/run_all_experiments.py --device cuda --resume

# 3. Generate tables for paper
python scripts/generate_latex_tables.py \
    --results outputs/experiments/summary.csv

# 4. View results
cat outputs/experiments/summary.csv
cat outputs/experiments/table_main_results.tex
```

## Performance Metrics

The system tracks and reports:

- **ROC-AUC**: Area under ROC curve
- **AUPRC**: Area under precision-recall curve
- **F1-Score**: Harmonic mean of precision and recall
- **Sensitivity**: True positive rate (recall)
- **Specificity**: True negative rate
- **Accuracy**: Overall correctness

Each metric includes:
- Mean value across folds
- Standard deviation

## Example Results CSV Format

| experiment_id | experiment_name | model | loss | roc_auc | roc_auc_std | auprc | auprc_std | ... |
|---|---|---|---|---|---|---|---|---|
| E1 | EfficientNet-B0 + BCE | efficientnet_b0 | bce | 0.8234 | 0.0145 | 0.4521 | 0.0234 | ... |
| E2 | EfficientNet-B0 + ... | efficientnet_b0 | weighted_bce | 0.8456 | 0.0123 | 0.4892 | 0.0156 | ... |

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in `defaults` of `experiments.yaml`
- Use smaller model (e.g., `efficientnet_b0` instead of `vit_small`)
- Enable gradient accumulation

### Config Generation Issues
```bash
# Regenerate all configs
python scripts/generate_experiments.py --force
```

### Missing Results
```bash
# Collect only existing results
python scripts/run_all_experiments.py --collect-only
```

### Resume Failed
```bash
# Force re-run specific failed experiments
python scripts/run_all_experiments.py \
    --device cuda \
    --experiments E5 E12 \
    --force
```

## Configuration File Reference

### `configs/experiments.yaml`

Defines:
- 15 experiments (E1-E15)
- Model name, loss function
- Which imbalance techniques to enable
- Number of folds and epochs
- Default hyperparameters

### Generated Config Example (`configs/experiments/config_E1.yaml`)

Contains merged configuration with:
- Base settings from `config_hpc.yaml`
- Experiment-specific overrides from `experiments.yaml`
- Metadata (`experiment.id`, `experiment.name`)

### `run_log.json`

Tracks:
- Completion status (completed/failed/timeout)
- Elapsed time per experiment
- Error messages if any

## Key Design Principles

1. **Reproducibility**: All configs version-controlled, deterministic
2. **Scalability**: Easy to add new experiments
3. **Robustness**: Automatic resume on interruption
4. **Traceability**: Complete audit trail in logs
5. **Automation**: One-command reproduction

## Files Overview

```
scripts/
├── main_train.py                    (Main training script - unchanged)
├── generate_experiments.py          (Generate configs)
├── run_all_experiments.py           (Orchestrate all experiments)
├── generate_latex_tables.py         (Create paper tables)
├── run_all.sh                       (Run all experiments)
├── run_all_hpc.sh                   (HPC submission script)
├── run_e1.sh                        (Run E1 only)
└── run_e9.sh                        (Run E9 only)

configs/
├── config_hpc.yaml                  (Base template)
├── experiments.yaml                 (Experiment definitions)
└── experiments/                     (Generated configs)
    ├── config_E1.yaml
    ├── config_E2.yaml
    ├── ...
    └── experiment_index.yaml
```

## Performance Notes

**Typical Timing (per experiment with 2-fold CV):**
- EfficientNet-B0: ~4 minutes per fold = ~8 min total
- ConvNeXt-Tiny: ~6 minutes per fold = ~12 min total
- ViT/Swin: ~10 minutes per fold = ~20 min total

**Total Estimated Time:**
- E1-E5 (EfficientNet): ~40 min
- E6-E13 (ConvNeXt): ~96 min
- E14 (ViT): ~20 min
- E15 (Swin): ~20 min
- **Total: ~3-4 hours on single GPU**

## Citation

When using this system, cite:

```bibtex
@article{skinCancer2026,
  title={Comprehensive Ablation Study of Loss Functions and Sampling Techniques for Skin Cancer Classification},
  author={Your Name},
  journal={Your Journal},
  year={2026}
}
```
