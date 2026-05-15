# 🎯 Experiment System - Setup Summary

## What Was Created

### 1. **Core System Files** ✅

```
configs/
├── experiments.yaml                     ← 15 experiment definitions (E1-E15)
└── experiments/                         ← Auto-generated configs
    ├── config_E1.yaml through config_E15.yaml
    └── experiment_index.yaml

scripts/
├── generate_experiments.py              ← Creates config files from definitions
├── run_all_experiments.py               ← Orchestrates all experiments
├── generate_latex_tables.py             ← Creates paper tables
├── run_all.sh                           ← Run all experiments (1 command)
├── run_all_hpc.sh                       ← Run on HPC cluster
├── quickstart.sh                        ← Complete pipeline in 1 script
├── run_e1.sh                            ← Quick run E1
└── run_e9.sh                            ← Quick run E9
```

### 2. **Documentation** 📖

```
README_EXPERIMENTS.md                   ← Quick start guide
EXPERIMENT_GUIDE.md                     ← Detailed technical guide
ARCHITECTURE_CONFIGS.md                 ← Config inheritance deep-dive
```

### 3. **15 Experiments Pre-Configured**

```
E1-E5:   EfficientNet-B0 with different loss functions & techniques
E6-E13:  ConvNeXt-Tiny with comprehensive ablations
E14:     Vision Transformer + CB-Focal Loss
E15:     Swin Transformer + CB-Focal Loss
```

---

## ⚡ Quick Start Commands

### Generate Configs (Already Done ✓)
```bash
# Configs auto-generated in: configs/experiments/config_E1.yaml through config_E15.yaml
ls configs/experiments/
```

### Run All Experiments (Takes ~3-4 hours)
```bash
bash scripts/run_all.sh
# OR
python scripts/run_all_experiments.py --device cuda --resume
```

### Run Specific Experiments
```bash
# Single
bash scripts/run_e1.sh

# Multiple
python scripts/run_all_experiments.py --device cuda --experiments E1 E5 E9

# Resume from interruption (automatic)
python scripts/run_all_experiments.py --device cuda --resume
```

### Generate Paper Tables
```bash
# Automatically done after run_all_experiments.py
# Manual generation:
python scripts/generate_latex_tables.py --results outputs/experiments/summary.csv
```

### One-Command Everything
```bash
bash scripts/quickstart.sh
# Generates configs + Runs all + Creates tables
```

---

## 📊 What You'll Get

After running all experiments:

```
outputs/experiments/
├── summary.csv                          ← Main results file (spreadsheet)
├── summary.json                         ← JSON format for processing
├── run_log.json                         ← Execution status log
│
├── table_main_results.tex               ← Full results table (for paper)
├── table_model_comparison.tex           ← Model comparison
├── table_loss_comparison.tex            ← Loss function impact
├── table_ablation_study.tex             ← Ablation study results
│
├── ranking_roc_auc.tex                  ← Top experiments by AUC
├── ranking_auprc.tex                    ← Top experiments by AUPRC
├── ranking_f1.tex                       ← Top experiments by F1
├── ranking_sensitivity.tex              ← Top experiments by Sensitivity
├── ranking_specificity.tex              ← Top experiments by Specificity
│
└── E1/ through E15/                     ← Individual experiment outputs
    ├── cv_results/
    │   ├── cv_results.csv               ← Fold-wise metrics
    │   ├── fold_0_best.pt               ← Best model fold 0
    │   └── fold_1_best.pt               ← Best model fold 1
    └── ...
```

---

## 🏗️ Architecture Overview

```
CONFIG HIERARCHY:

config_hpc.yaml (base template)
        ↓ merge with
experiments.yaml (E1-E15 definitions)
        ↓ generate_experiments.py
configs/experiments/config_E*.yaml (15 final configs)
        ↓ run_all_experiments.py
main_train.py (training script)
        ↓ collect results
outputs/experiments/summary.csv
        ↓ generate_latex_tables.py
*.tex tables for paper
```

**Key Principle**: Only differences specified between experiments → Base shared across all → Consistency + Efficiency

---

## 🎓 Understanding the System

### What Each File Does

| File | Purpose |
|------|---------|
| `configs/experiments.yaml` | Define E1-E15: which model, loss, techniques for each |
| `scripts/generate_experiments.py` | Merge base config + definitions → Create config_E*.yaml |
| `scripts/run_all_experiments.py` | Run all/some experiments, track status, collect results |
| `scripts/generate_latex_tables.py` | Read summary.csv → Create paper tables |
| `scripts/run_all.sh` | Convenience: execute everything |

### Config Inheritance Example

**E1** (Base + Override):
```yaml
# From base (config_hpc.yaml): batch_size=8, lr=0.0003, etc.
# Override from E1 definition: model=efficientnet_b0, loss=bce
# Result: config_E1.yaml = complete config for EfficientNet-B0 + BCE
```

**E5** (Different):
```yaml
# From base: batch_size=8, lr=0.0003, etc. (SAME as E1)
# Override from E5 definition: model=efficientnet_b0, loss=cb_focal, weighted_sampler=true
# Result: config_E5.yaml = complete config for EfficientNet-B0 + CB-FL + Weighted Sampler
```

→ **Fair comparison**: Same hyperparameters, only model/loss/technique differ

---

## 📝 Customization Guide

### Change Hyperparameter for ALL Experiments
```bash
# 1. Edit configs/experiments.yaml
nano configs/experiments.yaml
# Find "defaults:" section, change batch_size: 8 → 16

# 2. Regenerate configs
python scripts/generate_experiments.py

# 3. Re-run (with --force to override old results)
python scripts/run_all_experiments.py --device cuda --force
```

### Add New Experiment (E16)
```bash
# 1. Edit configs/experiments.yaml - add:
# E16:
#   name: "Custom Experiment"
#   model_name: densenet121
#   loss_type: cb_focal
#   weighted_sampler: true
#   cv_folds: 2
#   num_epochs: 20

# 2. Regenerate
python scripts/generate_experiments.py

# 3. Run
python scripts/run_all_experiments.py --experiments E16
```

### Modify Specific Experiment (E11)
```bash
# 1. Edit E11 in configs/experiments.yaml

# 2. Regenerate
python scripts/generate_experiments.py

# 3. Re-run with --force (override old results)
python scripts/run_all_experiments.py --experiments E11 --force
```

---

## 🔍 Monitoring & Recovery

### Check Status During Run
```bash
# See what's running
cat outputs/experiments/run_log.json | jq .

# Monitor progress
tail -f outputs/experiments/run_log.json
```

### Resume After Interruption
```bash
# Automatic resume (picks up where it stopped)
python scripts/run_all_experiments.py --device cuda --resume
```

### Re-run Failed Experiments
```bash
# Force re-run specific ones
python scripts/run_all_experiments.py --device cuda --experiments E5 E12 --force
```

### Collect Results Without Re-running
```bash
python scripts/run_all_experiments.py --device cuda --collect-only
```

---

## 🖥️ HPC Cluster Usage

### Submit All Experiments to Slurm
```bash
sbatch scripts/run_all_hpc.sh
# Job will: generate configs → run all experiments → create tables
```

### Monitor Job
```bash
# Check status
squeue -u $USER

# Check output
tail -f logs/ablation_*.log

# Check intermediate results
python scripts/run_all_experiments.py --collect-only
```

---

## 📈 Results Interpretation

Results CSV contains these columns:

| Column | Meaning |
|--------|---------|
| `experiment_id` | E1, E2, ..., E15 |
| `experiment_name` | "EfficientNet-B0 + BCE" |
| `model` | Model used (efficientnet_b0, convnext_tiny, etc.) |
| `loss` | Loss function (bce, cb_focal, etc.) |
| `weighted_sampler` | True/False |
| `smote`, `mixup`, `cutmix` | Technique flags |
| `roc_auc` | Mean AUC (0-1, higher is better) |
| `auprc` | Mean AUPRC (0-1, higher is better) |
| `f1` | F1 score (0-1, higher is better) |
| `sensitivity` | True positive rate (0-1, higher is better) |
| `specificity` | True negative rate (0-1, higher is better) |
| `accuracy` | Overall accuracy (0-1, higher is better) |
| `*_std` | Standard deviation across folds |

---

## ⏱️ Expected Timing

| Experiment | Time per Fold | Total (2-fold CV) |
|------------|---------------|-------------------|
| E1-E5 (EfficientNet) | ~4 min | ~8 min each |
| E6-E13 (ConvNeXt) | ~6 min | ~12 min each |
| E14 (ViT) | ~10 min | ~20 min |
| E15 (Swin) | ~10 min | ~20 min |

**Total Estimated**: ~3-4 hours on single GPU

---

## 🚨 Troubleshooting

### "CUDA out of memory"
```bash
# Lower batch size in configs/experiments.yaml defaults section
# batch_size: 8 → 4
python scripts/generate_experiments.py
```

### "Missing results after run"
```bash
# Ensure results properly collected
python scripts/run_all_experiments.py --collect-only
cat outputs/experiments/summary.csv
```

### "Want to regenerate configs"
```bash
python scripts/generate_experiments.py
```

### "Want to restart everything"
```bash
rm -rf configs/experiments/*
rm -rf outputs/experiments/*
python scripts/generate_experiments.py
python scripts/run_all_experiments.py --device cuda
```

---

## ✅ Reproduce Complete Paper

### One-liner
```bash
bash scripts/quickstart.sh
```

### Step-by-step
```bash
# 1. Generate configs
python scripts/generate_experiments.py

# 2. Run all experiments (grab coffee ☕)
python scripts/run_all_experiments.py --device cuda --resume

# 3. Create tables
python scripts/generate_latex_tables.py --results outputs/experiments/summary.csv

# 4. Check results
cat outputs/experiments/summary.csv
cat outputs/experiments/table_main_results.tex
```

---

## 📚 Documentation Files

- **README_EXPERIMENTS.md** ← Start here (quick reference)
- **EXPERIMENT_GUIDE.md** ← Detailed technical guide
- **ARCHITECTURE_CONFIGS.md** ← Config system deep-dive
- **ARCHITECTURE.md** ← Original project architecture
- **HPC_TRAINING_GUIDE.md** ← HPC cluster guide

---

## 🎉 Summary

✅ **15 Experiments Pre-Configured**
- All model/loss/technique combinations defined
- Config files auto-generated

✅ **Automated Execution**
- Run all with 1 command
- Auto-resume on interruption
- Status tracking

✅ **Results Aggregation**
- Automatic metric collection
- CSV, JSON, LaTeX formats

✅ **Paper-Ready Tables**
- Main results table
- Rankings by each metric
- Model/loss comparisons
- Ablation study table

✅ **Complete Documentation**
- Quick start guide
- Technical deep-dives
- Troubleshooting

---

## 🚀 Next Steps

1. **View experiment definitions**:
   ```bash
   cat configs/experiments.yaml
   ```

2. **Check generated configs**:
   ```bash
   ls -la configs/experiments/
   head configs/experiments/config_E1.yaml
   head configs/experiments/config_E9.yaml
   ```

3. **Run single experiment to test**:
   ```bash
   bash scripts/run_e1.sh
   ```

4. **Run all experiments**:
   ```bash
   bash scripts/run_all.sh
   ```

5. **Check results**:
   ```bash
   cat outputs/experiments/summary.csv
   ```

---

**Ready to reproduce the ablation study? Start with `bash scripts/quickstart.sh` 🚀**
