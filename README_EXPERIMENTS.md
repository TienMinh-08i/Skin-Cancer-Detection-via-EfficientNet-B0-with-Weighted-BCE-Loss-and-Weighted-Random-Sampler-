# Ablation Study Experiment System

> Fully automated system for running 15 skin cancer classification experiments with automatic results aggregation and paper table generation.

## ⚡ Quick Start (30 seconds)

```bash
# 1. Configs are already generated ✓
# 2. Run all experiments (takes ~3-4 hours)
bash scripts/run_all.sh

# 3. Tables automatically generated at the end
# 📊 Check results in: outputs/experiments/
```

## 🎯 What's Included

- **15 Complete Experiments** (E1-E15)
  - 3 models: EfficientNet-B0, ConvNeXt-Tiny, Vision/Swin Transformers
  - 4 loss functions: BCE, Weighted BCE, Focal, CB-Focal
  - 4 imbalance techniques: WeightedSampler, SMOTE, Mixup, CutMix

- **Automated Config Generation**
  - Base config + experiment definitions → 15 configs
  - Config hierarchy ensures consistency

- **Experiment Management**
  - Auto resume on interruption
  - Track status in `run_log.json`
  - Re-run failed experiments easily

- **Results Aggregation**
  - Collect metrics from all experiments
  - Save as CSV, JSON, LaTeX
  - Auto-generate paper tables

## 📋 Experiments (E1-E15)

| ID | Configuration | ID | Configuration |
|---|---|---|---|
| **E1** | EffB0 + BCE | **E9** | CN-Tiny + CB-Focal |
| **E2** | EffB0 + W-BCE | **E10** | CN-Tiny + CB-FL + WS |
| **E3** | EffB0 + Focal | **E11** | CN-Tiny + CB-FL + Mixup |
| **E4** | EffB0 + CB-FL | **E12** | CN-Tiny + CB-FL + CutMix |
| **E5** | EffB0 + CB-FL + WS | **E13** | CN-Tiny + CB-FL + SMOTE |
| **E6** | CN-Tiny + BCE | **E14** | ViT + CB-Focal |
| **E7** | CN-Tiny + W-BCE | **E15** | Swin-Tiny + CB-FL |
| **E8** | CN-Tiny + Focal | | |

*EffB0=EfficientNet-B0, CN-Tiny=ConvNeXt-Tiny, WS=WeightedSampler*

## 🚀 Usage

### Run Specific Experiments

```bash
# Single experiment
python scripts/run_all_experiments.py --device cuda --experiments E1

# Multiple experiments
python scripts/run_all_experiments.py --device cuda --experiments E1 E5 E9

# Using shortcuts
bash scripts/run_e1.sh    # Quick run E1
bash scripts/run_e9.sh    # Quick run E9
```

### Resume Interrupted Runs

```bash
# Auto-resumes from last completed
python scripts/run_all_experiments.py --device cuda --resume

# Force re-run everything
python scripts/run_all_experiments.py --device cuda --force
```

### Collect Results Only

```bash
# Generate tables from existing results
python scripts/run_all_experiments.py --device cuda --collect-only
```

### Generate Paper Tables

```bash
# Create all LaTeX tables
python scripts/generate_latex_tables.py --results outputs/experiments/summary.csv
```

## 📊 Output Files

```
outputs/experiments/
├── summary.csv                      # All metrics
├── summary.json                     # JSON format
├── table_main_results.tex          # Full results table
├── table_model_comparison.tex      # Model performance comparison
├── table_loss_comparison.tex       # Loss function impact
├── table_ablation_study.tex        # Ablation study (technique impact)
├── ranking_roc_auc.tex             # Ranked by AUC
├── ranking_auprc.tex               # Ranked by AUPRC
├── ranking_f1.tex                  # Ranked by F1
├── ranking_sensitivity.tex         # Ranked by Sensitivity
├── ranking_specificity.tex         # Ranked by Specificity
└── run_log.json                    # Execution log
```

## 🔧 Configuration

### View Experiment Definitions

```bash
cat configs/experiments.yaml
```

### Add New Experiment

1. Edit `configs/experiments.yaml` and add E16
2. Regenerate configs: `python scripts/generate_experiments.py`
3. Run: `python scripts/run_all_experiments.py --experiments E16`

### Modify Hyperparameters

Edit `configs/experiments.yaml` → `defaults` section for all experiments:

```yaml
defaults:
  batch_size: 16          # Change batch size
  num_epochs: 30          # Change training epochs
  learning_rate: 0.0001   # Change learning rate
```

Then regenerate and re-run.

## 📈 Results Interpretation

Results CSV contains:
- **experiment_id**: E1, E2, ..., E15
- **experiment_name**: Human-readable name
- **model, loss**: Configuration used
- **roc_auc, auprc, f1**: Performance metrics
- **sensitivity, specificity**: Class-specific metrics
- **accuracy**: Overall performance

Metrics are **mean ± std** across 2-fold CV.

## 🖥️ HPC Cluster Usage

Submit to Slurm:
```bash
sbatch scripts/run_all_hpc.sh
```

Monitor:
```bash
squeue -u $USER
tail -f logs/ablation_*.log
```

## ⏱️ Timing

- **EfficientNet-B0**: ~8 min per experiment (5 experiments = 40 min)
- **ConvNeXt-Tiny**: ~12 min per experiment (8 experiments = 96 min)
- **ViT/Swin**: ~20 min per experiment (2 experiments = 40 min)
- **Total**: ~3-4 hours on single GPU

## 📚 File Structure

```
scripts/
├── main_train.py                # Main training (unchanged)
├── generate_experiments.py      # Create configs from definitions
├── run_all_experiments.py       # Orchestrate all runs
├── generate_latex_tables.py     # Create paper tables
├── run_all.sh                   # Run all (local)
├── run_all_hpc.sh               # Run all (HPC)
├── quickstart.sh                # One-command everything
└── run_e*.sh                    # Individual experiments

configs/
├── config_hpc.yaml              # Base template
├── experiments.yaml             # Experiment definitions
└── experiments/
    ├── config_E1.yaml ... config_E15.yaml
    └── experiment_index.yaml
```

## 🐛 Troubleshooting

**Out of memory?**
- Lower `batch_size` in `configs/experiments.yaml`
- Use `--experiments E1` to run single model

**Configs missing?**
```bash
python scripts/generate_experiments.py
```

**Results not showing?**
```bash
python scripts/run_all_experiments.py --collect-only
```

**Re-run failed experiments?**
```bash
python scripts/run_all_experiments.py --force --experiments E5 E12
```

## 📖 Full Documentation

See [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) for:
- Detailed architecture explanation
- Config inheritance mechanism
- Adding new experiments
- Configuration customization
- Resume & recovery logic
- Reproduce entire paper

## ✅ One-Command Paper Reproduction

```bash
# Everything: generate → run → tables
bash scripts/quickstart.sh
```

Or manually:
```bash
# 1. Generate configs
python scripts/generate_experiments.py

# 2. Run all experiments
python scripts/run_all_experiments.py --device cuda --resume

# 3. Generate tables
python scripts/generate_latex_tables.py --results outputs/experiments/summary.csv

# 4. Check results
cat outputs/experiments/summary.csv
```

## 📝 Log Files

All runs logged to `outputs/experiments/run_log.json`:
```json
{
  "E1": {"status": "completed", "elapsed_time": 480},
  "E2": {"status": "failed", "return_code": 1},
  "E3": {"status": "timeout"}
}
```

## 📧 Contact

Questions? Check [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) for detailed explanations.
