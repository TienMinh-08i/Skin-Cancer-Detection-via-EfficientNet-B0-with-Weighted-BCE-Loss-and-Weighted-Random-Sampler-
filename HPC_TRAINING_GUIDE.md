# HPC (High Performance Computing) Training Guide

When using HPC systems, you typically don't have internet access during compute jobs. This guide explains how to pre-download models and use them offline.

## Overview

The workflow has 2 steps:

1. **Download models** (on a machine with internet)
2. **Train offline** (on HPC system)

## Step 1: Download Pretrained Models

### On a local machine with internet:

```bash
# Clone or prepare the project
cd /path/to/como

# Create models cache directory
mkdir -p ~/models_cache

# Download all supported models
python scripts/download_models.py --cache-dir ~/models_cache
```

This downloads ~450MB of models:
- EfficientNet-B0
- EfficientNet-B3
- ConvNeXt-Tiny
- Vision Transformer-Tiny
- Swin Transformer-Tiny

**Cache structure after download:**
```
~/models_cache/
├── checkpoints/
│   └── pytorch_model.bin files
├── hub/
│   └── model metadata
└── ...
```

### Transfer to HPC:

```bash
# Upload to HPC system
scp -r ~/models_cache username@hpc.server:/path/to/project/
```

## Step 2: Configure for Offline Training

### Edit the config file:

```bash
# Copy HPC template
cp configs/config_hpc_template.yaml configs/config_hpc.yaml

# Edit the paths
vim configs/config_hpc.yaml
```

**Important settings:**
```yaml
# Must point to your downloaded models
models_cache_dir: /path/to/project/models_cache

# Your dataset paths
dataset:
  dataset_root: /path/to/dataset
  csv_path: /path/to/dataset/metadata.csv
  image_dir: /path/to/dataset/images
```

## Step 3: Run Training on HPC

### Interactive job:
```bash
# SSH to HPC login node
ssh username@hpc.server

# Load modules
module load python/3.11
module load cuda/12.0

# Activate environment
source venv/bin/activate

# Run training
python scripts/main_train.py \
  --config configs/config_hpc.yaml \
  --device cuda
```

### SLURM job script:

Create `train_job.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=skin_cancer_cv
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_%j.log

# Load modules
module load python/3.11
module load cuda/12.0

# Set cache directories
export TORCH_HOME=/path/to/project/models_cache
export TIMM_HOME=/path/to/project/models_cache

# Activate environment
source venv/bin/activate

# Run training
python scripts/main_train.py \
  --config configs/config_hpc.yaml \
  --device cuda

# Summary
echo "Training completed!"
```

Submit:
```bash
sbatch train_job.sh
```

## Configuration Methods

### Method 1: Set in YAML config (Recommended)
```yaml
# configs/config_hpc.yaml
models_cache_dir: /path/to/models_cache
```

### Method 2: Set environment variables
```bash
export TORCH_HOME=/path/to/models_cache
export TIMM_HOME=/path/to/models_cache
python scripts/main_train.py --config configs/config_hpc.yaml
```

### Method 3: Programmatic
```python
from pathlib import Path
from src.configs import Config
from src.models import build_model

config = Config()
config.models_cache_dir = Path("/path/to/models_cache")

model = build_model(
    "efficientnet_b0",
    pretrained=True,
    cache_dir=config.models_cache_dir
)
```

## Troubleshooting

### Model download fails
**Error:** "Failed to download pretrained weights"
**Solution:** Ensure internet is connected and try again on local machine

### Training fails with "module not found"
**Error:** `ModuleNotFoundError: No module named 'timm'`
**Solution:** Run `pip install -r requirements.txt`

### CUDA out of memory
**Error:** `RuntimeError: CUDA out of memory`
**Solution:** Reduce batch_size in config:
```yaml
training:
  batch_size: 16  # From 32
```

### Models not found during training
**Error:** `FileNotFoundError: models not cached`
**Solution:** Verify `models_cache_dir` in config is correct:
```bash
ls /path/to/models_cache  # Should show files
```

## Performance Tips

1. **Enable mixed precision (default)** - 50% memory savings
   ```yaml
   training:
     use_amp: true
   ```

2. **Increase workers for faster data loading**
   ```yaml
   training:
     num_workers: 8  # Match number of CPU cores
   ```

3. **Pin memory for GPU transfer**
   ```yaml
   training:
     pin_memory: true
   ```

4. **Use gradient accumulation for larger effective batch**
   ```yaml
   training:
     gradient_accumulation_steps: 2  # 2x batch size effect
   ```

## Example Full Workflow

```bash
# 1. LOCAL MACHINE
# Download models
python scripts/download_models.py --cache-dir ~/models_cache

# 2. TRANSFER TO HPC
scp -r ~/models_cache myuser@hpc.server:~/project/

# 3. ON HPC
ssh myuser@hpc.server
cd ~/project

# Setup
pip install -r requirements.txt
cp configs/config_hpc_template.yaml configs/config_hpc.yaml
# Edit config_hpc.yaml with your paths

# Run training
python scripts/main_train.py --config configs/config_hpc.yaml --device cuda

# Check results
cat outputs/cv_results/cv_summary_metrics.csv
```

## For Different HPC Systems

### XSEDE/Extreme Science and Engineering Discovery Environment
```bash
# Load modules
module load python
module load cuda/11.0

# Run training as above
```

### Local HPC Cluster
Consult your cluster documentation for module/environment setup

### Colab (Alternative)
If you have internet in Colab:
```bash
!pip install -r requirements.txt
!python scripts/main_train.py --config configs/config_isic2019.yaml
```
(No model pre-download needed, Colab has internet)

## Support

For issues:
1. Check [../README.md](../README.md#Troubleshooting)
2. See [hpc_workflow.py](hpc_workflow.py) for additional examples
3. Review SLURM job output: `cat slurm_*.log`

---

**Quick Reference:**
```bash
# Step 1: Download (local machine with internet)
python scripts/download_models.py --cache-dir ~/models_cache

# Step 2: Configure (on HPC)
cp configs/config_hpc_template.yaml configs/config_hpc.yaml
vim configs/config_hpc.yaml  # Set your paths

# Step 3: Train (on HPC)
python scripts/main_train.py --config configs/config_hpc.yaml --device cuda
```
