"""HPC training workflow example."""
import os
from pathlib import Path

# ============================================================================
# STEP 1: Download models on a machine with internet access
# ============================================================================
# Run this command on a machine that has internet:
#   python scripts/download_models.py --cache-dir /path/to/models_cache
#
# This downloads all pretrained models to /path/to/models_cache
# Then transfer this directory to your HPC system

# ============================================================================
# STEP 2: Set up config for HPC training
# ============================================================================

CONFIG_EXAMPLE = """
# configs/config_hpc.yaml
seed: 42
device: cuda  # or 'cpu'
project_name: skin-cancer-classification
models_cache_dir: /path/to/models_cache  # Set to your HPC cache directory

dataset:
  dataset_root: /path/to/dataset
  csv_path: /path/to/dataset/metadata.csv
  image_dir: /path/to/dataset/images
  image_size: 224
  cv_folds: 5

augmentation:
  rotation_degrees: 25.0
  horizontal_flip_p: 0.5
  vertical_flip_p: 0.5

imbalance:
  method: weighted_sampler
  weighted_sampler_enabled: true

loss:
  loss_type: cb_focal
  focal_alpha: 0.25
  focal_gamma: 2.0
  cb_focal_beta: 0.9999

model:
  model_name: efficientnet_b0  # or convnext_tiny
  pretrained: true  # Will use cached models from models_cache_dir

optimizer:
  optimizer_type: adamw
  learning_rate: 0.0003
  weight_decay: 0.0001

scheduler:
  scheduler_type: cosine
  cosine_t_max: 20

training:
  batch_size: 32
  num_epochs: 20
  num_workers: 4
  pin_memory: true
  early_stopping_patience: 5
  early_stopping_metric: auprc
  use_amp: true
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0

threshold_optimization:
  enabled: true
  methods:
    - youden
    - f1
    - clinical
  clinical_fn_cost_ratio: 1.0

evaluation:
  threshold: 0.5
  save_predictions: true
  compute_gradcam: true
  compute_error_analysis: true
  error_analysis_n_samples: 20

outputs:
  root_dir: ./outputs
"""

# ============================================================================
# HPC WORKFLOW
# ============================================================================
HPC_WORKFLOW = """
STEP-BY-STEP HPC WORKFLOW:
=========================

1. ON LOCAL MACHINE (with internet):
   ────────────────────────────────
   
   # Create model cache
   python scripts/download_models.py --cache-dir ~/models_cache
   
   # This downloads:
   # - efficientnet_b0
   # - efficientnet_b3
   # - convnext_tiny
   # - vit_tiny_patch16_224
   # - swin_tiny_patch4_window7_224
   
   # Model cache structure:
   # ~/models_cache/
   #   ├── checkpoints/
   #   └── hub/
   
   # Transfer to HPC:
   scp -r ~/models_cache username@hpc:/path/to/project/


2. ON HPC SYSTEM:
   ──────────────
   
   # Create config with models_cache_dir
   # Edit configs/config_hpc.yaml and set:
   #   models_cache_dir: /path/to/project/models_cache
   
   # Install Python dependencies
   pip install -r requirements.txt
   
   # Run training (offline, no internet needed!)
   python scripts/main_train.py --config configs/config_hpc.yaml --device cuda


3. ALTERNATIVE: Set environment variables
   ──────────────────────────────────────
   
   # Instead of setting in config, you can set environment variables:
   export TORCH_HOME=/path/to/project/models_cache
   export TIMM_HOME=/path/to/project/models_cache
   
   python scripts/main_train.py --config configs/config_hpc.yaml


4. SLURM JOB SCRIPT EXAMPLE:
   ──────────────────────────
   
   #!/bin/bash
   #SBATCH --job-name=skin_cancer_training
   #SBATCH --time=48:00:00
   #SBATCH --gres=gpu:1
   #SBATCH --mem=64GB
   
   # Set module environment
   module load python/3.11
   module load cuda/12.0
   
   # Set offline model cache
   export TORCH_HOME=/path/to/project/models_cache
   export TIMM_HOME=/path/to/project/models_cache
   
   # Activate environment
   source venv/bin/activate
   
   # Run training
   python scripts/main_train.py \\
     --config configs/config_hpc.yaml \\
     --device cuda
"""

# ============================================================================
# PYTHON EXAMPLE
# ============================================================================
PYTHON_EXAMPLE = """
import os
from pathlib import Path
from src.configs import Config
from src.models import build_model

# Method 1: Set cache_dir in config
config = Config()
config.models_cache_dir = Path("/path/to/models_cache")

model = build_model(
    "efficientnet_b0",
    pretrained=True,
    cache_dir=config.models_cache_dir
)

# Method 2: Set environment variables
os.environ["TORCH_HOME"] = "/path/to/models_cache"
os.environ["TIMM_HOME"] = "/path/to/models_cache"

model = build_model("efficientnet_b0", pretrained=True)
"""

if __name__ == "__main__":
    print(__doc__)
    print(CONFIG_EXAMPLE)
    print(HPC_WORKFLOW)
    print(PYTHON_EXAMPLE)
