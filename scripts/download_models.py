"""Pre-download pretrained models for offline HPC usage."""
import argparse
import os
from pathlib import Path

import torch
import timm


def download_pretrained_models(cache_dir: Path) -> None:
    """Download all supported pretrained models to cache directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set torch cache and timm cache to the specified directory
    os.environ["TORCH_HOME"] = str(cache_dir)
    os.environ["TIMM_HOME"] = str(cache_dir)
    
    # Models to download
    models = [
        "efficientnet_b0",
        "efficientnet_b3",
        "convnext_tiny",
        "vit_tiny_patch16_224",
        "swin_tiny_patch4_window7_224",
    ]
    
    print(f"Cache directory: {cache_dir}")
    print(f"Setting TORCH_HOME={os.environ['TORCH_HOME']}")
    print(f"Setting TIMM_HOME={os.environ['TIMM_HOME']}\n")
    
    for model_name in models:
        try:
            print(f"Downloading {model_name}...", end=" ", flush=True)
            model = timm.create_model(model_name, pretrained=True)
            print(f"✓ Success ({sum(p.numel() for p in model.parameters()):,} params)")
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    print(f"\n✓ All models downloaded to {cache_dir}")
    print(f"\nTo use these models offline, set environment variables before training:")
    print(f"  export TORCH_HOME={cache_dir}")
    print(f"  export TIMM_HOME={cache_dir}")
    print(f"\nOr add to your training script:")
    print(f"  os.environ['TORCH_HOME'] = '{cache_dir}'")
    print(f"  os.environ['TIMM_HOME'] = '{cache_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-download pretrained models for offline HPC training"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./models_cache",
        help="Directory to store downloaded models (default: ./models_cache)",
    )
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir).resolve()
    download_pretrained_models(cache_dir)
