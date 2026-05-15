"""Run training for all models specified in config."""
import argparse
import subprocess
import sys
from pathlib import Path
import yaml
from typing import List

def get_available_models() -> List[str]:
    """Get list of available models to train."""
    # Models that are tested and available
    models = [
        "efficientnet_b0",
        "convnext_tiny",
    ]
    return models

def modify_config_for_model(config_path: str, model_name: str) -> str:
    """
    Create a temporary config file with the specified model.
    
    Args:
        config_path: Path to the base config file
        model_name: Model name to use
        
    Returns:
        Path to the modified config file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update model name
    config['model']['model_name'] = model_name
    
    # Create temp config path
    temp_config_path = Path(config_path).parent / f"config_temp_{model_name}.yaml"
    
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    return str(temp_config_path)

def run_model(config_path: str, model_name: str, device: str = 'cuda'):
    """Run training for a single model."""
    print(f"\n{'='*80}")
    print(f"Training Model: {model_name}")
    print(f"{'='*80}\n")
    
    # Create temporary config for this model
    temp_config = modify_config_for_model(config_path, model_name)
    
    try:
        # Run main_train.py with the modified config
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "main_train.py"),
            "--config", temp_config,
            "--device", device,
        ]
        
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {model_name} training completed successfully\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {model_name} training failed with error code {e.returncode}\n")
        return False
    finally:
        # Clean up temp config
        Path(temp_config).unlink(missing_ok=True)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run training for all models")
    parser.add_argument("--config", type=str, default="configs/config_hpc.yaml",
                        help="Base config file path")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Specific models to run (default: all)")
    
    args = parser.parse_args()
    
    # Get models to train
    available_models = get_available_models()
    models_to_train = args.models if args.models else available_models
    
    print(f"\nAvailable models: {available_models}")
    print(f"Models to train: {models_to_train}\n")
    
    # Run training for each model
    successful = []
    failed = []
    
    for model_name in models_to_train:
        if model_name not in available_models:
            print(f"⚠ Skipping unknown model: {model_name}")
            continue
        
        if run_model(args.config, model_name, args.device):
            successful.append(model_name)
        else:
            failed.append(model_name)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Successful: {len(successful)}/{len(models_to_train)} models")
    if successful:
        for model in successful:
            print(f"  ✓ {model}")
    
    if failed:
        print(f"Failed: {len(failed)}/{len(models_to_train)} models")
        for model in failed:
            print(f"  ✗ {model}")
    
    sys.exit(0 if len(failed) == 0 else 1)

if __name__ == "__main__":
    main()
