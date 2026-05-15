"""Generate config files for all experiments."""
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any

def load_experiments() -> Dict[str, Dict[str, Any]]:
    """Load experiment definitions."""
    experiments_file = Path(__file__).parent.parent / "configs" / "experiments.yaml"
    with open(experiments_file, 'r') as f:
        data = yaml.safe_load(f)
    return data['experiments'], data['defaults']

def load_base_config(config_path: str) -> Dict[str, Any]:
    """Load base config template."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def merge_config(base_config: Dict, experiment: Dict, defaults: Dict) -> Dict:
    """Merge base config with experiment-specific settings."""
    config = base_config.copy()
    
    # Apply defaults
    config['model']['model_name'] = defaults.get('model_name', config['model']['model_name'])
    config['model']['pretrained'] = defaults.get('pretrained', config['model']['pretrained'])
    config['training']['batch_size'] = defaults.get('batch_size', config['training']['batch_size'])
    config['training']['num_epochs'] = defaults.get('num_epochs', config['training']['num_epochs'])
    config['optimizer']['learning_rate'] = defaults.get('learning_rate', config['optimizer']['learning_rate'])
    config['optimizer']['weight_decay'] = defaults.get('weight_decay', config['optimizer']['weight_decay'])
    config['dataset']['image_size'] = defaults.get('image_size', config['dataset']['image_size'])
    config['dataset']['random_state'] = defaults.get('random_state', config['dataset']['random_state'])
    config['training']['use_amp'] = defaults.get('use_amp', config['training']['use_amp'])
    config['training']['early_stopping_patience'] = defaults.get('early_stopping_patience', config['training']['early_stopping_patience'])
    config['training']['early_stopping_metric'] = defaults.get('early_stopping_metric', config['training']['early_stopping_metric'])
    
    # Apply experiment-specific settings
    if 'model_name' in experiment:
        config['model']['model_name'] = experiment['model_name']
    if 'cv_folds' in experiment:
        config['dataset']['cv_folds'] = experiment['cv_folds']
    if 'num_epochs' in experiment:
        config['training']['num_epochs'] = experiment['num_epochs']
    
    # Loss configuration
    if 'loss_type' in experiment:
        config['loss']['loss_type'] = experiment['loss_type']
    
    # Imbalance handling
    config['imbalance']['weighted_sampler_enabled'] = experiment.get('weighted_sampler', False)
    config['imbalance']['smote_enabled'] = experiment.get('smote', False)
    config['imbalance']['mixup_enabled'] = experiment.get('mixup', False)
    config['imbalance']['cutmix_enabled'] = experiment.get('cutmix', False)
    
    return config

def generate_configs(base_config_path: str, output_dir: str):
    """Generate all experiment config files."""
    experiments, defaults = load_experiments()
    base_config = load_base_config(base_config_path)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    experiment_configs = {}
    
    for exp_id, exp_config in experiments.items():
        # Merge configurations
        merged_config = merge_config(base_config, exp_config, defaults)
        
        # Save config (without adding experiment metadata to avoid Config class issues)
        config_file = output_path / f"config_{exp_id}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)
        
        experiment_configs[exp_id] = {
            'name': exp_config['name'],
            'config_path': str(config_file),
            'model': exp_config.get('model_name'),
            'loss': exp_config.get('loss_type'),
            'weighted_sampler': exp_config.get('weighted_sampler', False),
            'smote': exp_config.get('smote', False),
            'mixup': exp_config.get('mixup', False),
            'cutmix': exp_config.get('cutmix', False),
        }
        
        print(f"✓ Generated {config_file}")
    
    # Save experiment index
    index_file = output_path / "experiment_index.yaml"
    with open(index_file, 'w') as f:
        yaml.dump(experiment_configs, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Generated {len(experiment_configs)} experiment configs")
    print(f"✓ Saved experiment index to {index_file}")
    
    return experiment_configs

def main():
    parser = argparse.ArgumentParser(description="Generate experiment configs")
    parser.add_argument("--base-config", type=str, default="configs/config_hpc.yaml",
                        help="Base config file path")
    parser.add_argument("--output-dir", type=str, default="configs/experiments",
                        help="Output directory for experiment configs")
    
    args = parser.parse_args()
    
    generate_configs(args.base_config, args.output_dir)

if __name__ == "__main__":
    main()
