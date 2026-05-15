"""Run all experiments with proper tracking and result aggregation."""
import argparse
import subprocess
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import pandas as pd
from datetime import datetime

class ExperimentRunner:
    """Manage running experiments."""
    
    def __init__(self, base_dir: str = ".", device: str = "cuda"):
        self.base_dir = Path(base_dir)
        self.device = device
        self.experiments_dir = self.base_dir / "configs" / "experiments"
        self.results_dir = self.base_dir / "outputs" / "experiments"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load experiment index
        self.experiment_index = self._load_experiment_index()
        self.run_log = self._load_or_create_run_log()
        
    def _load_experiment_index(self) -> Dict:
        """Load experiment index from YAML."""
        index_file = self.experiments_dir / "experiment_index.yaml"
        if not index_file.exists():
            raise FileNotFoundError(f"Experiment index not found: {index_file}")
        
        with open(index_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_or_create_run_log(self) -> Dict:
        """Load or create run log."""
        log_file = self.results_dir / "run_log.json"
        if log_file.exists():
            with open(log_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_run_log(self):
        """Save run log to file."""
        log_file = self.results_dir / "run_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.run_log, f, indent=2)
    
    def run_experiment(self, exp_id: str, force: bool = False) -> bool:
        """Run a single experiment.
        
        Args:
            exp_id: Experiment ID (e.g., "E1")
            force: Force re-run even if completed
            
        Returns:
            True if successful, False otherwise
        """
        if exp_id not in self.experiment_index:
            print(f"✗ Unknown experiment: {exp_id}")
            return False
        
        # Check if already completed
        if not force and self.run_log.get(exp_id, {}).get('status') == 'completed':
            print(f"⊘ Skipping {exp_id} (already completed)")
            return True
        
        exp_info = self.experiment_index[exp_id]
        config_path = exp_info['config_path']
        exp_name = exp_info['name']
        
        print(f"\n{'='*80}")
        print(f"Running {exp_id}: {exp_name}")
        print(f"{'='*80}\n")
        
        # Create experiment output directory
        exp_output_dir = self.results_dir / exp_id
        exp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare command
        cmd = [
            sys.executable,
            str(self.base_dir / "scripts" / "main_train.py"),
            "--config", config_path,
            "--device", self.device,
        ]
        
        # Set output directory in config (temporary)
        config_output_dir = exp_output_dir / "cv_results"
        
        start_time = time.time()
        
        try:
            # Run training
            result = subprocess.run(
                cmd,
                cwd=str(self.base_dir),
                capture_output=False,
                timeout=None,
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"\n✓ {exp_id} completed successfully ({elapsed_time/60:.1f} min)")
                self.run_log[exp_id] = {
                    'status': 'completed',
                    'start_time': datetime.now().isoformat(),
                    'elapsed_time': elapsed_time,
                    'name': exp_name,
                }
                self._save_run_log()
                return True
            else:
                print(f"\n✗ {exp_id} failed with return code {result.returncode}")
                self.run_log[exp_id] = {
                    'status': 'failed',
                    'return_code': result.returncode,
                    'name': exp_name,
                }
                self._save_run_log()
                return False
                
        except subprocess.TimeoutExpired:
            print(f"\n✗ {exp_id} timed out")
            self.run_log[exp_id] = {
                'status': 'timeout',
                'name': exp_name,
            }
            self._save_run_log()
            return False
        except Exception as e:
            print(f"\n✗ {exp_id} failed with error: {e}")
            self.run_log[exp_id] = {
                'status': 'error',
                'error': str(e),
                'name': exp_name,
            }
            self._save_run_log()
            return False
    
    def run_all(self, force: bool = False, resume: bool = True) -> Tuple[List[str], List[str]]:
        """Run all experiments.
        
        Args:
            force: Force re-run all experiments
            resume: Resume from last completed experiment
            
        Returns:
            Tuple of (successful_list, failed_list)
        """
        exp_ids = sorted(self.experiment_index.keys())
        
        if resume and not force:
            # Find last completed
            completed = [k for k, v in self.run_log.items() if v.get('status') == 'completed']
            if completed:
                last_completed = sorted(completed)[-1]
                idx = exp_ids.index(last_completed)
                exp_ids = exp_ids[idx+1:]
                print(f"Resuming from {last_completed}...")
                print(f"Remaining experiments: {exp_ids}")
        
        successful = []
        failed = []
        
        for exp_id in exp_ids:
            if self.run_experiment(exp_id, force=force):
                successful.append(exp_id)
            else:
                failed.append(exp_id)
        
        return successful, failed
    
    def collect_results(self) -> pd.DataFrame:
        """Collect and aggregate results from all completed experiments."""
        results = []
        
        for exp_id in sorted(self.experiment_index.keys()):
            exp_info = self.experiment_index[exp_id]
            exp_output_dir = self.results_dir / exp_id
            
            # Look for results CSV
            results_file = exp_output_dir / "cv_results" / "cv_results.csv"
            if not results_file.exists():
                continue
            
            df = pd.read_csv(results_file)
            summary = {
                'experiment_id': exp_id,
                'experiment_name': exp_info['name'],
                'model': exp_info['model'],
                'loss': exp_info['loss'],
                'weighted_sampler': exp_info['weighted_sampler'],
                'smote': exp_info['smote'],
                'mixup': exp_info['mixup'],
                'cutmix': exp_info['cutmix'],
            }
            
            # Add mean metrics
            for metric in ['roc_auc', 'auprc', 'f1', 'sensitivity', 'specificity', 'accuracy']:
                if metric in df.columns:
                    summary[metric] = df[metric].mean()
                    summary[f'{metric}_std'] = df[metric].std()
            
            results.append(summary)
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    def save_results(self, df: pd.DataFrame):
        """Save aggregated results."""
        if df.empty:
            print("No results to save")
            return
        
        # Save CSV
        csv_file = self.results_dir / "summary.csv"
        df.to_csv(csv_file, index=False)
        print(f"✓ Saved {csv_file}")
        
        # Save JSON
        json_file = self.results_dir / "summary.json"
        df.to_json(json_file, orient='records', indent=2)
        print(f"✓ Saved {json_file}")
        
        # Save LaTeX table
        self._save_latex_table(df)
    
    def _save_latex_table(self, df: pd.DataFrame):
        """Generate and save LaTeX table."""
        if df.empty:
            return
        
        # Create ranking versions
        metrics = ['roc_auc', 'auprc', 'f1', 'sensitivity', 'specificity']
        
        # Main table
        cols = ['experiment_id', 'experiment_name', 'model', 'loss']
        for metric in metrics:
            if metric in df.columns:
                cols.append(metric)
        
        df_table = df[cols].copy()
        
        # Format numbers
        for metric in metrics:
            if metric in df_table.columns:
                df_table[metric] = df_table[metric].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        
        # Generate LaTeX
        latex_str = df_table.to_latex(index=False, escape=False)
        
        # Save
        tex_file = self.results_dir / "summary.tex"
        with open(tex_file, 'w') as f:
            f.write(latex_str)
        print(f"✓ Saved {tex_file}")
        
        # Save rankings
        self._save_rankings(df)
    
    def _save_rankings(self, df: pd.DataFrame):
        """Save rankings by different metrics."""
        metrics = ['roc_auc', 'auprc', 'f1', 'sensitivity', 'specificity']
        
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            ranked = df[['experiment_id', 'experiment_name', metric]].copy()
            ranked = ranked.sort_values(metric, ascending=False).reset_index(drop=True)
            ranked['rank'] = range(1, len(ranked) + 1)
            
            tex_file = self.results_dir / f"ranking_{metric}.tex"
            latex_str = ranked.to_latex(index=False, escape=False)
            with open(tex_file, 'w') as f:
                f.write(latex_str)
            print(f"✓ Saved {tex_file}")

def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--base-dir", type=str, default=".",
                        help="Base directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--experiments", type=str, nargs="+", default=None,
                        help="Specific experiments to run (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run all experiments")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from last completed experiment")
    parser.add_argument("--collect-only", action="store_true",
                        help="Only collect results without running")
    
    args = parser.parse_args()
    
    # Generate configs if not exist
    gen_script = Path(args.base_dir) / "scripts" / "generate_experiments.py"
    if gen_script.exists():
        experiments_dir = Path(args.base_dir) / "configs" / "experiments"
        if not experiments_dir.exists() or len(list(experiments_dir.glob("config_E*.yaml"))) == 0:
            print("Generating experiment configs...")
            subprocess.run([sys.executable, str(gen_script)])
    
    runner = ExperimentRunner(args.base_dir, args.device)
    
    if args.collect_only:
        print("Collecting results...")
        df = runner.collect_results()
        runner.save_results(df)
        print("\nResults Summary:")
        print(df)
        return
    
    if args.experiments:
        # Run specific experiments
        successful = []
        failed = []
        for exp_id in args.experiments:
            if runner.run_experiment(exp_id, force=args.force):
                successful.append(exp_id)
            else:
                failed.append(exp_id)
    else:
        # Run all experiments
        successful, failed = runner.run_all(force=args.force, resume=args.resume)
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Successful: {len(successful)}")
    if successful:
        for exp_id in successful:
            print(f"  ✓ {exp_id}")
    
    print(f"\nFailed: {len(failed)}")
    if failed:
        for exp_id in failed:
            print(f"  ✗ {exp_id}")
    
    # Collect and save results
    print("\nCollecting results...")
    df = runner.collect_results()
    runner.save_results(df)
    
    if not df.empty:
        print("\nTop 5 by AUC:")
        if 'roc_auc' in df.columns:
            print(df[['experiment_id', 'experiment_name', 'roc_auc']].nlargest(5, 'roc_auc'))
    
    sys.exit(0 if len(failed) == 0 else 1)

if __name__ == "__main__":
    main()
