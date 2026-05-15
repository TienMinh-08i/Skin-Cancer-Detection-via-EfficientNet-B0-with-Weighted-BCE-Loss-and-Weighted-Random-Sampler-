"""Generate comprehensive LaTeX tables for the paper."""
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List
import numpy as np

class LatexTableGenerator:
    """Generate LaTeX tables for paper."""
    
    def __init__(self, results_csv: str):
        self.df = pd.read_csv(results_csv)
        self.output_dir = Path(results_csv).parent
        
    def generate_main_table(self):
        """Generate main results table."""
        metrics = ['roc_auc', 'auprc', 'f1', 'sensitivity', 'specificity', 'accuracy']
        
        table_data = []
        for _, row in self.df.iterrows():
            row_data = {
                'Exp': row['experiment_id'],
                'Model': row['model'],
                'Loss': row['loss'],
                'Sampler': '✓' if row['weighted_sampler'] else '',
                'SMOTE': '✓' if row['smote'] else '',
                'Mixup': '✓' if row['mixup'] else '',
                'CutMix': '✓' if row['cutmix'] else '',
            }
            
            for metric in metrics:
                if metric in row and pd.notna(row[metric]):
                    row_data[metric.upper()] = f"{row[metric]:.4f}"
                else:
                    row_data[metric.upper()] = "N/A"
            
            table_data.append(row_data)
        
        df_table = pd.DataFrame(table_data)
        
        # Generate LaTeX
        latex = df_table.to_latex(index=False, escape=False)
        
        output_file = self.output_dir / "table_main_results.tex"
        with open(output_file, 'w') as f:
            f.write(latex)
        
        print(f"✓ Generated {output_file}")
    
    def generate_ranking_tables(self):
        """Generate ranking tables for each metric."""
        metrics = {
            'roc_auc': 'AUC (ROC)',
            'auprc': 'AUPRC',
            'f1': 'F1-Score',
            'sensitivity': 'Sensitivity',
            'specificity': 'Specificity',
            'accuracy': 'Accuracy',
        }
        
        for metric, label in metrics.items():
            if metric not in self.df.columns:
                continue
            
            ranked = self.df[['experiment_id', 'experiment_name', 'model', 'loss', metric]].copy()
            ranked = ranked.sort_values(metric, ascending=False).reset_index(drop=True)
            ranked['Rank'] = range(1, len(ranked) + 1)
            
            # Format columns
            ranked['Exp'] = ranked['experiment_id']
            ranked['Model'] = ranked['model']
            ranked['Loss'] = ranked['loss']
            ranked[label] = ranked[metric].apply(lambda x: f"{x:.4f}")
            
            df_output = ranked[['Rank', 'Exp', 'Model', 'Loss', label]]
            
            latex = df_output.to_latex(index=False, escape=False)
            
            output_file = self.output_dir / f"ranking_{metric}.tex"
            with open(output_file, 'w') as f:
                f.write(latex)
            
            print(f"✓ Generated {output_file}")
    
    def generate_model_comparison_table(self):
        """Generate model comparison table."""
        models = self.df['model'].unique()
        
        metrics = ['roc_auc', 'auprc', 'f1', 'sensitivity', 'specificity']
        
        comparison = []
        for model in sorted(models):
            model_df = self.df[self.df['model'] == model]
            row = {'Model': model}
            
            for metric in metrics:
                if metric in model_df.columns:
                    mean_val = model_df[metric].mean()
                    std_val = model_df[metric].std()
                    row[metric.upper()] = f"{mean_val:.4f} ± {std_val:.4f}"
                else:
                    row[metric.upper()] = "N/A"
            
            comparison.append(row)
        
        df_comp = pd.DataFrame(comparison)
        latex = df_comp.to_latex(index=False, escape=False)
        
        output_file = self.output_dir / "table_model_comparison.tex"
        with open(output_file, 'w') as f:
            f.write(latex)
        
        print(f"✓ Generated {output_file}")
    
    def generate_loss_comparison_table(self):
        """Generate loss function comparison table."""
        losses = self.df['loss'].unique()
        
        metrics = ['roc_auc', 'auprc', 'f1', 'sensitivity', 'specificity']
        
        comparison = []
        for loss in sorted(losses):
            loss_df = self.df[self.df['loss'] == loss]
            row = {'Loss': loss}
            
            for metric in metrics:
                if metric in loss_df.columns:
                    mean_val = loss_df[metric].mean()
                    std_val = loss_df[metric].std()
                    row[metric.upper()] = f"{mean_val:.4f} ± {std_val:.4f}"
                else:
                    row[metric.upper()] = "N/A"
            
            comparison.append(row)
        
        df_comp = pd.DataFrame(comparison)
        latex = df_comp.to_latex(index=False, escape=False)
        
        output_file = self.output_dir / "table_loss_comparison.tex"
        with open(output_file, 'w') as f:
            f.write(latex)
        
        print(f"✓ Generated {output_file}")
    
    def generate_ablation_study_table(self):
        """Generate ablation study comparison (techniques impact)."""
        techniques = {
            'Weighted Sampler': 'weighted_sampler',
            'SMOTE': 'smote',
            'Mixup': 'mixup',
            'CutMix': 'cutmix',
        }
        
        metrics = ['roc_auc', 'auprc', 'f1', 'sensitivity', 'specificity']
        
        ablation = []
        for tech_name, tech_col in techniques.items():
            with_tech = self.df[self.df[tech_col] == True]
            without_tech = self.df[self.df[tech_col] == False]
            
            row = {'Technique': tech_name}
            
            if len(with_tech) > 0 and len(without_tech) > 0:
                for metric in metrics:
                    with_mean = with_tech[metric].mean() if metric in with_tech.columns else np.nan
                    without_mean = without_tech[metric].mean() if metric in without_tech.columns else np.nan
                    
                    if not np.isnan(with_mean) and not np.isnan(without_mean):
                        diff = with_mean - without_mean
                        row[f'{metric.upper()} Δ'] = f"{diff:+.4f}"
                    else:
                        row[f'{metric.upper()} Δ'] = "N/A"
            
            ablation.append(row)
        
        df_ablation = pd.DataFrame(ablation)
        latex = df_ablation.to_latex(index=False, escape=False)
        
        output_file = self.output_dir / "table_ablation_study.tex"
        with open(output_file, 'w') as f:
            f.write(latex)
        
        print(f"✓ Generated {output_file}")
    
    def generate_all(self):
        """Generate all LaTeX tables."""
        print("\nGenerating LaTeX tables...\n")
        self.generate_main_table()
        self.generate_ranking_tables()
        self.generate_model_comparison_table()
        self.generate_loss_comparison_table()
        self.generate_ablation_study_table()
        print(f"\n✓ All tables saved to {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables")
    parser.add_argument("--results", type=str, default="outputs/experiments/summary.csv",
                        help="Path to summary.csv")
    
    args = parser.parse_args()
    
    if not Path(args.results).exists():
        print(f"Error: {args.results} not found")
        return
    
    generator = LatexTableGenerator(args.results)
    generator.generate_all()

if __name__ == "__main__":
    main()
