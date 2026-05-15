#!/bin/bash
# Quick start script for experiment system
# Run all experiments and generate paper tables

set -e  # Exit on error

echo "================================"
echo "Skin Cancer Classification"
echo "Ablation Study Automation"
echo "================================"
echo ""

cd "$(dirname "$0")" || exit

# Check if configs exist
if [ ! -d "configs/experiments" ] || [ ! -f "configs/experiments/config_E1.yaml" ]; then
    echo "📋 Generating experiment configs..."
    python scripts/generate_experiments.py \
        --base-config configs/config_hpc.yaml \
        --output-dir configs/experiments
    echo "✓ Configs generated"
    echo ""
fi

# Run all experiments
echo "🚀 Starting experiment runs..."
echo "This will take approximately 3-4 hours on a single GPU"
echo ""

python scripts/run_all_experiments.py \
    --device cuda \
    --resume

# Generate tables
echo ""
echo "📊 Generating LaTeX tables..."
python scripts/generate_latex_tables.py \
    --results outputs/experiments/summary.csv

echo ""
echo "✓ All done!"
echo ""
echo "Results saved to: outputs/experiments/"
echo ""
echo "Key files:"
echo "  - summary.csv               (All results)"
echo "  - summary.json              (JSON format)"
echo "  - table_main_results.tex    (Full results table)"
echo "  - ranking_roc_auc.tex       (Ranking by AUC)"
echo "  - ranking_auprc.tex         (Ranking by AUPRC)"
echo "  - table_ablation_study.tex  (Ablation study)"
echo ""
