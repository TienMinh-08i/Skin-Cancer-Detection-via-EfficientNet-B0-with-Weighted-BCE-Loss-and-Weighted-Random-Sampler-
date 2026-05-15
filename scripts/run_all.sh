#!/bin/bash
# Run all experiments E1-E15
cd "$(dirname "$0")/.."
python scripts/run_all_experiments.py --device cuda --resume
