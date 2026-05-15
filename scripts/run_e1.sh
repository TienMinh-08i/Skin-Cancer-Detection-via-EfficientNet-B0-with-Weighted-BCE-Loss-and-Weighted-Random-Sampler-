#!/bin/bash
# Run experiment E1: EfficientNet-B0 + BCE
cd "$(dirname "$0")/.."
python scripts/run_all_experiments.py --device cuda --experiments E1
