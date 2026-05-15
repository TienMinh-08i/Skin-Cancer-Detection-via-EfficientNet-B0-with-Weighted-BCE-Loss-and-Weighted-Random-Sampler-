#!/bin/bash
# Run experiment E9: ConvNeXt-Tiny + CB-Focal Loss
cd "$(dirname "$0")/.."
python scripts/run_all_experiments.py --device cuda --experiments E9
