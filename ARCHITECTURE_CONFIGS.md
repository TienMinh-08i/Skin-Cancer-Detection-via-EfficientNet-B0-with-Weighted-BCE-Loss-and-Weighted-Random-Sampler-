"""Configuration Inheritance System - Technical Deep Dive"""

CONFIG HIERARCHY:

┌─────────────────────────────────────────────────────────────┐
│  configs/config_hpc.yaml                                    │
│  (Base Template)                                            │
│  - Default hyperparameters for ALL experiments              │
│  - Dataset paths, image size, optimizer settings            │
│  - Learning rate, batch size, etc.                          │
│                                                             │
│  Example:                                                   │
│    model:                                                   │
│      model_name: efficientnet_b0  ← Will be overridden     │
│      pretrained: true                                       │
│    loss:                                                    │
│      loss_type: bce  ← Will be overridden                  │
│    imbalance:                                               │
│      weighted_sampler_enabled: false                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
                     (merge with)
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  configs/experiments.yaml                                   │
│  (Experiment Definitions)                                   │
│                                                             │
│  experiments:                                               │
│    E1:                                                      │
│      name: "EfficientNet-B0 + BCE"                         │
│      model_name: efficientnet_b0                            │
│      loss_type: bce                                         │
│      weighted_sampler: false                                │
│      cv_folds: 2                                            │
│                                                             │
│    E5:                                                      │
│      name: "EfficientNet-B0 + CB-FL + WS"                  │
│      model_name: efficientnet_b0                            │
│      loss_type: cb_focal                    ← Different    │
│      weighted_sampler: true                 ← Different    │
│      cv_folds: 2                                            │
│                                                             │
│    E9:                                                      │
│      name: "ConvNeXt-Tiny + CB-FL"                         │
│      model_name: convnext_tiny              ← Different    │
│      loss_type: cb_focal                                    │
│      weighted_sampler: false                                │
│      mixup: false                                           │
│                                                             │
│  defaults:                                                  │
│    batch_size: 8                                            │
│    learning_rate: 0.0003                                    │
│    num_epochs: 20                                           │
│    optimizer: adamw                                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
                  (generate_experiments.py)
                   Merges base + definitions
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  configs/experiments/config_E1.yaml                         │
│  (Generated Config - E1)                                    │
│                                                             │
│  Inherits ALL from base + overrides:                        │
│    model:                                                   │
│      model_name: efficientnet_b0            ← From E1       │
│    loss:                                                    │
│      loss_type: bce                         ← From E1       │
│    imbalance:                                               │
│      weighted_sampler_enabled: false        ← From E1       │
│    training:                                                │
│      batch_size: 8                          ← From defaults │
│      num_epochs: 20                         ← From defaults │
│    optimizer:                                               │
│      learning_rate: 0.0003                  ← From defaults │
│                                                             │
│    (Plus all other settings from base)                      │
│    experiment:                                              │
│      id: E1                                                 │
│      name: "EfficientNet-B0 + BCE"                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  configs/experiments/config_E5.yaml                         │
│  (Generated Config - E5)                                    │
│                                                             │
│  Inherits ALL from base + overrides:                        │
│    model:                                                   │
│      model_name: efficientnet_b0            ← From E5       │
│    loss:                                                    │
│      loss_type: cb_focal                    ← From E5       │
│    imbalance:                                               │
│      weighted_sampler_enabled: true         ← From E5       │
│      mixup_enabled: false                   ← From E5       │
│      cutmix_enabled: false                  ← From E5       │
│    training:                                                │
│      batch_size: 8                          ← From defaults │
│      num_epochs: 20                         ← From defaults │
│    optimizer:                                               │
│      learning_rate: 0.0003                  ← From defaults │
│    (Plus all other settings from base)                      │
│    experiment:                                              │
│      id: E5                                                 │
│      name: "EfficientNet-B0 + CB-FL + WS"                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  configs/experiments/config_E9.yaml                         │
│  (Generated Config - E9)                                    │
│                                                             │
│  Inherits ALL from base + overrides:                        │
│    model:                                                   │
│      model_name: convnext_tiny              ← From E9       │
│    loss:                                                    │
│      loss_type: cb_focal                    ← From E9       │
│    imbalance:                                               │
│      weighted_sampler_enabled: false        ← From E9       │
│      mixup_enabled: false                   ← From E9       │
│      cutmix_enabled: false                  ← From E9       │
│    training:                                                │
│      batch_size: 8                          ← From defaults │
│      num_epochs: 20                         ← From defaults │
│    optimizer:                                               │
│      learning_rate: 0.0003                  ← From defaults │
│    (Plus all other settings from base)                      │
│    experiment:                                              │
│      id: E9                                                 │
│      name: "ConvNeXt-Tiny + CB-FL"                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
                    (main_train.py)
                  Loads generated config
                   Trains the model
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  outputs/experiments/E1/cv_results/                         │
│  - fold_0_best.pt                                           │
│  - fold_1_best.pt                                           │
│  - cv_results.csv (metrics)                                 │
│                                                             │
│  outputs/experiments/E9/cv_results/                         │
│  - fold_0_best.pt                                           │
│  - fold_1_best.pt                                           │
│  - cv_results.csv (metrics)                                 │
│                                                             │
│  outputs/experiments/run_log.json                           │
│  {                                                          │
│    "E1": {"status": "completed", "elapsed_time": 480},     │
│    "E9": {"status": "completed", "elapsed_time": 720}      │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
                (run_all_experiments.py)
                 Collects all results
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  outputs/experiments/summary.csv                            │
│                                                             │
│  experiment_id,experiment_name,model,loss,roc_auc,...      │
│  E1,EfficientNet-B0 + BCE,efficientnet_b0,bce,0.8234,...  │
│  E2,...                                                     │
│  E5,EfficientNet-B0 + CB-FL + WS,efficientnet_b0,...      │
│  E9,ConvNeXt-Tiny + CB-FL,convnext_tiny,cb_focal,...      │
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
           (generate_latex_tables.py)
            Creates paper tables
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  outputs/experiments/                                       │
│  - table_main_results.tex                                   │
│  - ranking_roc_auc.tex                                      │
│  - ranking_auprc.tex                                        │
│  - table_model_comparison.tex                               │
│  - table_ablation_study.tex                                 │
│  - ...                                                      │
└─────────────────────────────────────────────────────────────┘


INHERITANCE RULES:

1. Base Config (config_hpc.yaml)
   - Provides ALL settings with defaults
   - Used as starting point for every experiment

2. Experiment Definition (experiments.yaml)
   - Only specifies what's DIFFERENT from base
   - Model name, loss, imbalance techniques
   - Inherits everything else from base

3. Generated Config (config_E*.yaml)
   - = Base Config + Experiment Overrides
   - Complete, self-contained configuration
   - Ready to pass to training script

4. Defaults Section (experiments.yaml)
   - Shared hyperparameters for all experiments
   - batch_size, learning_rate, num_epochs, etc.
   - Override in individual experiment if needed

EXAMPLE MODIFICATION SCENARIOS:

Scenario 1: Change batch size for ALL experiments
───────────────────────────────────────────────────
  1. Edit configs/experiments.yaml
     defaults:
       batch_size: 16  ← Change from 8
  
  2. Regenerate configs
     python scripts/generate_experiments.py
  
  3. All config_E*.yaml now have batch_size: 16
  
  4. Re-run experiments
     python scripts/run_all_experiments.py --force

Scenario 2: Change learning rate for E1 only
───────────────────────────────────────────────
  1. Edit configs/experiments.yaml
     E1:
       name: "EfficientNet-B0 + BCE"
       model_name: efficientnet_b0
       loss_type: bce
       learning_rate: 0.0001  ← Add this
  
  2. Regenerate configs
     python scripts/generate_experiments.py
  
  3. config_E1.yaml now has learning_rate: 0.0001
     (Other experiments keep default 0.0003)
  
  4. Re-run E1
     python scripts/run_all_experiments.py --experiments E1 --force

Scenario 3: Add new experiment (E16)
───────────────────────────────────────
  1. Edit configs/experiments.yaml
     experiments:
       ...
       E16:
         name: "Custom Experiment"
         model_name: densenet121
         loss_type: cb_focal
         weighted_sampler: true
         cv_folds: 2
         num_epochs: 20
  
  2. Regenerate configs
     python scripts/generate_experiments.py
  
  3. config_E16.yaml is created
  
  4. Run E16
     python scripts/run_all_experiments.py --experiments E16

Scenario 4: Modify experiment E11 (add CutMix)
──────────────────────────────────────────────
  1. Current E11:
     E11:
       name: "ConvNeXt-Tiny + CB-FL + Mixup"
       model_name: convnext_tiny
       loss_type: cb_focal
       mixup: true
     
     Change to:
     E11:
       name: "ConvNeXt-Tiny + CB-FL + Mixup + CutMix"
       model_name: convnext_tiny
       loss_type: cb_focal
       mixup: true
       cutmix: true  ← Add
  
  2. Regenerate configs
     python scripts/generate_experiments.py
  
  3. config_E11.yaml updated with cutmix_enabled: true
  
  4. Re-run E11 (with --force to override old results)
     python scripts/run_all_experiments.py --experiments E11 --force


EXECUTION FLOW:

Step 1: Generate
  ┌──────────────────────┐
  │ configs/config_hpc.  │
  │ yaml                 │
  └──────────────────────┘
             ↓
  ┌──────────────────────┐
  │ configs/experiments. │
  │ yaml                 │
  └──────────────────────┘
             ↓
  python scripts/generate_experiments.py
             ↓
  ┌──────────────────────────────────────┐
  │ configs/experiments/config_E1.yaml    │
  │ configs/experiments/config_E2.yaml    │
  │ ...                                  │
  │ configs/experiments/config_E15.yaml   │
  │ configs/experiments/experiment_      │
  │ index.yaml                           │
  └──────────────────────────────────────┘

Step 2: Execute
  python scripts/run_all_experiments.py --device cuda --resume
             ↓
  For each E1-E15:
    Load config_E*.yaml
    Run: python scripts/main_train.py --config config_E*.yaml
    Save results to outputs/experiments/E*/

Step 3: Aggregate
  python scripts/run_all_experiments.py --collect-only
             ↓
  Collects all cv_results.csv from each experiment
  Aggregates into outputs/experiments/summary.csv

Step 4: Generate Tables
  python scripts/generate_latex_tables.py
             ↓
  Reads summary.csv
  Creates LaTeX tables for paper


KEY BENEFITS:

1. DRY (Don't Repeat Yourself)
   - Base config written once
   - Only differences specified in experiment definitions
   - Easy to maintain consistency

2. Transparency
   - Can always see what's different between experiments
   - No hidden defaults

3. Reproducibility
   - Each experiment has complete config file
   - Can re-generate identical setup months later

4. Scalability
   - Adding E16, E17, E18 is trivial
   - Just add to experiments.yaml

5. Version Control
   - Changes to defaults tracked in experiments.yaml
   - Complete history of modifications
   - Can revert to previous experimental setup

6. Consistency
   - All experiments trained with same hyperparameters
   - Fair comparison
   - Only variables are model and loss/sampling
