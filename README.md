# neural-additive-models-xai-seminar
---
## Installation Steps

### Prerequisites
- Python 3.9.13 (64-bit)

### 1. Install requirements

Install all required dependencies:

```bash
pip install -r requirements.txt
```

## Repository Structure

```
neural-additive-models-xai-seminar/
│
├── src/                          # Source code
│   ├── neural_additive_models/  # NAM implementation (from Google Research + some changes and additions)
│   │   ├── data_utils.py        # Dataset loading utilities
│   │   ├── graph_builder.py     # Model graph construction
│   │   ├── nam_train.py         # Training script
│   │   └── models.py            # Model definitions
│   ├── utils.py                 # Utility functions (plotting, evaluation, etc.)
│   └── hp_tuning_utils.py       # Hyperparameter tuning utilities
│
├── scripts/                     # Python scripts for automation
│   ├── run_nam_hyperparameter_tuning.py    # NAM hyperparameter tuning
│   ├── run_nam_openml_analysis.py          # NAM evaluation/analysis
│   ├── run_ebm_hyperparameter_tuning.py    # EBM hyperparameter tuning
│   ├── run_ebm_openml_analysis.py          # EBM evaluation/analysis
│   └── run_gam_openml_analysis.py          # GAM evaluation/analysis
│
├── notebooks/                   # Jupyter notebooks
│   ├── analysis/                # Analysis and comparison notebooks
│   │   ├── dataset_selection.ipynb
│   │   ├── model_comparison_table_viz.ipynb
│   │   └── three_way_comparison_ebm_nam_gam.ipynb
│   ├── hyperparameter_tuning/   # Hyperparameter tuning notebooks
│   │   ├── nam_hyperparameter_tuning.ipynb
│   │   ├── ebm_hyperparameter_tuning.ipynb
│   │   └── gam_hyperparameter_tuning.ipynb
│   └── evaluation/              # Model evaluation notebooks
│       ├── NAMs_*.ipynb         # NAM evaluation notebooks
│       └── EBMs_*.ipynb         # EBM evaluation notebooks
│
├── config/                      # Configuration files
│   ├── hp_tuning/               # Hyperparameter search space configs
│   │   ├── correlated_linear.json
│   │   ├── correlated_nonlinear.json
│   │   ├── openml_regression.json
│   │   └── openml_classification.json
│   └── training/                # Training parameter configs
│       ├── nam_training_parameters_*.json
│       └── ...
│
├── results/                      # Experiment results
│   ├── hyperparameter_tuning/    # Hyperparameter tuning results
│   │   ├── nam/                  # NAM tuning results
│   │   ├── ebm/                  # EBM tuning results
│   │   └── gam/                  # GAM tuning results
│   ├── training/                 # Trained model checkpoints
│   │   ├── nam/                  # NAM model checkpoints
│   │   ├── ebm/                  # EBM model files
│   │   └── gam/                  # GAM model files
│   └── evaluation/               # Evaluation metrics and summaries
│       ├── *.json                # Performance metrics (RMSE, AUC, etc.)
│       └── plots/                 # Visualization plots (PNG files)
│           ├── nam_*/             # NAM plots by dataset
│           ├── ebm_*/             # EBM plots by dataset
│           └── gam_*/             # GAM plots by dataset
│
├── data/                         # Dataset files
│   └── compas-scores-two-years.csv
│
├── requirements.txt              # Python dependencies
├── README.md
├── compas_run.txt               # PowerShell script for training COMPAS models
├── housing_run.txt              # PowerShell script for training Housing models
└── HOW_TO_RUN_TUNING_FOR_NEW_DATASET.md  # Guide for new datasets
```

### Key Directories

- **`src/`**: Contains all source code including the NAM implementation and utility functions
- **`notebooks/`**: Organized by task (analysis, hyperparameter tuning, evaluation)
- **`config/`**: JSON configuration files for hyperparameter search spaces and training parameters
- **`results/`**: All experimental outputs (tuning results, trained models, evaluation metrics)
- **`data/`**: Local dataset files (most datasets are loaded from OpenML)
- **`scripts/`**: Python scripts for running hyperparameter tuning and evaluation (alternative to notebooks)

## Usage Guide

### Training Models for COMPAS and Housing Datasets

For the COMPAS (Recidivism) and Housing datasets, use the provided PowerShell scripts to train NAM models across multiple data splits.

#### Training COMPAS Models

Run the COMPAS training script (trains models for 20 data splits):

```powershell
. .\compas_run.txt
```

Or execute directly in PowerShell:

```powershell
for ($s = 1; $s -le 20; $s++) {
    python -m neural_additive_models.nam_train `
        --training_epochs=1000 `
        --learning_rate=0.02082 `
        --output_regularization=0.2078 `
        --l2_regularization=0 `
        --batch_size=1024 `
        --logdir=./compas `
        --dataset_name=Recidivism `
        --decay_rate=0.995 `
        --dropout=0.1 `
        --data_split=$s `
        --tf_seed=1 `
        --feature_dropout=0.05 `
        --num_basis_functions=64 `
        --units_multiplier=2 `
        --cross_val=false `
        --max_checkpoints_to_keep=1 `
        --save_checkpoint_every_n_epochs=10 `
        --n_models=1 `
        --num_splits=20 `
        --fold_num=1 `
        --activation=relu `
        --regression=false `
        --debug=false `
        --shallow=false `
        --use_dnn=false `
        --early_stopping_epochs=60
}
```

#### Training Housing Models

Run the Housing training script (trains models for 20 data splits):

```powershell
. .\housing_run.txt
```

### Hyperparameter Tuning

#### NAM Hyperparameter Tuning

**For OpenML Datasets:**

```bash
# Classification example
python scripts/run_nam_hyperparameter_tuning.py \
    --dataset_type openml \
    --dataset_id 31 \
    --task_type classification \
    --n_trials 50 \
    --random_seed 42

# Regression example
python scripts/run_nam_hyperparameter_tuning.py \
    --dataset_type openml \
    --dataset_id 44959 \
    --task_type regression \
    --n_trials 50 \
    --random_seed 42
```

**For Synthetic Datasets (Correlated Linear/Nonlinear):**

```bash
# Correlated linear dataset
python scripts/run_nam_hyperparameter_tuning.py \
    --dataset_type synthetic \
    --synthetic_type linear \
    --n_trials 50 \
    --random_seed 42

# Correlated nonlinear dataset
python scripts/run_nam_hyperparameter_tuning.py \
    --dataset_type synthetic \
    --synthetic_type nonlinear \
    --n_trials 50 \
    --random_seed 42
```

**Additional Options:**
- `--n_trials`: Number of random search trials (default: 50)
- `--random_seed`: Base random seed (default: 42)
- `--run_tag`: Optional suffix for output directory to avoid overwriting
- `--skip_if_exists`: Skip trials if results already exist
- `--per_split_timeout_s`: Optional timeout per split in seconds
- `--do_training`: After tuning, train models using best hyperparameters

**Output:** Best hyperparameters are saved to `results/hyperparameter_tuning/nam/hp_tuning_*/best_hp.json`

#### EBM Hyperparameter Tuning

**For OpenML Datasets:**

```bash
# Classification example
python scripts/run_ebm_hyperparameter_tuning.py \
    --dataset_id 31 \
    --task_type classification \
    --n_trials 50 \
    --random_seed 42

# Regression example
python scripts/run_ebm_hyperparameter_tuning.py \
    --dataset_id 44959 \
    --task_type regression \
    --n_trials 50 \
    --random_seed 42
```

**Additional Options:**
- `--n_trials`: Number of tuning trials (default: 50)
- `--random_seed`: Random seed (default: 42)
- `--test_size`: Test split fraction (default: 0.2)
- `--val_size`: Validation fraction of train_val (default: 0.2)
- `--run_tag`: Optional suffix for output filenames
- `--overwrite`: Allow overwriting existing output files

**Output:** Best hyperparameters are saved to `results/hyperparameter_tuning/ebm/best_hp_*.json`

### Model Evaluation and Analysis

#### NAM Evaluation (OpenML Datasets)

Evaluate trained NAM models and generate performance metrics and plots:

```bash
# Classification example
python scripts/run_nam_openml_analysis.py \
    --dataset_id 31 \
    --task_type classification \
    --num_folds 5 \
    --num_splits 20 \
    --random_state 42

# Regression example
python scripts/run_nam_openml_analysis.py \
    --dataset_id 44959 \
    --task_type regression \
    --num_folds 5 \
    --num_splits 20 \
    --random_state 42
```

**Additional Options:**
- `--num_folds`: Number of CV folds (default: 5)
- `--num_splits`: Number of splits/ensembles per fold (default: 20)
- `--random_state`: Random state for fold creation (default: 42)
- `--y_limits`: Optional y-axis limits for contribution plot (e.g., `--y_limits -0.1 0.1`)
- `--print_every`: Print progress every N splits (default: 5)

**Output:**
- Performance metrics: `results/evaluation/nam_OpenML_<id>_<task_type>_performance.json`
- Plots: `results/evaluation/plots/nam_openml_<id>_<task_type>/`

#### EBM Evaluation (OpenML Datasets)

Evaluate trained EBM models and generate performance metrics and plots:

```bash
# Classification example
python scripts/run_ebm_openml_analysis.py \
    --dataset_id 31 \
    --task_type classification \
    --num_folds 5 \
    --num_splits 20 \
    --random_state 42 \
    --save_plots

# Regression example
python scripts/run_ebm_openml_analysis.py \
    --dataset_id 44959 \
    --task_type regression \
    --num_folds 5 \
    --num_splits 20 \
    --random_state 42 \
    --save_plots
```

**Additional Options:**
- `--num_folds`: Number of folds (default: 5)
- `--num_splits`: Number of splits per fold (default: 20)
- `--random_state`: Random seed for folds (default: 42)
- `--save_plots`: Save plots to `results/evaluation/plots/`
- `--y_limits`: Optional y-axis limits for EBM shape plot
- `--overwrite`: Overwrite plots if they exist
- `--force_retrain`: Force retraining of all models (clears model cache)
- `--print_every`: Print progress every N splits (default: 5)

**Output:**
- Performance metrics: `results/evaluation/ebm_OpenML_<id>_<task_type>_performance.json`
- Plots: `results/evaluation/plots/ebm_OpenML_<id>_<task_type>/`

**Note:** If you get different results than expected, use `--force_retrain` to clear cached models that may have been trained with different hyperparameters.

### Scripts vs Notebooks

This repository provides both **Python scripts** (`scripts/`) and **Jupyter notebooks** (`notebooks/`) for running experiments. The scripts are command-line equivalents of the notebook cells and produce the same results.

**Note:** Instead of running the scripts, you can also run the corresponding notebooks directly to see the results and visualizations inline. The notebooks are located in:
- `notebooks/hyperparameter_tuning/` - For hyperparameter tuning
- `notebooks/evaluation/` - For model evaluation and analysis