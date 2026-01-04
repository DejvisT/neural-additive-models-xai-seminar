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
│   │   ├── graph_builder.py    # Model graph construction
│   │   ├── nam_train.py         # Training script
│   │   └── models.py            # Model definitions
│   ├── utils.py                 # Utility functions (plotting, evaluation, etc.)
│   └── hp_tuning_utils.py       # Hyperparameter tuning utilities
│
├── notebooks/                    # Jupyter notebooks
│   ├── analysis/                # Dataset selection
│   │   └── dataset_selection.ipynb
│   ├── hyperparameter_tuning/   # Hyperparameter tuning notebooks
│   │   ├── nam_hyperparameter_tuning.ipynb
│   │   └── ebm_hyperparameter_tuning.ipynb
│   └── evaluation/              # Model evaluation notebooks
│       ├── NAMs_*.ipynb         # NAM evaluation notebooks
│       └── EBMs_*.ipynb         # EBM evaluation notebooks
│
├── config/                       # Configuration files
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
│   ├── hyperparameter_tuning/   # Hyperparameter tuning results
│   │   ├── nam/                 # NAM tuning results
│   │   └── ebm/                 # EBM tuning results
│   ├── training/                # Trained model checkpoints
│   │   ├── nam/                 # NAM model checkpoints
│   │   └── ebm/                 # EBM model files
│   └── evaluation/              # Evaluation metrics and summaries
│       └── *.json               # Performance metrics (RMSE, AUC, etc.)
│
├── data/                         # Dataset files
│   └── compas-scores-two-years.csv
│
├── requirements.txt              # Python dependencies
├── README.md
└── HOW_TO_RUN_TUNING_FOR_NEW_DATASET.md  # Guide for new datasets
```

### Key Directories

- **`src/`**: Contains all source code including the NAM implementation and utility functions
- **`notebooks/`**: Organized by task (analysis, hyperparameter tuning, evaluation)
- **`config/`**: JSON configuration files for hyperparameter search spaces and training parameters
- **`results/`**: All experimental outputs (tuning results, trained models, evaluation metrics)
- **`data/`**: Local dataset files (most datasets are loaded from OpenML)
