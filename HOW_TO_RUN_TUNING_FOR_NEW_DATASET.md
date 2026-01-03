# Guide: Running Hyperparameter Tuning and Evaluation for a New Dataset

This guide explains how to run hyperparameter tuning for NAMs and EBMs, and create evaluation notebooks for a new dataset.

## Prerequisites

- The dataset must be available via OpenML (with a dataset ID) or be a built-in dataset supported by `data_utils.load_dataset()`
- For OpenML datasets, you need the dataset ID and know whether it's a classification or regression task

### Finding Dataset IDs

**To find available datasets and their OpenML IDs**, check the `notebooks/analysis/dataset_selection.ipynb` notebook. This notebook:
- Lists selected classification and regression datasets from OpenML
- Shows dataset IDs (stored in the DataFrame index)
- Displays dataset names, number of instances, features, and other metadata
- Filters datasets based on size and complexity constraints

Run the notebook to see the selected datasets, or check the output cells to find dataset IDs and their corresponding task types (classification or regression).

---

## Part 1: Hyperparameter Tuning for NAMs

### Step 1: Update the NAM Hyperparameter Tuning Notebook

**File:** `notebooks/hyperparameter_tuning/nam_hyperparameter_tuning.ipynb`

**Cell 1 (Configuration):** Update the following variables:

```python
# Options for DATASET_TYPE: "synthetic" or "openml"
DATASET_TYPE = "openml"  # Change to "synthetic" if using synthetic data

# For synthetic datasets (when DATASET_TYPE == "synthetic"):
SYNTHETIC_TYPE = "nonlinear"  # Options: "linear" or "nonlinear"

# For OpenML datasets (when DATASET_TYPE == "openml"):
OPENML_DATASET_ID = 44966  # ⬅️ CHANGE THIS to your OpenML dataset ID
TASK_TYPE = "regression"  # ⬅️ CHANGE THIS to "classification" or "regression"
```

**Example for a new dataset:**
```python
DATASET_TYPE = "openml"
OPENML_DATASET_ID = 37  # Your new dataset ID
TASK_TYPE = "classification"  # or "regression"
```

### Step 2: Run the Notebook

1. Execute all cells in `notebooks/hyperparameter_tuning/nam_hyperparameter_tuning.ipynb`
2. The notebook will:
   - Load the appropriate configuration from `config/hp_tuning/` (automatically selects `openml_classification.json` or `openml_regression.json` based on `TASK_TYPE`)
   - Perform 50 random search trials
   - Save best hyperparameters to: `results/hyperparameter_tuning/nam/hp_tuning_openml_{ID}_{task_type}/best_hp.json`
   - Train the final model with best hyperparameters (20 splits per fold)

**Note:** The notebook automatically handles the dataset name format: `OpenML_{ID}_{task_type}`

---

## Part 2: Hyperparameter Tuning for EBMs

### Step 1: Update the EBM Hyperparameter Tuning Notebook

**File:** `notebooks/hyperparameter_tuning/ebm_hyperparameter_tuning.ipynb`

**Cell 3 (Configuration):** Update the dataset name:

```python
# Dataset configuration
dataset_name = 'OpenML_44966_regression'  # ⬅️ CHANGE THIS
is_regression = True  # ⬅️ CHANGE THIS (True for regression, False for classification)
```

**Example for a new dataset:**
```python
dataset_name = 'OpenML_37_classification'  # Format: OpenML_{ID}_{task_type}
is_regression = False  # False for classification, True for regression
```


### Step 2: Run the Notebook

1. Execute all cells in `notebooks/hyperparameter_tuning/ebm_hyperparameter_tuning.ipynb`
2. The notebook will:
   - Perform 50 random search trials
   - Save best hyperparameters to: `results/hyperparameter_tuning/ebm/best_hp_{dataset_name}.json`
   - Display summary statistics

**Note:** Unlike NAMs, EBMs don't train the final models in the tuning notebook. Training happens in the evaluation notebook.

---

## Part 3: Create Evaluation Notebooks

After hyperparameter tuning is complete, you need to create evaluation notebooks for both NAMs and EBMs.

### Step 1: Create NAM Evaluation Notebook

**Create new file:** `notebooks/evaluation/NAMs_openml_{ID}_{task_type}.ipynb`

**Template:** Copy from `notebooks/evaluation/NAMs_openml_44966_regression.ipynb` and update:

**Cell 1 (Configuration):**
```python
OPENML_DATASET_ID = 44966  # ⬅️ CHANGE THIS
TASK_TYPE = "regression"  # ⬅️ CHANGE THIS
NUM_FOLDS = 5
NUM_SPLITS = 20
```

**Cell 6 (Gather Predictions):** The `base_logdir` path is automatically constructed from the dataset name, so it should work without changes if you're using OpenML datasets.

**Note:** For NAMs, the training happens in the hyperparameter tuning notebook, so the evaluation notebook only loads and evaluates existing models.

### Step 2: Create EBM Evaluation Notebook

**Create new file:** `notebooks/evaluation/EBMs_openml_{ID}_{task_type}.ipynb`

**Template:** Copy from `notebooks/evaluation/EBMs_openml_44966_regression.ipynb` and update:

**Cell 1 (Configuration):**
```python
OPENML_DATASET_ID = 44966  # ⬅️ CHANGE THIS
TASK_TYPE = "regression"  # ⬅️ CHANGE THIS
NUM_FOLDS = 5
NUM_SPLITS = 20
```

**Cell 5 (Load Hyperparameters):** The path to the best hyperparameters file is automatically constructed, but verify it matches:
```python
best_hp_file = results_dir / f'best_hp_{dataset_name.replace("/", "_").replace(":", "_")}.json'
```

**Cell 7 (Train Models):** This cell will train 20 models per fold automatically. The `base_logdir` is set to:
```python
base_logdir = project_root / 'results' / 'training' / 'ebm' / f'openml_{OPENML_DATASET_ID}_{TASK_TYPE}'
```

**Note:** For EBMs, the evaluation notebook trains the models (20 per fold) and then evaluates them.

---

## Summary Checklist

### For NAMs:
- [ ] Update `OPENML_DATASET_ID` and `TASK_TYPE` in `nam_hyperparameter_tuning.ipynb` (Cell 1)
- [ ] Run the entire `nam_hyperparameter_tuning.ipynb` notebook
- [ ] Create evaluation notebook: `NAMs_openml_{ID}_{task_type}.ipynb`
- [ ] Update `OPENML_DATASET_ID` and `TASK_TYPE` in the evaluation notebook (Cell 1)
- [ ] Run the evaluation notebook

### For EBMs:
- [ ] Update `dataset_name` and `is_regression` in `ebm_hyperparameter_tuning.ipynb` (Cell 3)
- [ ] Run the entire `ebm_hyperparameter_tuning.ipynb` notebook
- [ ] Create evaluation notebook: `EBMs_openml_{ID}_{task_type}.ipynb`
- [ ] Update `OPENML_DATASET_ID` and `TASK_TYPE` in the evaluation notebook (Cell 1)
- [ ] Run the evaluation notebook (this will train 20 models per fold)

---

## File Structure After Completion

```
results/
├── hyperparameter_tuning/
│   ├── nam/
│   │   └── hp_tuning_openml_{ID}_{task_type}/
│   │       └── best_hp.json
│   └── ebm/
│       └── best_hp_OpenML_{ID}_{task_type}.json
└── training/
    ├── nam/
    │   └── openml_{ID}_{task_type}/
    │       └── fold_{1-5}/split_{1-20}/model_0/best_checkpoint/
    └── ebm/
        └── openml_{ID}_{task_type}/
            └── fold_{1-5}/split_{1-20}/ebm_model.pkl
```

---

