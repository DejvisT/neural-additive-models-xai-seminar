import neural_additive_models.data_utils as data_utils
import numpy as np
import os
import re
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from neural_additive_models.models import NAM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import time
import subprocess
from pathlib import Path
import neural_additive_models.graph_builder as graph_builder
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import seaborn as sns

def load_col_min_max(dataset_name, correlated_n=None, rho=None, seed=None):
  """Loads the dataset according to the `dataset_name` passed."""
  if dataset_name == 'Housing':
    dataset = data_utils.load_california_housing_data()
  elif dataset_name == 'BreastCancer':
    dataset = data_utils.load_breast_data()
  elif dataset_name == 'Recidivism':
    dataset = data_utils.load_recidivism_data()
  elif dataset_name == 'Fico':
    dataset = data_utils.load_fico_score_data()
  elif dataset_name == 'Mimic2':
    dataset = load_mimic2_data()
  elif dataset_name == 'Credit':
    dataset = data_utils.load_credit_data()
  elif dataset_name == 'Correlated_linear':
    dataset = data_utils.load_correlated_linear_data(correlated_n, rho, seed)
  elif dataset_name == 'Correlated_nonlinear':
    dataset = data_utils.load_correlated_nonlinear_data(correlated_n, rho, seed)
  else:
    raise ValueError('{} not found!'.format(dataset_name))

  if 'full' in dataset:
    dataset = dataset['full']
  x = dataset['X']
  col_min_max = {}
  for col in x:
    unique_vals = x[col].unique()
    col_min_max[col] = (np.min(unique_vals), np.max(unique_vals))
  return col_min_max


def inverse_min_max_scaler(x, min_val, max_val):
  return (x + 1)/2 * (max_val - min_val) + min_val 



def load_nam_checkpoint(ckpt_dir: str, hyperparameters=None):
    """
    Load a NAM (Neural Additive Model) from a TensorFlow v1 checkpoint directory.

    Args:
        ckpt_dir (str): Path to the checkpoint directory containing .index and .data files.
        hyperparameters (dict, optional): Dict with keys 'dropout', 'feature_dropout', 'activation', 'shallow'.
                                         If None, uses defaults (dropout=0.0, feature_dropout=0.0, activation='relu', shallow=False).

    Returns:
        (nam, sess): A tuple containing the restored NAM model and active TensorFlow session.
    """
    # --- Locate checkpoint ---
    ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt_path is None:
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.index')]
        if not ckpt_files:
            raise FileNotFoundError(f"No valid checkpoint found in {ckpt_dir}")
        name = ckpt_files[0].split('.index')[0]
        ckpt_path = os.path.join(ckpt_dir, name)
    print(f"Using checkpoint: {ckpt_path}")

    # --- Read variable shapes to reconstruct model architecture ---
    reader = tf.train.NewCheckpointReader(ckpt_path)
    var_map = reader.get_variable_to_shape_map()

    units_by_idx = {}
    for name, shape in var_map.items():
        m = re.match(r"^model_0/activation_layer_(\d+)/beta$", name)
        if m:
            units_by_idx[int(m.group(1))] = shape[1]

    if not units_by_idx:
        raise ValueError("Could not infer unit shapes from checkpoint metadata.")

    num_units_list = [units_by_idx[i] for i in sorted(units_by_idx)]
    num_inputs = len(num_units_list)

    print("Feature widths:", num_units_list)
    print("Num input features:", num_inputs)

    # --- Auto-detect shallow from checkpoint variables ---
    # Check if checkpoint has hidden layer variables (h1_0/bias, etc.) - indicates non-shallow
    has_hidden_layers = any('h1_' in name or 'h2_' in name for name in var_map.keys())
    detected_shallow = not has_hidden_layers
    
    # --- Build the model with hyperparameters ---
    tf.reset_default_graph()
    hp = hyperparameters or {}
    
    # Use detected shallow if not explicitly provided, otherwise use provided value
    shallow_value = hp.get('shallow')
    if shallow_value is None:
        shallow_value = detected_shallow
        print(f"Auto-detected shallow={shallow_value} from checkpoint structure")
    else:
        if shallow_value != detected_shallow:
            print(f"⚠️  WARNING: Provided shallow={shallow_value} but checkpoint suggests shallow={detected_shallow}")
            print(f"   Using provided value: shallow={shallow_value}")
    
    nam = NAM(
        num_inputs=num_inputs,
        num_units=num_units_list,
        dropout=hp.get('dropout', 0.0),
        feature_dropout=hp.get('feature_dropout', 0.0),
        activation=hp.get('activation', 'relu'),
        shallow=shallow_value,
        trainable=False,
        name_scope='model_0'
    )
    _ = nam(np.zeros((1, num_inputs), np.float32), training=False)

    # --- Restore weights ---
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)
    print("✅ Restored NAM from checkpoint.")

    return nam, sess


def partition(lst, batch_size):
    lst_len = len(lst)
    index = 0
    while index < lst_len:
        yield lst[index: batch_size + index]
        index += batch_size


def generate_predictions(gen, nn_model, sess):
    """Run predictions batch-by-batch inside a TF1 session."""
    y_pred = []
    while True:
        try:
            x = next(gen)
            pred = sess.run(nn_model(x, training=False))
            y_pred.extend(pred)
        except StopIteration:
            break
    return np.array(y_pred)


def get_test_predictions(nn_model, x_test, sess, batch_size=1024):
    num_samples = x_test.shape[0]
    preds = []
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        batch = x_test[start:end]
        preds.append(sess.run(nn_model(batch, training=False)))
    return np.concatenate(preds, axis=0)


def get_feature_predictions(nn_model, dataset_name, sess, chunk_size=50000):
    """Compute feature predictions for all unique values safely in chunks."""
    unique_features = compute_features(dataset_name)
    feature_predictions = []

    for c, vals in enumerate(unique_features):
        preds_all = []
        n = vals.shape[0]
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            batch = vals[start:end]
            preds = sess.run(nn_model.feature_nns[c](batch, training=nn_model._false))
            preds_all.append(preds)
        feature_predictions.append(np.concatenate(preds_all, axis=0))
    return feature_predictions


def compute_features(dataset_name):
    x_data, _, _ = data_utils.load_dataset(dataset_name)
    n_features = x_data.shape[1]
    unique_features = []

    for i in range(n_features):
        col = np.ascontiguousarray(x_data[:, i])
        # Sort first, then unique -> less memory than np.unique on unsorted
        col.sort()
        uniq = np.unique(col)
        unique_features.append(uniq.reshape(-1, 1))
    return unique_features


def prepare_feature_arrays(data_x, column_names, col_min_max, inverse_min_max_scaler):
    """Split scaled features and inverse transform to original space."""
    num_features = data_x.shape[1]
    single_features = np.split(data_x, num_features, axis=1)
    unique_features = [np.unique(x, axis=0) for x in single_features]

    single_features_original = {}
    unique_features_original = {}

    for i, col in enumerate(column_names):
        min_val, max_val = col_min_max[col]
        unique_features_original[col] = inverse_min_max_scaler(unique_features[i][:, 0], min_val, max_val)
        single_features_original[col] = inverse_min_max_scaler(single_features[i][:, 0], min_val, max_val)

    return single_features_original, unique_features_original, unique_features


def get_dataset_config(dataset_name, column_names):
    """Return COL_NAMES, FEATURE_LABEL_MAPPING, and CATEGORICAL_NAMES for dataset."""
    FEATURE_LABEL_MAPPING = {
        'Recidivism': {
            'race': (['African\nAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Native\nAmerican', 'Other'], 90),
            'sex': (['Female', 'Male'], None)
        },
        'Mimic2': {
            'AIDS': (['No', 'Yes'], None),
            'Lymphoma': (['No', 'Yes'], None),
            'MetastaticCancer': (['No', 'Yes'], None)
        },
        'Fico': {},
        'Housing': {},
        'Correlated_linear': {},
        'Correlated_nonlinear': {},
        'Credit':{}
    }

    COL_NAMES = {
        'Recidivism': {
            'age': 'Age', 'race': 'Race', 'sex': 'Gender',
            'priors_count': 'Prior Counts', 'length_of_stay': 'Length of Stay',
            'c_charge_degree': 'Charge Degree'
        },
        'Housing': {
            'MedInc': 'Median Income', 'HouseAge': 'Median House Age',
            'AveRooms': '# Avg Rooms', 'AveBedrms': '# Avg Bedrooms',
            'Population': 'Block Population', 'AveOccup': '# Avg Occupancy',
            'Latitude': 'Latitude', 'Longitude': 'Longitude'
        },
        'Fico':  {
            'MSinceOldestTradeOpen': 'Months Since Oldest Trade Open',
            'MSinceMostRecentTradeOpen':	'Months Since Most Recent Trade',
            'AverageMInFile':	'Average Months in File',
            'NumSatisfactoryTrades': '# Satisfactory Trades',	
            'NumTrades60Ever2DerogPubRec': '# Trades 60+ Ever',	
            'NumTrades90Ever2DerogPubRec':	'# Trades 90+ Ever',	
            'NumTotalTrades': '# Total Trades',
            'NumTradesOpeninLast12M': '# Trades Open in Last 12 Months',
            'PercentTradesNeverDelq':	'% Trades Never Delinquent',
            'MSinceMostRecentDelq':	'Months Since Most Recent Delinquency',	
            'MaxDelq2PublicRecLast12M':	'Max Delq/Public Records Last Year',
            'MaxDelqEver':	'Max Delinquency Ever',
            'PercentInstallTrades':	'% Installment Trades',	
            'NetFractionInstallBurden':	'Net Fraction Installment Burden',
            'NumInstallTradesWBalance': 'Number Installment Trades with Balance',	
            'MSinceMostRecentInqexcl7days':	'Months Since Most Recent Inquiry\n excluding 7 days',	
            'NumInqLast6M': '# Inquiries in Last 6 Months',
            'NumInqLast6Mexcl7days': '# Inquiries in Last 6 Months \n excluding 7 days',
            'NetFractionRevolvingBurden':	'Net Fraction Revolving Burden',
            'NumRevolvingTradesWBalance':	'# Revolving Trades with Balance',	
            'NumBank2NatlTradesWHighUtilization':	'# Bank/Natl Trades with high utilization ratio',	
            'PercentTradesWBalance': '% Trades with Balance',
            'delinquent': 'Delinquent',
            'inquiry': 'Inquiry',
        }
    }

    if dataset_name in ['Credit', 'Mimic2', 'Correlated_linear', 'Correlated_nonlinear']:
        COL_NAMES[dataset_name] = {x: x for x in column_names}

    if dataset_name in ['Housing', 'Credit', 'Correlated_linear', 'Correlated_nonlinear']:
        categorical_names = []
    elif dataset_name == 'Mimic2':
        categorical_names = ['AIDS','AdmissionType','GCS','Lymphoma','Temperature','MetastaticCancer','Renal']
    elif dataset_name == 'Recidivism':
        categorical_names = ['race','sex','c_charge_degree']
    elif dataset_name == 'Fico':
        categorical_names = ['delinquent','inquiry','MaxDelqEver','MaxDelq2PublicRecLast12M']
    else:
        raise ValueError(f"{dataset_name} not found!")

    return COL_NAMES, FEATURE_LABEL_MAPPING, categorical_names


# Correlated synthetic helpers

def _format_rho(rho: float) -> str:
  """Formats rho for folder names like rho_0, rho_03, rho_095."""
  if rho == 0 or rho == 0.0:
    return '0'
  s = str(rho)
  if '.' not in s:
    return s
  frac = s.split('.', 1)[1].rstrip('0')
  if frac == '':
    return '0'
  decimals = len(frac)
  scale = 10 ** decimals
  n = int(round(float(rho) * scale))
  width = decimals + 1
  return f"{n:0{width}d}"


def _to_flag(v):
  if v is None:
    return 'None'
  if isinstance(v, bool):
    return 'true' if v else 'false'
  return str(v)


def build_cmd(params: dict) -> str:
  """Build `python -m neural_additive_models.nam_train ...` command from params."""
  cmd = ['python', '-m', 'neural_additive_models.nam_train']
  for k, v in params.items():
    cmd.append(f"--{k}={_to_flag(v)}")
  return ' '.join(cmd)


def run_nam_train_sweep(
    rhos,
    seeds,
    fixed: dict,
    best_hp: dict,
    base_logdir: Path,
    skip_if_checkpoint_exists: bool = True,
    per_run_timeout_s: float = 3600,
    save_summary: bool = True,
    summary_filename: str = "sweep_summary.json",
):
  """Run `nam_train` over a (rho, seed) grid.

  Returns:
    - runs: list of per-run dicts
    - summary_path: where the JSON summary was written
  """
  runs = []
  for rho in rhos:
    rho_dir = base_logdir / f"rho_{_format_rho(rho)}"
    for seed in seeds:
      run_dir = rho_dir / f"seed_{seed}"
      run_dir.mkdir(parents=True, exist_ok=True)
      logdir = str(run_dir)

      # checkpoint existence check (nam_train nests fold_1/split_1/...)
      best_ckpt_dir = (
          run_dir
          / f"fold_{fixed['fold_num']}"
          / f"split_{fixed['data_split']}"
          / "model_0"
          / "best_checkpoint"
      )
      ckpt_marker = best_ckpt_dir / "checkpoint"
      if skip_if_checkpoint_exists and ckpt_marker.exists():
        continue

      params = {
          **fixed,
          **best_hp,
          "correlated_rho": rho,
          "correlated_seed": seed,
          "logdir": logdir,
      }

      cmd = build_cmd(params)
      print(f"\n[rho={rho}, seed={seed}]\n{cmd}")
      
      # Set PYTHONPATH to include src/ so neural_additive_models can be found
      project_root = Path(__file__).parent.parent
      src_path = str(project_root / 'src')
      
      # Get existing PYTHONPATH if it exists
      existing_pythonpath = os.environ.get('PYTHONPATH', '')
      if existing_pythonpath:
        pythonpath = f"{src_path}{os.pathsep}{existing_pythonpath}"
      else:
        pythonpath = src_path
      
      # Set environment variable in the subprocess environment
      env = os.environ.copy()
      env['PYTHONPATH'] = pythonpath
      
      t0 = time.time()
      res = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=per_run_timeout_s,
        env=env,
      )
      dt = time.time() - t0

      runs.append(
          {
              "rho": float(rho),
              "seed": int(seed),
              "logdir": logdir,
              "returncode": int(res.returncode),
              "train_time_s": float(dt),
              "stdout_tail": res.stdout[-4000:] if res.stdout else "",
              "stderr_tail": res.stderr[-4000:] if res.stderr else "",
          }
      )

      if res.returncode != 0:
        print("  FAILED")
      else:
        print(f"  OK in {dt:.1f}s")

  summary_path = base_logdir / summary_filename
  if save_summary:
    with open(summary_path, "w") as f:
      json.dump(
          {
              "best_hp": best_hp,
              "fixed": fixed,
              "rhos": list(rhos),
              "seeds": list(seeds),
              "runs": runs,
          },
          f,
          indent=2,
      )
    print("Saved summary:", summary_path)
    print("Total runs recorded:", len(runs))

  return runs, summary_path


def get_hyperparameter_tuning_indices(
    data_x,
    data_y,
    test_size: float = 0.2,
    val_size: float = 0.2,
    is_regression: bool = True,
    random_state: int = 42,
    combine_train_val: bool = False,
):
  """Get train/test indices matching hyperparameter_tuning=true split.
  
  This function recreates the exact same split used during training when
  hyperparameter_tuning=true, returning indices instead of data arrays.
  
  Args:
    data_x: Full dataset features
    data_y: Full dataset labels
    test_size: Proportion for test set (default: 0.2)
    val_size: Proportion for validation set (default: 0.2)
    is_regression: Whether this is regression (affects stratification)
    random_state: Random seed (must match training config's hp_random_state)
    combine_train_val: If True, combine train+val indices (for final training)
  
  Returns:
    train_indices: Array of indices for training (or train+val if combine_train_val=True)
    test_indices: Array of indices for testing
  """
  from sklearn.model_selection import train_test_split
  
  # Create indices array
  n_samples = len(data_x)
  all_indices = np.arange(n_samples)
  
  # First split: separate test set
  can_stratify = not is_regression and (data_y.dtype.kind in ['i', 'u'] or 
                                        np.all(data_y == data_y.astype(int)))
  
  train_val_indices, test_indices = train_test_split(
      all_indices,
      test_size=test_size,
      random_state=random_state,
      stratify=data_y if can_stratify else None
  )
  
  # Second split: separate validation set from remaining data
  val_size_adjusted = val_size / (1 - test_size)
  train_indices, val_indices = train_test_split(
      train_val_indices,
      test_size=val_size_adjusted,
      random_state=random_state,
      stratify=data_y[train_val_indices] if can_stratify else None
  )
  
  # If combine_train_val, merge train and val indices
  if combine_train_val:
    train_indices = np.concatenate([train_indices, val_indices])
  
  return train_indices, test_indices


def create_fold_indices(
    data_x,
    data_y,
    num_folds: int = 5,
    is_regression: bool = True,
    random_state: int = 42,
):
  """Create train and test indices for all folds in K-Fold cross validation.
  
  Args:
    data_x: Training data features.
    data_y: Training data labels/targets.
    num_folds: Number of folds for cross-validation.
    is_regression: If True, use KFold. If False, use StratifiedKFold.
    random_state: Random seed for reproducibility.
  
  Returns:
    fold_train_indices: List of arrays, training indices for each fold.
    fold_test_indices: List of arrays, test indices for each fold.
  """
  if is_regression:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
  else:
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
  
  all_splits = list(kfold.split(data_x, data_y))
  
  fold_train_indices = []
  fold_test_indices = []
  
  for fold_num in range(1, num_folds + 1):
    train_idx, test_idx = all_splits[fold_num - 1]
    fold_train_indices.append(train_idx)
    fold_test_indices.append(test_idx)
  
  return fold_train_indices, fold_test_indices


def analyze_correlated_shape_functions(
    dataset_name: str,
    rho: float,
    seeds,
    fixed: dict,
    best_hp: dict,
    fold_num_trained: int = 1,
    n_folds: int = 2,
    split_num_trained: int = 1,
    random_state: int = 42,
    n_grid: int = 400,
    ckpt_base_dir=None,
    print_every: int = 5,
):
  """Load per-seed checkpoints and compute per-feature predictions.

  Returns:
    - seed_rmses
    - all_hist_data
    - all_mean_pred
    - unique_features_original
    - single_features_original
    - column_names
  """

  if ckpt_base_dir is None:
    ckpt_base_dir = f"./{dataset_name.lower()}"

  # Hyperparams for checkpoint restore
  hp_for_restore = dict(best_hp)
  hp_for_restore["activation"] = fixed["activation"]
  hp_for_restore["shallow"] = fixed["shallow"]
  hp_for_restore["dropout"] = float(hp_for_restore.get("dropout", 0.0))
  hp_for_restore["feature_dropout"] = float(hp_for_restore.get("feature_dropout", 0.0))

  x0, _, column_names = data_utils.load_dataset(
      dataset_name,
      correlated_n=fixed["correlated_n"],
      correlated_rho=rho,
      correlated_seed=0,
  )

  # Load min/max from raw data for inverse transformation
  col_min_max = load_col_min_max(
      dataset_name,
      correlated_n=fixed["correlated_n"],
      rho=rho,
      seed=0,
  )

  unique_features = []
  unique_features_original = {}
  single_features_original = {}
  for i, col in enumerate(column_names):
    grid = np.linspace(-1.0, 1.0, n_grid, dtype=np.float32)
    unique_features.append(grid.reshape(-1, 1))

    min_val, max_val = col_min_max[col]
    unique_features_original[col] = inverse_min_max_scaler(grid, min_val, max_val)
    single_features_original[col] = inverse_min_max_scaler(x0[:, i], min_val, max_val)

  all_hist_data = []
  all_mean_pred = []
  seed_rmses = []

  for seed in seeds:
    if print_every and seed % int(print_every) == 0:
      print(f"Processing seed {seed}...")

    data_x, data_y, _ = data_utils.load_dataset(
        dataset_name,
        correlated_n=fixed["correlated_n"],
        correlated_rho=rho,
        correlated_seed=seed,
    )

    # If hyperparameter_tuning=true, use get_train_val_test_split
    # Otherwise, use KFold
    if fixed.get("hyperparameter_tuning", False):
      # Use the same split as training (train/val/test split)
      train_indices, test_idx = get_hyperparameter_tuning_indices(
          data_x,
          data_y,
          test_size=fixed.get("test_size", 0.2),
          val_size=fixed.get("val_size", 0.2),
          is_regression=fixed.get("regression", True),
          random_state=fixed.get("hp_random_state", 42),
          combine_train_val=fixed.get("combine_train_val_for_final_training", False),
      )
    else:
      # Use KFold (original behavior for non-hyperparameter-tuning mode)
      kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
      _, test_idx = list(kfold.split(data_x, data_y))[fold_num_trained - 1]

    ckpt_dir = (
        f"{ckpt_base_dir}/rho_{_format_rho(rho)}/seed_{seed}/"
        f"fold_{fold_num_trained}/split_{split_num_trained}/model_0/best_checkpoint"
    )
    nam, sess = load_nam_checkpoint(ckpt_dir, hyperparameters=hp_for_restore)

    preds_full = get_test_predictions(nam, data_x.astype(np.float32), sess)
    rmse = graph_builder.calculate_metric(data_y[test_idx], preds_full[test_idx], regression=True)
    seed_rmses.append(float(rmse))

    feature_predictions = []
    for i, vals in enumerate(unique_features):
      preds = sess.run(nam.feature_nns[i](vals, training=nam._false))
      feature_predictions.append(preds.reshape(-1))

    avg_hist_data, mean_pred, _ = compute_mean_predictions(
        data_x, column_names, unique_features, feature_predictions
    )
    all_hist_data.append(avg_hist_data)
    all_mean_pred.append(mean_pred)
    sess.close()

  split_method = "train/val/test" if fixed.get("hyperparameter_tuning", False) else f"{n_folds}-fold CV"
  print(
      f"\nRMSE over seeds (rho={rho}, {split_method}, fold_num={fold_num_trained}): "
      f"{np.mean(seed_rmses):.4f} ± {np.std(seed_rmses):.4f}"
  )

  return (
      seed_rmses,
      all_hist_data,
      all_mean_pred,
      unique_features_original,
      single_features_original,
      column_names,
  )


def evaluate_correlated_models(
    dataset_name: str,
    rhos,
    seeds,
    fixed: dict,
    best_hp: dict,
    compute_shape_rmse: bool = False,
    true_coefs=None,
    fold_num_trained=None,
    n_folds=None,
    split_num_trained=None,
    random_state: int = 42,
    n_grid: int = 400,
    ckpt_base_dir=None,
):
  """Evaluate correlated models: compute test RMSE and optionally shape RMSE.
  
  Supports both Correlated_linear and Correlated_nonlinear datasets.
  Loads each model once and computes both metrics.
  
  Args:
    dataset_name: Must be "Correlated_linear" or "Correlated_nonlinear"
    compute_shape_rmse: If True, also compute shape function RMSE against ground truth
      (only supported for Correlated_linear and Correlated_nonlinear)
    true_coefs: Required if compute_shape_rmse=True and dataset_name="Correlated_linear"
      (not needed for Correlated_nonlinear as truth functions are hardcoded)
    All other args: Standard evaluation parameters
  
  Returns:
    If compute_shape_rmse=False:
      - rmse_by_rho_seed: dict[rho][seed] -> test RMSE
      - rmse_df: DataFrame with columns [rho, seed, rmse]
      - summary: DataFrame grouped by rho with mean/std/count of rmse
      - rho_rmses: list of np.ndarray (rmse values per rho)
    If compute_shape_rmse=True:
      - Same as above, plus:
      - shape_df: DataFrame with columns [rho, seed, feature, coef, mse, rmse]
      - shape_by_rho_seed_feature: dict[rho][seed][feature] -> shape RMSE
  """

  if dataset_name not in {"Correlated_linear", "Correlated_nonlinear"}:
    raise ValueError(f"dataset_name must be 'Correlated_linear' or 'Correlated_nonlinear', got '{dataset_name}'")

  if compute_shape_rmse:
    if dataset_name == "Correlated_linear" and true_coefs is None:
      raise ValueError("For Correlated_linear with compute_shape_rmse=True, you must pass true_coefs")

  if fold_num_trained is None:
    fold_num_trained = int(fixed.get("fold_num", 1))
  if n_folds is None:
    n_folds = int(fixed.get("n_folds", 2))
  if split_num_trained is None:
    split_num_trained = int(fixed.get("data_split", 1))
  if ckpt_base_dir is None:
    ckpt_base_dir = f"./{dataset_name.lower()}"

  hp_for_restore = dict(best_hp)
  hp_for_restore["activation"] = fixed["activation"]
  hp_for_restore["shallow"] = fixed["shallow"]
  hp_for_restore["dropout"] = float(hp_for_restore.get("dropout", 0.0))
  hp_for_restore["feature_dropout"] = float(hp_for_restore.get("feature_dropout", 0.0))

  # Prepare shape RMSE computation
  if compute_shape_rmse:
    x_scaled_grid = np.linspace(-1.0, 1.0, n_grid, dtype=np.float32)
    def _truth_fn_for_col(col_name: str):
      if dataset_name == "Correlated_linear":
        coef = float(true_coefs.get(col_name, 0.0))
        return (lambda x, c=coef: c * x), coef
      if col_name == "X1":
        return (lambda x: 2.0 * np.sin(2 * np.pi * x)), np.nan
      if col_name == "X2":
        return (lambda x: 1.5 * (x**2 - 1.0)), np.nan
      if col_name == "X3":
        return (lambda x: np.log(1.0 + x**2)), np.nan
      if col_name == "X4":
        return (lambda x: 2.0 * (x**2 - 0.5)), np.nan
      return (lambda x: 0.0 * x), np.nan

  rmse_by_rho_seed = {}
  rmse_rows = []
  shape_rows = []
  shape_by_rho_seed_feature = {}

  for rho_val in rhos:
    rmse_by_rho_seed[float(rho_val)] = {}
    if compute_shape_rmse:
      shape_by_rho_seed_feature[float(rho_val)] = {}

    for seed in seeds:
      if compute_shape_rmse:
        shape_by_rho_seed_feature[float(rho_val)][int(seed)] = {}

      # Load dataset
      data_x, data_y, column_names = data_utils.load_dataset(
          dataset_name,
          correlated_n=fixed["correlated_n"],
          correlated_rho=rho_val,
          correlated_seed=seed,
      )

      # Get test indices (use same split method as training)
      if fixed.get("hyperparameter_tuning", False):
        train_indices, test_idx = get_hyperparameter_tuning_indices(
            data_x,
            data_y,
            test_size=fixed.get("test_size", 0.2),
            val_size=fixed.get("val_size", 0.2),
            is_regression=fixed.get("regression", True),
            random_state=fixed.get("hp_random_state", 42),
            combine_train_val=fixed.get("combine_train_val_for_final_training", False),
        )
      else:
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        _, test_idx = list(kfold.split(data_x, data_y))[fold_num_trained - 1]

      # Load model once
      ckpt_dir = (
          f"{ckpt_base_dir}/rho_{_format_rho(rho_val)}/seed_{seed}/"
          f"fold_{fold_num_trained}/split_{split_num_trained}/model_0/best_checkpoint"
      )
      nam, sess = load_nam_checkpoint(ckpt_dir, hyperparameters=hp_for_restore)

      # Compute test RMSE
      preds_full = get_test_predictions(nam, data_x.astype(np.float32), sess)
      rmse = float(graph_builder.calculate_metric(data_y[test_idx], preds_full[test_idx], regression=True))
      rmse_by_rho_seed[float(rho_val)][int(seed)] = rmse
      rmse_rows.append({"rho": float(rho_val), "seed": int(seed), "rmse": rmse})

      # Compute shape RMSE if requested
      if compute_shape_rmse:
        if dataset_name == "Correlated_linear":
          raw = data_utils.load_correlated_linear_data(n=fixed["correlated_n"], rho=rho_val, seed=seed)
        else:
          raw = data_utils.load_correlated_nonlinear_data(n=fixed["correlated_n"], rho=rho_val, seed=seed)
        raw_cols = list(raw["X"].columns)

        for j, feat in enumerate(column_names):
          learned_grid = sess.run(nam.feature_nns[j](x_scaled_grid.reshape(-1, 1), training=nam._false)).reshape(-1)
          learned_on_samples = sess.run(nam.feature_nns[j](data_x[:, [j]].astype(np.float32), training=nam._false)).reshape(-1)
          learned_mean = float(learned_on_samples.mean())
          learned_centered = learned_grid - learned_mean

          truth_fn, coef_val = _truth_fn_for_col(raw_cols[j])
          x_scaled_samples = data_x[:, j].astype(np.float32)
          true_mean = float(np.mean(truth_fn(x_scaled_samples)))
          true_centered = truth_fn(x_scaled_grid) - true_mean

          mse = float(np.mean((learned_centered - true_centered) ** 2))
          rmse_shape = float(np.sqrt(mse))

          shape_rows.append({
              "rho": float(rho_val),
              "seed": int(seed),
              "feature": feat,
              "coef": float(coef_val) if coef_val == coef_val else np.nan,
              "mse": mse,
              "rmse": rmse_shape,
          })
          shape_by_rho_seed_feature[float(rho_val)][int(seed)][feat] = rmse_shape

      sess.close()

  # Build DataFrames
  rmse_df = pd.DataFrame(rmse_rows).sort_values(["rho", "seed"]).reset_index(drop=True)
  summary = rmse_df.groupby("rho")["rmse"].agg(["mean", "std", "count"]).reset_index()
  rho_rmses = [rmse_df.loc[rmse_df["rho"] == float(rv), "rmse"].to_numpy() for rv in rhos]

  if compute_shape_rmse:
    shape_df = pd.DataFrame(shape_rows).sort_values(["rho", "seed", "feature"]).reset_index(drop=True)
    return (rmse_by_rho_seed, rmse_df, summary, rho_rmses), (shape_df, shape_by_rho_seed_feature)
  else:
    return rmse_by_rho_seed, rmse_df, summary, rho_rmses


def evaluate_ensemble_across_folds(
    all_preds_per_fold,
    fold_test_indices,
    data_y,
    is_regression: bool = True,
    verbose: bool = True,
):
  """Evaluate predictions across folds.
  
  Computes ensemble predictions (mean across models) for each fold and evaluates
  performance on test sets. Supports both regression (RMSE) and classification (AUC).
  
  Args:
    all_preds_per_fold: List of lists. Each inner list contains predictions from
      multiple models for that fold. Shape: [n_folds][n_models][n_test_samples]
    fold_test_indices: List of arrays, test set indices for each fold.
    data_y: Array of true labels/targets for the full dataset.
    is_regression: If True, compute RMSE. If False, compute AUC.
    verbose: If True, print per-fold metrics and summary statistics.
  
  Returns:
    fold_metrics: Array of metric values (RMSE or AUC) for each fold.
    avg_metric: Mean metric across folds.
    std_metric: Standard deviation of metric across folds.
  """
  fold_metrics = []
  metric_name = "RMSE" if is_regression else "AUC"
  
  for fold_idx in range(len(all_preds_per_fold)):
    # Get test set indices and labels for this fold
    test_indices = fold_test_indices[fold_idx]
    y_test_fold = data_y[test_indices]
    
    # Ensemble predictions (mean across models)
    preds = np.vstack(all_preds_per_fold[fold_idx])  # (n_models, n_test_samples)
    ensemble_pred = preds.mean(axis=0)
    
    # Compute metric on test set
    metric = graph_builder.calculate_metric(
        y_test_fold,
        ensemble_pred,
        regression=is_regression
    )
    fold_metrics.append(float(metric))
    
    if verbose:
      print(f"Fold {fold_idx+1} {metric_name} (test set only): {metric:.4f} (n={len(test_indices)})")
  
  fold_metrics = np.array(fold_metrics)
  avg_metric = fold_metrics.mean()
  std_metric = fold_metrics.std()
  
  if verbose:
    print(f"\nAverage Ensemble {metric_name} across folds (test sets only): {avg_metric:.4f}")
    print(f"Std of Ensemble {metric_name} across folds: {std_metric:.4f}")
  
  return fold_metrics, avg_metric, std_metric


def gather_nam_predictions_and_hist_data(
    fold_test_indices,
    data_x,
    column_names,
    unique_features,
    dataset_name,
    base_logdir=None,
    num_folds=5,
    num_splits=20,
    print_every=5,
):
  """Gather predictions and histogram data from NAM checkpoints across folds and splits.
  
  Args:
    fold_test_indices: List of arrays, test indices for each fold.
    data_x: Full dataset features.
    column_names: List of feature column names.
    unique_features: Unique feature values for each feature (from prepare_feature_arrays).
    dataset_name: Name of the dataset (used for checkpoint directory and feature predictions).
    base_logdir: Base directory for checkpoints. If None, uses f"./{dataset_name.lower()}".
    num_folds: Number of folds (default: 5).
    num_splits: Number of splits per fold (default: 20).
    print_every: Print progress every N splits (default: 5).
  
  Returns:
    all_preds_per_fold: List of lists. Each inner list contains test predictions from
      multiple models for that fold. Shape: [n_folds][n_splits][n_test_samples]
    all_hist_data: List of histogram data for each split.
    all_mean_pred: List of mean predictions for each split.
  """
  import numpy as np
  
  if base_logdir is None:
    base_logdir = f"./{dataset_name.lower()}"
  
  all_preds_per_fold = [[] for _ in range(num_folds)]
  all_hist_data = []
  all_mean_pred = []
  
  for fold in range(1, num_folds + 1):
    fold_idx = fold - 1
    test_indices = fold_test_indices[fold_idx]
    print(f"Processing fold {fold} (test set size: {len(test_indices)})...")
    
    for split in range(1, num_splits + 1):
      if split % print_every == 0:
        print(f"  Processing split {split}...")
      ckpt_dir = f"{base_logdir}/fold_{fold}/split_{split}/model_0/best_checkpoint"
      
      nam, sess = load_nam_checkpoint(ckpt_dir)
      
      # Get predictions on the FULL dataset (needed for feature importance plots)
      preds_full = get_test_predictions(nam, data_x.astype(np.float32), sess)
      
      # Store only test set predictions for metric calculation
      preds_test = preds_full[test_indices]
      all_preds_per_fold[fold_idx].append(preds_test)
      
      # For feature importance plots, we still need full dataset
      feature_predictions = get_feature_predictions(nam, dataset_name, sess)
      avg_hist_data, mean_pred, _ = compute_mean_predictions(
          data_x, column_names, unique_features, feature_predictions
      )
      
      all_hist_data.append(avg_hist_data)
      all_mean_pred.append(mean_pred)
  
  return all_preds_per_fold, all_hist_data, all_mean_pred


def plot_rmse_boxplot_vs_rho(rmse_df, dataset_label="Linear", save_path=None):
    """
    Visualize the RMSE per rho using boxplots for the distributions across seeds.
    Works for both linear and nonlinear datasets.

    Args:
        rmse_df (pd.DataFrame): DataFrame with columns "rho" and "rmse"
        dataset_label (str): Label used in the plot title (e.g., "Linear", "Nonlinear")
        save_path (str or Path, optional): If set, save figure and close; else plt.show()
    """
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=rmse_df, x="rho", y="rmse", palette="crest")
    plt.xlabel("Correlation (rho)")
    plt.ylabel("Test RMSE (across seeds)")
    plt.title(f"Test RMSE Distribution vs. Correlation ({dataset_label})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def analyze_shape_rmse(shape_df, dataset_label="Linear or Nonlinear", save_dir=None):
    """
    Print summary statistics and visualize shape RMSE for NAMs analysis.

    Args:
        shape_df (pd.DataFrame): DataFrame containing columns ['rho', 'seed', 'feature', ..., 'rmse']
        dataset_label (str): Used in print statements and plot titles
        save_dir (str or Path, optional): If set, save figures to this directory and close; else plt.show()
    """
    # Style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100

    # Shape RMSE summary
    print("\nShape RMSE Statistics:")
    print(f"   Overall mean: {shape_df['rmse'].mean():.4f}")
    print(f"   Overall std: {shape_df['rmse'].std():.4f}")
    print(f"   Min: {shape_df['rmse'].min():.4f}")
    print(f"   Max: {shape_df['rmse'].max():.4f}")

    # Per-feature shape RMSE stats
    feature_stats = shape_df.groupby("feature")["rmse"].agg(["mean", "std"]).sort_values("mean")
    print("\nPer-feature average shape RMSE (mean ± std):")
    for feat, row in feature_stats.iterrows():
        print(f"   {feat}: {row['mean']:.4f} ± {row['std']:.4f}")

    # Per-feature, per-rho shape RMSE heatmap
    pivot_mean = shape_df.pivot_table(index="feature", columns="rho", values="rmse", aggfunc="mean")

    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot_mean, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': 'Shape RMSE'})
    plt.title(f"Shape RMSE (mean across seeds) per Feature and Rho\n{dataset_label}")
    plt.xlabel("Correlation (rho)")
    plt.ylabel("Feature")
    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / "shape_rmse_heatmap.png", bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # Trend per feature across rhos, with std shaded
    plt.figure(figsize=(8, 5))
    for feature in shape_df["feature"].unique():
        grouped = shape_df[shape_df["feature"] == feature].groupby("rho")["rmse"]
        means = grouped.mean()
        stds = grouped.std()
        plt.plot(means.index, means.values, marker='o', label=feature, linewidth=2)
        plt.fill_between(
            means.index,
            means.values - stds.values,
            means.values + stds.values,
            alpha=0.25
        )
    plt.xlabel("Correlation (rho)")
    plt.ylabel("Shape RMSE (mean ± std)")
    plt.title(f"Shape RMSE per Feature across Rhos\n{dataset_label}")
    plt.legend(
        title="Feature",
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        borderaxespad=0.
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / "shape_rmse_trend.png", bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    
def compute_mean_predictions(data_x, column_names, unique_features, feature_predictions):
    """Compute index alignment and mean bias per feature."""
    avg_hist_data = {col: pred for col, pred in zip(column_names, feature_predictions)}
    all_indices, mean_pred = {}, {}

    for i, col in enumerate(column_names):
        x_i = data_x[:, i]
        all_indices[col] = np.searchsorted(unique_features[i][:, 0], x_i, 'left')

    for col in column_names:
        mean_pred[col] = np.mean([avg_hist_data[col][i] for i in all_indices[col]])

    return avg_hist_data, mean_pred, all_indices


def compute_mean_feature_importance(avg_hist_data, mean_pred):
    mean_abs_score = {}
    for feature, contribs in avg_hist_data.items():
        mean_abs_score[feature] = np.mean(np.abs(contribs - mean_pred[feature]))
    
    feature_names, mean_importances = zip(*mean_abs_score.items())
    return np.array(feature_names), np.array(mean_importances)



def plot_mean_feature_importance(feature_names, mean_importances, dataset_name, width=0.4, horizontal=False):
    sorted_idx = np.argsort(mean_importances)
    sorted_names = np.array(feature_names)[sorted_idx]
    sorted_values = mean_importances[sorted_idx]

    plt.figure(figsize=(7, 5))
    
    if horizontal:
        plt.barh(sorted_names, sorted_values, height=width, edgecolor='k')
        plt.xlabel("Mean Absolute Contribution", fontsize='x-large')
        plt.ylabel("Feature", fontsize='x-large')
    else:
        ind = np.arange(len(sorted_names))
        plt.bar(ind, sorted_values, width, edgecolor='k')
        plt.xticks(ind, sorted_names, rotation=90, fontsize='large')
        plt.ylabel("Mean Absolute Contribution", fontsize='x-large')
    
    plt.title(f"Feature Importance — {dataset_name}", fontsize='x-large', pad=10)


def plot_feature_importance_across_splits(
    all_hist_data,
    all_mean_pred,
    dataset_name,
    figsize=(10, 6),
    save_path=None,
):
  """Compute and plot feature importance aggregated across multiple splits with error bars.
  
  Args:
    all_hist_data: List of histogram data dictionaries (one per split).
    all_mean_pred: List of mean prediction dictionaries (one per split).
    dataset_name: Name of the dataset (for plot title).
    figsize: Figure size tuple (default: (10, 6)).
    save_path: If set, save figure to this path and close; otherwise plt.show().
  
  Returns:
    sorted_features: Array of feature names sorted by mean importance.
    sorted_mean: Array of mean importances (sorted).
    sorted_std: Array of std importances (sorted).
  """
  import numpy as np
  import matplotlib.pyplot as plt
  
  # Compute per-split importances
  all_feature_importances = []
  
  for avg_hist_data, mean_pred in zip(all_hist_data, all_mean_pred):
    feature_names, importances = compute_mean_feature_importance(avg_hist_data, mean_pred)
    all_feature_importances.append(importances)
  
  # Combine into matrix (splits x features)
  all_feature_importances = np.vstack(all_feature_importances)
  
  # Average and std across splits
  importance_mean = all_feature_importances.mean(axis=0)
  importance_std = all_feature_importances.std(axis=0)
  
  # Sort features by mean importance
  sorted_idx = np.argsort(importance_mean)
  sorted_features = np.array(feature_names)[sorted_idx]
  sorted_mean = importance_mean[sorted_idx]
  sorted_std = importance_std[sorted_idx]
  
  # Plot averaged feature importance with error bars
  plt.figure(figsize=figsize)
  plt.bar(sorted_features, sorted_mean, yerr=sorted_std, capsize=5, edgecolor="k")
  plt.xticks(rotation=90)
  plt.ylabel("Mean Absolute Contribution")
  plt.title(f"Feature Importance Across Splits — {dataset_name}")
  plt.tight_layout()
  if save_path:
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
  else:
    plt.show()
  
  return sorted_features, sorted_mean, sorted_std


def plot_all_hist(hist_data, num_rows, num_cols, color_base, mean_pred,
                  unique_features, categorical_names, col_mapping,
                  feature_mapping, dataset_label='Feature Contribution',
                  linewidth=3.0, alpha=1.0, feature_to_use=None,
                  ymin=None, ymax=None, x_limits=None, y_limits=None):

    # detect multi-model input
    if isinstance(hist_data, dict):
        hist_list = [hist_data]
        mean_list = [mean_pred]
        first_hist = hist_data
    else:
        hist_list = hist_data
        mean_list = mean_pred
        first_hist = hist_data[0]

    hist_data_pairs = sorted(first_hist.items(), key=lambda x: x[0])

    if feature_to_use:
        hist_data_pairs = [pair for pair in hist_data_pairs if pair[0] in feature_to_use]

    # plot each feature
    for i, (name, _) in enumerate(hist_data_pairs):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        x = unique_features[name]

        # plot individual model curves
        for h, m in zip(hist_list, mean_list):
            pred = h[name] - m[name]
            if name in categorical_names:
                x_round = np.round(x, 1)
                x_plot = x_round - 0.5
                x_plot[-1] += 1
                step_loc = "mid" if len(x_round) <= 2 else "post"
                ax.step(x_plot, pred, color=color_base, alpha=0.1,
                        linewidth=1, where=step_loc)
            else:
                ax.plot(x, pred, color=color_base, alpha=0.1, linewidth=1)

        # plot average curve
        avg_curve = np.mean([h[name] - m[name] for h, m in zip(hist_list, mean_list)], axis=0)

        if name in categorical_names:
            ax.step(x_plot, avg_curve, color=color_base, linewidth=3, where=step_loc)
            labels, rot = feature_mapping.get(name, (x_round, None))
            ax.set_xticks(x_round)
            ax.set_xticklabels(labels, rotation=rot, fontsize='large')
        else:
            ax.plot(x, avg_curve, color=color_base, linewidth=3)
            ax.tick_params(labelsize='large')

        # Handle y_limits: can be a tuple (global) or dict (per-feature)
        if y_limits is not None:
            if isinstance(y_limits, (tuple, list)) and len(y_limits) == 2:
                # Global y_limits for all features
                feature_ymin, feature_ymax = y_limits
                ax.set_ylim(feature_ymin, feature_ymax)
            elif isinstance(y_limits, dict) and name in y_limits:
                # Per-feature y_limits
                feature_ymin, feature_ymax = y_limits[name]
                ax.set_ylim(feature_ymin, feature_ymax)
            else:
                ax.set_ylim(ymin, ymax)
        else:
            ax.set_ylim(ymin, ymax)

        if x_limits is not None and name in x_limits:
            x_limit_val = x_limits[name]
            if isinstance(x_limit_val, (tuple, list)) and len(x_limit_val) == 2:
                min_x, max_x = x_limit_val
            else:
                min_x = np.min(x)
                max_x = x_limit_val
            if name in categorical_names:
                min_x -= 0.5
                max_x += 0.5
            ax.set_xlim(min_x, max_x)
        else:
            min_x, max_x = np.min(x), np.max(x)
            if name in categorical_names:
                min_x -= 0.5
                max_x += 0.5
            ax.set_xlim(min_x, max_x)

        if i % num_cols == 0:
            ax.set_ylabel(dataset_label, fontsize='x-large')
        ax.set_xlabel(col_mapping.get(name, name), fontsize='x-large')

    return ymin, ymax

def shade_by_density_blocks(hist_data, unique_features, single_features,
                            n_blocks=5, color=(0.9, 0.5, 0.5),
                            categorical_names=None, feature_to_use=None,
                            ymin=None, ymax=None, x_limits=None, y_limits=None):

    fig = plt.gcf()
    axes = fig.get_axes()

    hist_data_pairs = sorted(hist_data.items(), key=lambda x: x[0])
    if feature_to_use:
        hist_data_pairs = [v for v in hist_data_pairs if v[0] in feature_to_use]

    for i, (name, _) in enumerate(hist_data_pairs):
        ax = axes[i]
        x = unique_features[name]
        data = single_features[name]

        if x_limits is not None and name in x_limits:

            x_limit_val = x_limits[name]
            if isinstance(x_limit_val, (tuple, list)) and len(x_limit_val) == 2:

                min_x_orig, max_x_orig = x_limit_val
            else:
                min_x_orig = np.min(x)
                max_x_orig = x_limit_val
        else:
            # Use data limits
            min_x_orig, max_x_orig = np.min(x), np.max(x)
        
        if categorical_names and name in categorical_names:
            min_x = min_x_orig - 0.5
            max_x = max_x_orig + 0.5
        else:
            min_x = min_x_orig
            max_x = max_x_orig
        
        # Handle y_limits: can be a tuple (global) or dict (per-feature)
        if y_limits is not None:
            if isinstance(y_limits, (tuple, list)) and len(y_limits) == 2:
                # Global y_limits for all features
                feature_ymin, feature_ymax = y_limits
            elif isinstance(y_limits, dict) and name in y_limits:
                # Per-feature y_limits
                feature_ymin, feature_ymax = y_limits[name]
            else:
                feature_ymin, feature_ymax = ymin, ymax
        else:
            feature_ymin, feature_ymax = ymin, ymax

        data_filtered = data[(data >= min_x_orig) & (data <= max_x_orig)]
        
        if len(data_filtered) == 0:
            continue

        x_visible = x[(x >= min_x_orig) & (x <= max_x_orig)]
        x_n_blocks = min(n_blocks, max(len(x_visible), 1))
        
        range_size = max_x_orig - min_x_orig
        if range_size > 0 and range_size < 10:
            x_n_blocks = min(x_n_blocks, int(range_size) + 1)
        
        density, bin_edges = np.histogram(data_filtered, bins=x_n_blocks, range=(min_x, max_x))
        if np.max(density) > 0:
            density = density / np.max(density)
        else:
            density = np.zeros(x_n_blocks)

        for p in range(x_n_blocks):
            start = bin_edges[p]
            end = bin_edges[p + 1]
            alpha = min(1.0, 0.01 + density[p])

            rect = patches.Rectangle(
                (start, feature_ymin),
                end - start,
                feature_ymax - feature_ymin,
                facecolor=color,
                edgecolor=color,
                linewidth=0,
                alpha=alpha
            )
            ax.add_patch(rect)

def plot_nam_contributions_with_density(
    hist_data,
    unique_features,
    single_features,
    categorical_names,
    col_mapping,
    feature_mapping,
    mean_pred,
    feature_to_use=None,
    colors=None,
    n_blocks=20,
    num_cols=4,
    figsize_scale=4.5,
    dataset_label="Feature Contribution",
    return_limits=False,
    x_limits=None,
    y_limits=None,
    save_path=None,
):
    if colors is None:
        colors = [[0.9, 0.4, 0.5], [0.5, 0.9, 0.4], [0.4, 0.5, 0.9], [0.9, 0.5, 0.9]]

    num_features = len(hist_data) if feature_to_use is None else len(feature_to_use)
    num_rows = int(np.ceil(num_features / num_cols))

    # build figure
    fig = plt.figure(
        figsize=(num_cols * figsize_scale, num_rows * figsize_scale),
        facecolor='w',
        edgecolor='k'
    )

    # detect single or multi-model
    if isinstance(hist_data, dict):
        hist_list = [hist_data]
        mean_list = [mean_pred]
    else:
        hist_list = hist_data
        mean_list = mean_pred

    # ---- Compute unified y limits ----
    global_vals = []
    for h, m in zip(hist_list, mean_list):
        for name in h:
            global_vals.append(h[name] - m[name])
    global_vals = np.concatenate(global_vals)

    base_min = np.min(global_vals)
    base_max = np.max(global_vals)

    ymin = base_min - 1
    ymax = base_max + 1

    # ---- plot curves ----
    plot_all_hist(
        hist_data=hist_data,
        num_rows=num_rows,
        num_cols=num_cols,
        color_base=colors[2],
        mean_pred=mean_pred,
        unique_features=unique_features,
        categorical_names=categorical_names,
        col_mapping=col_mapping,
        feature_mapping=feature_mapping,
        dataset_label=dataset_label,
        feature_to_use=feature_to_use,
        ymin=ymin,
        ymax=ymax,
        x_limits=x_limits,
        y_limits=y_limits
    )

    # ---- shading ----
    shade_by_density_blocks(
        hist_data=hist_data[0] if isinstance(hist_data, list) else hist_data,
        unique_features=unique_features,
        single_features=single_features,
        n_blocks=n_blocks,
        color=colors[0],
        categorical_names=categorical_names,
        feature_to_use=feature_to_use,
        ymin=ymin,
        ymax=ymax,
        x_limits=x_limits,
        y_limits=y_limits
    )

    plt.subplots_adjust(hspace=0.25)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    if return_limits:
        return fig, (ymin, ymax)
    return fig


def gather_ebm_predictions_and_shape_functions(
    fold_test_indices,
    data_x,
    data_y,
    column_names,
    dataset_name,
    best_hp,
    fixed_hp,
    is_regression=True,
    num_folds=5,
    num_splits=20,
    print_every=5,
    base_logdir=None,
):
    """
    Gather EBM predictions and shape functions across folds and splits.
    
    Args:
        fold_test_indices: List of test indices for each fold
        data_x: Full dataset features
        data_y: Full dataset targets
        column_names: List of feature names
        dataset_name: Name of the dataset
        best_hp: Best hyperparameters dict
        fixed_hp: Fixed hyperparameters dict
        is_regression: Whether this is a regression task
        num_folds: Number of folds
        num_splits: Number of splits per fold
        print_every: Print progress every N splits
        base_logdir: Base directory for saving/loading models (optional)
    
    Returns:
        all_preds_per_fold: List of lists of predictions [fold][split][samples]
        all_shape_functions: List of shape function data for plotting
        all_mean_pred: List of mean predictions per split
    """
    from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
    from interpret import show, preserve
    import pickle
    import os
    
    all_preds_per_fold = [[] for _ in range(num_folds)]
    all_shape_functions = []
    all_mean_pred = []
    
    for fold in range(1, num_folds + 1):
        fold_idx = fold - 1
        test_indices = fold_test_indices[fold_idx]
        print(f"Processing fold {fold} (test set size: {len(test_indices)})...")
        
        # Create fold-specific train/test split
        train_indices = np.setdiff1d(np.arange(len(data_x)), test_indices)
        X_train_fold = data_x[train_indices]
        y_train_fold = data_y[train_indices]
        X_test_fold = data_x[test_indices]
        y_test_fold = data_y[test_indices]
        
        for split in range(1, num_splits + 1):
            should_print = (split % print_every == 0) or (split == 1) or (split == num_splits)
            
            if should_print:
                print(f"  Processing split {split}/{num_splits}...", end=' ', flush=True)
            
            # Check if model exists
            model_path = None
            if base_logdir:
                model_path = os.path.join(base_logdir, f'fold_{fold}', f'split_{split}', 'ebm_model.pkl')
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Load or train model
            if model_path and os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                if should_print:
                    print("(loaded)", flush=True)
            else:
                # Train new model
                seed = (fold - 1) * num_splits + split
                
                if is_regression:
                    model = ExplainableBoostingRegressor(
                        **best_hp,
                        random_state=seed,
                        n_jobs=fixed_hp.get('n_jobs', -1),
                        early_stopping_rounds=fixed_hp.get('early_stopping_rounds', 50),
                        validation_size=fixed_hp.get('validation_size', 0.2)
                    )
                else:
                    model = ExplainableBoostingClassifier(
                        **best_hp,
                        random_state=seed,
                        n_jobs=fixed_hp.get('n_jobs', -1),
                        early_stopping_rounds=fixed_hp.get('early_stopping_rounds', 50),
                        validation_size=fixed_hp.get('validation_size', 0.2)
                    )
                
                # Convert to DataFrame for EBM
                import pandas as pd
                X_train_df = pd.DataFrame(X_train_fold, columns=column_names)
                model.fit(X_train_df, y_train_fold)
                
                # Save model if path provided
                if model_path:
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                
                if should_print:
                    print("(trained)", flush=True)
            
            # Get predictions on test set
            import pandas as pd
            X_test_df = pd.DataFrame(X_test_fold, columns=column_names)
            if is_regression:
                preds_test = model.predict(X_test_df)
            else:
                preds_test = model.predict_proba(X_test_df)[:, 1]
            
            all_preds_per_fold[fold_idx].append(preds_test)
            
            # Extract shape functions using interpretML's API
            # interpretML provides shape functions through explain_global()
            global_explanation = model.explain_global()
            
            # Store shape function data
            shape_data = {}
            
            # Extract shape functions - interpretML stores them in the model's internal structure
            # We can access them through the explanation or directly from the model
            try:
                # Method 1: Try to get data from explanation object
                try:
                    all_data = global_explanation.data()
                    if isinstance(all_data, dict):
                        # Extract features from the data dictionary
                        for feat_name in column_names:
                            if feat_name in all_data:
                                feat_info = all_data[feat_name]
                                if isinstance(feat_info, dict) and 'names' in feat_info and 'scores' in feat_info:
                                    x_vals = np.array(feat_info['names']).tolist() if feat_info['names'] is not None else []
                                    y_vals = np.array(feat_info['scores']).tolist() if feat_info['scores'] is not None else []
                                    if x_vals and y_vals and len(x_vals) == len(y_vals):
                                        shape_data[feat_name] = {
                                            'x': x_vals,
                                            'y': y_vals,
                                            'type': 'continuous'
                                        }
                except:
                    pass
                
                # Method 2: Access shape functions directly from the model
                # EBM models store shape functions in model.term_features_ and model.term_scores_
                # Only use this method if Method 1 didn't work
                if not shape_data and hasattr(model, 'term_features_') and hasattr(model, 'term_scores_'):
                    # Get feature indices and their corresponding scores
                    term_features = model.term_features_
                    term_scores = model.term_scores_
                    
                    # Create a mapping from feature index to feature name
                    for feat_idx, feat_name in enumerate(column_names):
                        # Find terms that correspond to this feature (single feature terms)
                        # Single feature terms have length 1
                        found = False
                        for term_idx, term_feat in enumerate(term_features):
                            if len(term_feat) == 1 and term_feat[0] == feat_idx:
                                # This is a single-feature term for our feature
                                scores = term_scores[term_idx]
                                
                                # Get the bin edges and centers from the model
                                # EBM stores bin edges in model.bin_edges_
                                if hasattr(model, 'bin_edges_') and feat_idx < len(model.bin_edges_):
                                    bin_edges = model.bin_edges_[feat_idx]
                                    if bin_edges is not None and len(bin_edges) > 0:
                                        # Use bin centers as x values
                                        bin_edges = np.array(bin_edges) if not isinstance(bin_edges, np.ndarray) else bin_edges
                                        scores = np.array(scores) if not isinstance(scores, np.ndarray) else scores
                                        
                                        if len(bin_edges) == len(scores) + 1:
                                            # Bin edges, compute centers
                                            x_vals = [(bin_edges[i] + bin_edges[i+1]) / 2.0 for i in range(len(bin_edges)-1)]
                                        elif len(bin_edges) == len(scores):
                                            # Already centers
                                            x_vals = bin_edges.tolist()
                                        else:
                                            # Fallback: use indices
                                            x_vals = list(range(len(scores)))
                                        
                                        y_vals = scores.tolist()
                                        
                                        if x_vals and y_vals and len(x_vals) == len(y_vals):
                                            shape_data[feat_name] = {
                                                'x': x_vals,
                                                'y': y_vals,
                                                'type': 'continuous'
                                            }
                                            found = True
                                            break
                                elif hasattr(model, 'feature_bounds_') and feat_idx < len(model.feature_bounds_):
                                    # Alternative: use feature bounds to create x values
                                    bounds = model.feature_bounds_[feat_idx]
                                    if bounds is not None and len(bounds) == 2:
                                        # Create evenly spaced x values
                                        x_vals = np.linspace(bounds[0], bounds[1], len(scores)).tolist()
                                        y_vals = np.array(scores).tolist() if not isinstance(scores, np.ndarray) else scores.tolist()
                                        if x_vals and y_vals and len(x_vals) == len(y_vals):
                                            shape_data[feat_name] = {
                                                'x': x_vals,
                                                'y': y_vals,
                                                'type': 'continuous'
                                            }
                                            found = True
                                            break
                        
                        # If still not found, create x values from data range
                        if not found and feat_idx < data_x.shape[1]:
                            # Use the actual data range for this feature
                            feat_data = data_x[:, feat_idx]
                            x_min, x_max = float(np.min(feat_data)), float(np.max(feat_data))
                            scores = None
                            for term_idx, term_feat in enumerate(term_features):
                                if len(term_feat) == 1 and term_feat[0] == feat_idx:
                                    scores = term_scores[term_idx]
                                    break
                            
                            if scores is not None:
                                scores = np.array(scores) if not isinstance(scores, np.ndarray) else scores
                                x_vals = np.linspace(x_min, x_max, len(scores)).tolist()
                                y_vals = scores.tolist()
                                if x_vals and y_vals and len(x_vals) == len(y_vals):
                                    shape_data[feat_name] = {
                                        'x': x_vals,
                                        'y': y_vals,
                                        'type': 'continuous'
                                    }
                
                # Method 3: Try accessing through explanation's internal structure
                if not shape_data:
                    try:
                        # Try to get feature names and their data
                        for feat_name in column_names:
                            try:
                                feat_data = global_explanation.data(feat_name)
                                if isinstance(feat_data, dict):
                                    if 'names' in feat_data and 'scores' in feat_data:
                                        x_vals = np.array(feat_data['names']).tolist() if feat_data['names'] is not None else []
                                        y_vals = np.array(feat_data['scores']).tolist() if feat_data['scores'] is not None else []
                                        if x_vals and y_vals and len(x_vals) == len(y_vals):
                                            shape_data[feat_name] = {
                                                'x': x_vals,
                                                'y': y_vals,
                                                'type': 'continuous'
                                            }
                            except:
                                pass
                    except:
                        pass
                        
            except Exception as e:
                # If extraction fails, shape_data will be empty
                if split % print_every == 0:
                    print(f"    Warning: Could not extract shape functions: {type(e).__name__}: {str(e)[:100]}")
                pass
            
            all_shape_functions.append(shape_data)
            
            # Compute mean prediction
            if is_regression:
                mean_pred = float(preds_test.mean())
            else:
                mean_pred = float(preds_test.mean())
            all_mean_pred.append(mean_pred)
    
    return all_preds_per_fold, all_shape_functions, all_mean_pred


def plot_ebm_shape_functions(
    column_names,
    all_shape_functions,
    y_limits=(-0.1, 0.1),
    n_cols=4,
    figsize_scale=4.0,
    test_data=None,
    data_x=None,
    fold_test_indices=None,
    n_blocks=20
):
    """
    Plot EBM shape functions across all splits (one line per split) plus mean.
    Similar style to NAM plots.
    
    Args:
        column_names: List of feature names
        all_shape_functions: List of dictionaries, each containing shape function data
                           for each split. Each dict should have format:
                           {feature_name: {'x': [...], 'y': [...], 'type': 'continuous'}}
        y_limits: Tuple of (ymin, ymax) for y-axis limits. Default is (-0.1, 0.1).
        n_cols: Number of columns in the subplot grid. Default is 4.
        figsize_scale: Scaling factor for figure size. Default is 4.0.
        test_data: Deprecated parameter (kept for backward compatibility, not used).
        data_x: Deprecated parameter (kept for backward compatibility, not used).
        fold_test_indices: Deprecated parameter (kept for backward compatibility, not used).
        n_blocks: Deprecated parameter (kept for backward compatibility, not used).
    
    Returns:
        fig: matplotlib figure object
    """
    n_features = len(column_names)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize_scale, n_rows * figsize_scale))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    # Color for plotting (similar to NAM style)
    color_base = [0.4, 0.5, 0.9]  # Blue color similar to NAM
    
    for i, feat_name in enumerate(column_names):
        ax = axes[i]
        
        # Collect shape function data across all splits
        all_x = []
        all_y = []
        
        for shape_data in all_shape_functions:
            if feat_name in shape_data:
                feat_data = shape_data[feat_name]
                if 'x' in feat_data and 'y' in feat_data:
                    all_x.append(feat_data['x'])
                    all_y.append(feat_data['y'])
        
        if all_x and all_y:
            # Find common x values (use first split's x values)
            x_vals = np.array(all_x[0])
            y_vals = []
            
            # Plot individual split lines
            for x_arr, y_arr in zip(all_x, all_y):
                x_arr = np.array(x_arr)
                y_arr = np.array(y_arr)
                # Only plot if x arrays match in length
                if len(x_arr) == len(x_vals):
                    ax.plot(x_vals, y_arr, color=color_base, alpha=0.1, linewidth=1)
                    y_vals.append(y_arr)
            
            # Plot mean line
            if y_vals:
                y_mean = np.mean(y_vals, axis=0)
                ax.plot(x_vals, y_mean, color=color_base, linewidth=3)
            
            ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
            ax.set_xlabel(feat_name, fontsize='x-large')
            ax.set_ylabel('Contribution', fontsize='x-large')
            ax.set_title(feat_name, fontsize='x-large')
            ax.set_ylim(y_limits[0], y_limits[1])
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize='large')
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def save_performance_metrics(
    all_preds_per_fold,
    fold_test_indices,
    data_y,
    dataset_id,
    dataset_name,
    task_type,
    model_type,
    num_folds,
    num_splits,
    is_regression,
    project_root,
    verbose=True
):
    """
    Save detailed performance metrics for statistical analysis.
    
    Args:
        all_preds_per_fold: List of lists of predictions [fold][split][samples]
        fold_test_indices: List of test indices for each fold
        data_y: Full dataset targets
        dataset_id: OpenML dataset ID
        dataset_name: Dataset name string
        task_type: "classification" or "regression"
        model_type: "NAM" or "EBM"
        num_folds: Number of folds
        num_splits: Number of splits per fold
        is_regression: Whether this is a regression task
        project_root: Path to project root directory
        verbose: Whether to print summary
    
    Returns:
        Path to saved JSON file
    """
    import json
    from pathlib import Path
    from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
    import numpy as np
    
    # Compute additional metrics per fold
    detailed_metrics = {
        'dataset_id': dataset_id,
        'dataset_name': dataset_name,
        'task_type': task_type,
        'model_type': model_type,
        'num_folds': num_folds,
        'num_splits': num_splits,
        'folds': []
    }
    
    for fold_idx in range(num_folds):
        test_indices = fold_test_indices[fold_idx]
        y_test_fold = data_y[test_indices]
        
        # Ensemble predictions (mean across splits)
        fold_preds = np.array(all_preds_per_fold[fold_idx])
        ensemble_pred = np.mean(fold_preds, axis=0)
        
        # Compute metrics
        if is_regression:
            rmse = np.sqrt(mean_squared_error(y_test_fold, ensemble_pred))
            mae = mean_absolute_error(y_test_fold, ensemble_pred)
            fold_data = {
                'fold': fold_idx + 1,
                'test_set_size': len(test_indices),
                'rmse': float(rmse),
                'mae': float(mae),
                'metric_name': 'RMSE',
                'metric_value': float(rmse)
            }
        else:
            auc = roc_auc_score(y_test_fold, ensemble_pred)
            fold_data = {
                'fold': fold_idx + 1,
                'test_set_size': len(test_indices),
                'auc': float(auc),
                'metric_name': 'AUC',
                'metric_value': float(auc)
            }
        
        # Add per-split metrics (individual model performance)
        split_metrics = []
        for split_idx in range(num_splits):
            split_pred = fold_preds[split_idx]
            if is_regression:
                split_rmse = np.sqrt(mean_squared_error(y_test_fold, split_pred))
                split_mae = mean_absolute_error(y_test_fold, split_pred)
                split_metrics.append({
                    'split': split_idx + 1,
                    'rmse': float(split_rmse),
                    'mae': float(split_mae)
                })
            else:
                split_auc = roc_auc_score(y_test_fold, split_pred)
                split_metrics.append({
                    'split': split_idx + 1,
                    'auc': float(split_auc)
                })
        
        fold_data['per_split_metrics'] = split_metrics
        detailed_metrics['folds'].append(fold_data)
    
    # Add summary statistics
    fold_metrics_array = np.array([f['metric_value'] for f in detailed_metrics['folds']])
    avg_metric_value = np.mean(fold_metrics_array)
    std_metric_value = np.std(fold_metrics_array)
    metric_name = detailed_metrics['folds'][0]['metric_name']
    
    detailed_metrics['summary'] = {
        'mean_metric': float(avg_metric_value),
        'std_metric': float(std_metric_value),
        'min_metric': float(np.min(fold_metrics_array)),
        'max_metric': float(np.max(fold_metrics_array)),
        'metric_name': metric_name
    }
    
    # Save to file
    results_dir = Path(project_root) / 'results' / 'evaluation'
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f'{model_type.lower()}_{dataset_name.replace("/", "_").replace(":", "_")}_performance.json'
    
    with open(output_file, 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    
    if verbose:
        print(f"\n{'='*70}")
        print("PERFORMANCE METRICS SAVED")
        print(f"{'='*70}")
        print(f"Saved to: {output_file}")
        print(f"Summary:")
        print(f"  Mean {metric_name}: {avg_metric_value:.4f}")
        print(f"  Std {metric_name}: {std_metric_value:.4f}")
        print(f"  Min {metric_name}: {detailed_metrics['summary']['min_metric']:.4f}")
        print(f"  Max {metric_name}: {detailed_metrics['summary']['max_metric']:.4f}")
        print(f"{'='*70}")
    
    return output_file
