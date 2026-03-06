import json
import os
import random
import subprocess
import sys
import time
import shlex
from pathlib import Path
import numpy as np
import pandas as pd
import neural_additive_models.data_utils as data_utils
import neural_additive_models.graph_builder as graph_builder
from utils import load_nam_checkpoint, get_test_predictions


def load_hp_tuning_conf(conf_path):
    """Load hp_search_space + fixed_hp from JSON.

    - discrete choices: list of values
    - continuous ranges: [min, max] (2-number list) -> converted to tuple(min, max)
    """
    conf_path = Path(conf_path)
    with open(conf_path, 'r', encoding='utf-8') as f:
        conf = json.load(f)

    hp_search_space = {}
    for k, v in conf['hp_search_space'].items():
        if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            hp_search_space[k] = (float(v[0]), float(v[1]))
        else:
            hp_search_space[k] = v

    fixed_hp = conf['fixed_hp']
    return hp_search_space, fixed_hp


def sample_hyperparameters(hp_search_space, random_seed=None):
    """Sample hyperparameters from the search space.
    
    Handles both discrete sets (lists) and continuous intervals (tuples).
    
    Args:
        hp_search_space: Dictionary of hyperparameter search spaces
        random_seed: Optional seed. If None, uses current random state.
                    If provided, creates a new random state for this sample.
    """
    # Create a separate random state to avoid affecting global state
    if random_seed is not None:
        rng = random.Random(random_seed)
        np_rng = np.random.RandomState(random_seed)
    else:
        rng = random
        np_rng = np.random

    hp_config = {}
    for key, values in hp_search_space.items():
        if isinstance(values, tuple):
            # Continuous interval: sample uniformly from [min, max)
            min_val, max_val = values
            hp_config[key] = rng.uniform(min_val, max_val)
        else:
            # Discrete set: choose randomly
            # Use random.choice for list, or np_rng.choice for numpy arrays
            if isinstance(values, list):
                hp_config[key] = rng.choice(values)
            else:
                hp_config[key] = np_rng.choice(values)
    return hp_config


def generate_training_command(fixed_hp, hyperparameters, trial_num, data_split, base_logdir):
    """Generate a training command string from hyperparameters."""
    # Convert to absolute path to avoid path resolution issues
    base_logdir = Path(base_logdir).resolve()
    logdir = str(base_logdir / f'trial_{trial_num}')

    all_hp = {**fixed_hp, **hyperparameters}
    all_hp['logdir'] = logdir

    cmd = [sys.executable, '-m', 'neural_additive_models.nam_train']

    for key, value in all_hp.items():
        if key == 'data_split':
            value = data_split

        if value is None:
            arg = f'--{key}=None'
        elif isinstance(value, bool):
            arg = f'--{key}={"true" if value else "false"}'
        else:
            # Properly quote values that contain spaces or special characters
            value_str = str(value)
            # On Windows, use double quotes; on Unix, shlex.quote handles it
            # Check if value contains spaces or special characters that need quoting
            if ' ' in value_str or ('\\' in value_str and os.name == 'nt'):
                # Windows: use double quotes and escape any existing double quotes
                if os.name == 'nt':  # Windows
                    escaped_value = value_str.replace('"', '\\"')
                    arg = f'--{key}="{escaped_value}"'
                else:  # Unix/Linux/Mac
                    quoted_value = shlex.quote(value_str)
                    arg = f'--{key}={quoted_value}'
            else:
                arg = f'--{key}={value_str}'

        cmd.append(arg)

    return ' '.join(cmd), logdir


def run_trials(fixed_hp, hyperparameters, base_logdir, n_trials, num_splits, trials_to_run=None,
               skip_if_exists=True, per_split_timeout_s=3600):
    """Run training for each trial and split, saving per-trial split results JSON."""
    if trials_to_run is None:
        trials_to_run = range(1, n_trials + 1)

    for trial_num in trials_to_run:
        trial_data = hyperparameters[trial_num - 1]

        split_results_file = os.path.join(base_logdir, f'trial_{trial_num}_split_results.json')
        if skip_if_exists and os.path.exists(split_results_file):
            print(f"Skipping trial {trial_num}: {split_results_file} already exists")
            continue

        print(f"\n=== Trial {trial_num}/{n_trials} ===")
        split_results = []

        for split_idx in range(num_splits):
            print(f"Running split {split_idx+1}/{num_splits} for trial {trial_num}...")

            split_cmd, _ = generate_training_command(
                fixed_hp,
                trial_data['hyperparameters'],
                trial_num,
                split_idx + 1,  # data_split is 1-indexed
                base_logdir,
            )

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
            
            # Set working directory to project root to ensure consistent path resolution
            cwd = str(project_root)
            
            start_time = time.time()
            result = subprocess.run(
                split_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=per_split_timeout_s,
                env=env,
                cwd=cwd,
            )
            training_time = time.time() - start_time
            success = result.returncode == 0

            print(f"  Completed in {training_time:.1f}s (success={success})")

            split_results.append({
                'split': split_idx,
                'command': split_cmd,
                'training_time': training_time,
                'success': success,
                'returncode': result.returncode,
                'stdout_tail': result.stdout[-4000:] if result.stdout else '',
                'stderr_tail': result.stderr[-4000:] if result.stderr else '',
            })

        with open(split_results_file, 'w') as f:
            json.dump({
                'trial': trial_num,
                'hyperparameters': trial_data['hyperparameters'],
                'split_results': split_results,
            }, f, indent=2)

        print(f"Saved: {split_results_file}")


def extract_validation_score_for_split(logdir, fixed_hp, split_num, hyperparameters=None):
    """Extract validation score for a specific split.

    Args:
        logdir: Base logdir for the trial (e.g., './hp_tuning/trial_1')
        fixed_hp: Dict of fixed hyperparameters/config (must include dataset + split settings).
        split_num: Split number (1-indexed)
        hyperparameters: Trial hyperparameters (e.g. dropout/l2/lr/feature_dropout/output_regularization).

    Returns:
        Validation score (RMSE for regression, AUROC for classification) or None if failed.
    """
    
    # Load dataset and recreate the split (must match the training split)
    data_x, data_y, _ = data_utils.load_dataset(
        fixed_hp['dataset_name'],
        correlated_n=fixed_hp.get('correlated_n'),
        correlated_rho=fixed_hp.get('correlated_rho'),
        correlated_seed=fixed_hp.get('correlated_seed'),
    )
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = \
        data_utils.get_train_val_test_split(
            data_x, data_y,
            test_size=fixed_hp['test_size'],
            val_size=fixed_hp['val_size'],
            stratified=not fixed_hp['regression'],
            random_state=fixed_hp['hp_random_state']
        )

    # In hyperparameter tuning mode, the logdir structure is: logdir/fold_1/split_{split_num}/model_0/best_checkpoint
    checkpoint_dir = os.path.join(logdir, 'fold_1', f'split_{split_num}', 'model_0', 'best_checkpoint')

    if not os.path.exists(checkpoint_dir):
        return None

    # Reconstruct the exact same model architecture that was trained.
    hp = dict(hyperparameters or {})
    hp['activation'] = fixed_hp['activation']
    hp['shallow'] = fixed_hp['shallow']
    hp['dropout'] = float(hp.get('dropout', 0.0))
    hp['feature_dropout'] = float(hp.get('feature_dropout', 0.0))

    nam, sess = load_nam_checkpoint(checkpoint_dir, hyperparameters=hp)

    # Get validation predictions
    preds_val = get_test_predictions(nam, x_val.astype(np.float32), sess)

    # Calculate metric
    score = graph_builder.calculate_metric(
        y_val, preds_val, regression=fixed_hp['regression']
    )

    sess.close()
    return float(score)


def build_validation_scores_df(split_results_files, base_logdir, fixed_hp):
    """Build a tidy DataFrame of validation scores from saved split-results JSONs."""
    rows = []
    for split_file in split_results_files:
        with open(split_file, 'r') as f:
            data = json.load(f)

        logdir = os.path.join(base_logdir, f"trial_{data['trial']}")
        for split_result in data['split_results']:
            if not split_result.get('success'):
                continue

            validation_score = extract_validation_score_for_split(
                logdir,
                fixed_hp,
                split_result['split'] + 1,
                hyperparameters=data.get('hyperparameters'),
            )

            if validation_score is None:
                continue

            rows.append({
                'trial': data['trial'],
                'split': split_result['split'],
                'validation_score': validation_score,
            })

    return pd.DataFrame(rows)


def select_and_save_best_hp(df, base_logdir, out_filename='best_hp.json'):
    """Select best trial (lowest mean validation_score) and save its hyperparameters."""

    mean_by_trial = df.groupby('trial')['validation_score'].mean()
    best_trial = int(mean_by_trial.idxmin())

    split_results_path = os.path.join(base_logdir, f'trial_{best_trial}_split_results.json')
    with open(split_results_path, 'r') as f:
        best_data = json.load(f)

    print(f"Best trial: {best_trial} (avg validation score: {mean_by_trial.min():.4f})")
    print("\nHyperparameters:")
    for key, value in best_data['hyperparameters'].items():
        print(f"  {key}: {value}")

    out_path = os.path.join(base_logdir, out_filename)
    with open(out_path, 'w') as fout:
        json.dump(best_data['hyperparameters'], fout, indent=2)
    print(f"\nBest hyperparameters saved to: {out_path}")

    return best_trial, best_data