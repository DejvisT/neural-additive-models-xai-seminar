"""
run_ebm_hyperparameter_tuning.py

Script equivalent of ebm_hyperparameter_tuning.ipynb, kept similar for correctness checks.

Notebook mapping:
  Cell 1: imports + project_root + src path + data_utils
  Cell 3: dataset config + hp_search_space + fixed_hp + tuning params + results_dir
  Cell 5: load dataset + determine regression + train/val/test split
  Cell 7: sample_hyperparameters + generate configs
  Cell 9: train_and_evaluate_ebm + tuning loop
  Cell 10: select best + save best_hp + save all results + summary stats

Adds:
  - CLI args for dataset_id and task_type
  - Plot saving (tuning curve + training time curve)
  - Optional run_tag to avoid overwriting existing files
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "src").is_dir() and (p / "results").is_dir():
            return p
    # fallback
    for p in [start, *start.parents]:
        if (p / "src").is_dir():
            return p
    return start


def ensure_src_on_path(project_root: Path) -> None:
    src_path = project_root / "src"
    if src_path.is_dir() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hyperparameter tuning for Explainable Boosting Machines (EBM) on OpenML datasets.",
    )
    p.add_argument("--dataset_id", type=int, required=True, help="OpenML dataset ID, e.g. 31")
    p.add_argument(
        "--task_type",
        choices=["classification", "regression"],
        required=True,
        help="Task type (classification or regression).",
    )
    p.add_argument("--n_trials", type=int, default=50, help="Number of tuning trials (default: 50)")
    p.add_argument("--random_seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--test_size", type=float, default=0.2, help="Test split fraction (default: 0.2)")
    p.add_argument("--val_size", type=float, default=0.2, help="Validation fraction of train_val (default: 0.2)")
    p.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional suffix for output filenames to avoid overwriting (e.g. refactor_test).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files (default: False).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# search space + fixed HP
# ---------------------------------------------------------------------------

def get_hp_search_space() -> Dict[str, Any]:
    return {
        "learning_rate": [0.001, 0.1],  # continuous range
        "max_bins": [16, 32, 64, 128, 256, 512],
        "max_interaction_bins": [8, 16, 32, 64, 128],
        "interactions": [0, 2, 5, 10, 15, 20],
        "outer_bags": [1, 2, 4, 8, 16],
        "inner_bags": [0, 2, 4, 8, 16],
        "min_samples_leaf": [1, 2, 3, 5, 10],
        "max_leaves": [2, 3, 5, 10, 15, 20],
    }


def get_fixed_hp(random_state: int) -> Dict[str, Any]:
    # same as notebook but allow random_state via CLI
    return {
        "random_state": random_state,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
        "validation_size": 0.125,
    }


# ---------------------------------------------------------------------------
# sample_hyperparameters
# ---------------------------------------------------------------------------

def sample_hyperparameters(search_space: Dict[str, Any], random_seed: Optional[int] = None) -> Dict[str, Any]:
    np.random.seed(random_seed)
    hp: Dict[str, Any] = {}

    for key, values in search_space.items():
        if (
            isinstance(values, list)
            and len(values) == 2
            and all(isinstance(x, (int, float)) for x in values)
        ):
            hp[key] = float(np.random.uniform(values[0], values[1]))
        else:
            value = np.random.choice(values)
            hp[key] = value.item() if isinstance(value, np.generic) else value

    return hp


# ---------------------------------------------------------------------------
# train_and_evaluate_ebm
# ---------------------------------------------------------------------------

def train_and_evaluate_ebm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hyperparameters: Dict[str, Any],
    fixed_hp: Dict[str, Any],
    is_regression: bool,
) -> Tuple[float, float]:
    # Import here to keep startup fast / similar to notebook
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

    if is_regression:
        model = ExplainableBoostingRegressor(**hyperparameters, **fixed_hp)
    else:
        model = ExplainableBoostingClassifier(**hyperparameters, **fixed_hp)

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    if is_regression:
        y_pred = model.predict(X_val)
        score = float(np.sqrt(mean_squared_error(y_val, y_pred)))  # RMSE (lower better)
    else:
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        score = float(roc_auc_score(y_val, y_pred_proba))          # AUC (higher better)

    return score, training_time


# ---------------------------------------------------------------------------
# Plotting helpers (added)
# ---------------------------------------------------------------------------

def save_tuning_plots(df_success: pd.DataFrame, out_dir: Path, dataset_name: str, is_regression: bool) -> None:
    import matplotlib.pyplot as plt

    metric_name = "RMSE" if is_regression else "AUC"

    # Plot validation score vs trial
    plt.figure()
    plt.plot(df_success["trial"].values, df_success["validation_score"].values)
    plt.xlabel("Trial")
    plt.ylabel(f"Validation {metric_name}")
    plt.title(f"EBM tuning: {dataset_name}")
    score_path = out_dir / f"tuning_curve_{dataset_name}.png"
    plt.savefig(score_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Plot training time vs trial
    plt.figure()
    plt.plot(df_success["trial"].values, df_success["training_time"].values)
    plt.xlabel("Trial")
    plt.ylabel("Training time (s)")
    plt.title(f"EBM tuning time: {dataset_name}")
    time_path = out_dir / f"training_time_{dataset_name}.png"
    plt.savefig(time_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    OPENML_DATASET_ID = args.dataset_id
    TASK_TYPE = args.task_type
    n_trials = args.n_trials
    random_seed = args.random_seed

    # project root + src + data_utils
    project_root = find_project_root(Path(__file__).resolve().parent)
    ensure_src_on_path(project_root)
    import neural_additive_models.data_utils as data_utils

    # config objects
    dataset_name = f"OpenML_{OPENML_DATASET_ID}_{TASK_TYPE}"
    is_regression = (TASK_TYPE == "regression")
    hp_search_space = get_hp_search_space()
    fixed_hp = get_fixed_hp(random_seed)

    results_dir = project_root / "results" / "hyperparameter_tuning" / "ebm"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Output filenames
    safe_name = dataset_name.replace("/", "_").replace(":", "_")
    tag = f"_{args.run_tag}" if args.run_tag else ""

    best_hp_file = results_dir / f"best_hp_{safe_name}{tag}.json"
    results_file = results_dir / f"tuning_results_{safe_name}{tag}.json"

    # Guard against overwrite
    if not args.overwrite and (best_hp_file.exists() or results_file.exists()):
        raise FileExistsError(
            f"Refusing to overwrite existing files:\n"
            f"  {best_hp_file}\n"
            f"  {results_file}\n"
            f"Use --overwrite or --run_tag to write new outputs."
        )

    # load dataset + split
    print(f"Loading dataset: {dataset_name}")
    data_x, data_y, column_names = data_utils.load_dataset(dataset_name)

    print(f"Dataset shape: {data_x.shape}")
    print(f"Target shape: {data_y.shape}")
    print(f"Number of features: {data_x.shape[1]}")
    print(f"Task type: {'Regression' if is_regression else 'Classification'}")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        data_x,
        data_y,
        test_size=args.test_size,
        random_state=random_seed,
        stratify=data_y if not is_regression else None,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=args.val_size,
        random_state=random_seed,
        stratify=y_train_val if not is_regression else None,
    )

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # generate hyperparameter configurations
    hyperparameters: List[Dict[str, Any]] = []
    for trial in range(n_trials):
        trial_seed = random_seed + trial
        hp_config = sample_hyperparameters(hp_search_space, trial_seed)
        hyperparameters.append({"trial": trial + 1, "hyperparameters": hp_config})

    print(f"Generated {n_trials} hyperparameter configurations")

    # tuning loop
    print("=" * 70)
    print(f"HYPERPARAMETER TUNING - {n_trials} trials")
    print("=" * 70)

    trial_results: List[Dict[str, Any]] = []
    metric_name = "RMSE" if is_regression else "AUC"

    for trial_data in hyperparameters:
        trial_num = trial_data["trial"]
        hp = trial_data["hyperparameters"]

        print(f"\nTrial {trial_num}/{n_trials}...", end=" ", flush=True)

        try:
            score, train_time = train_and_evaluate_ebm(
                X_train, y_train, X_val, y_val,
                hyperparameters=hp,
                fixed_hp=fixed_hp,
                is_regression=is_regression,
            )

            print(f"{metric_name}: {score:.4f} ({train_time:.1f}s)")
            trial_results.append(
                {
                    "trial": trial_num,
                    "hyperparameters": hp,
                    "validation_score": score,
                    "training_time": train_time,
                    "success": True,
                }
            )
        except Exception as e:
            print(f"Failed: {str(e)[:80]}")
            trial_results.append(
                {
                    "trial": trial_num,
                    "hyperparameters": hp,
                    "validation_score": None,
                    "training_time": None,
                    "success": False,
                    "error": str(e),
                }
            )

    print("\n" + "=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)

    # select best + save
    df_results = pd.DataFrame(trial_results)
    df_success = df_results[df_results["success"]].copy()

    if len(df_success) == 0:
        print("No successful trials!")
        # still save raw results for debugging
        with open(results_file, "w") as f:
            json.dump(trial_results, f, indent=2)
        print(f"Saved all results to: {results_file}")
        return 1

    if is_regression:
        best_idx = df_success["validation_score"].idxmin()  # lower RMSE better
    else:
        best_idx = df_success["validation_score"].idxmax()  # higher AUC better

    best_trial = df_success.loc[best_idx]

    print(f"Best trial: {int(best_trial['trial'])}")
    print(f"Best {metric_name}: {best_trial['validation_score']:.4f}")
    print(f"Training time: {best_trial['training_time']:.1f}s")
    print("\nBest hyperparameters:")
    for k, v in best_trial["hyperparameters"].items():
        print(f"  {k}: {v}")

    with open(best_hp_file, "w") as f:
        json.dump(best_trial["hyperparameters"], f, indent=2)
    print(f"\nSaved best hyperparameters to: {best_hp_file}")

    with open(results_file, "w") as f:
        json.dump(trial_results, f, indent=2)
    print(f"Saved all results to: {results_file}")

    # Summary stats (same spirit as notebook)
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Successful trials: {len(df_success)}/{n_trials}")
    print(f"Mean {metric_name}: {df_success['validation_score'].mean():.4f}")
    print(f"Std {metric_name}: {df_success['validation_score'].std():.4f}")
    print(f"Mean training time: {df_success['training_time'].mean():.1f}s")

    # Added: plots
    try:
        save_tuning_plots(df_success, results_dir, safe_name + tag, is_regression=is_regression)
        print(f"Saved plots to: {results_dir}")
    except Exception as e:
        print(f"Plot saving failed (non-fatal): {e}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())