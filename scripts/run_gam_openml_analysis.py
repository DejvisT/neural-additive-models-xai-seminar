from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score
from pygam import LinearGAM, LogisticGAM, s


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GAM hyperparameter tuning + fold training for OpenML.")
    p.add_argument("--dataset_id", type=int, required=True, help="OpenML dataset id, e.g. 31, 1068, 44959")
    p.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["classification", "regression"],
        help="Task type.",
    )
    p.add_argument("--n_trials", type=int, default=50, help="Number of tuning trials (default: 50)")
    p.add_argument("--random_seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--num_folds", type=int, default=5, help="Number of CV folds (default: 5)")
    p.add_argument("--skip_training", action="store_true", help="Only tune, do not train per-fold models")
    p.add_argument("--run_tag", type=str, default=None, help="Suffix to avoid overwriting outputs (e.g. refactor_test)")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting outputs")
    return p.parse_args()


def build_terms(n_features: int, n_splines: int, spline_order: int):
    """One smooth term per feature, no interactions (fair comparison vs NAM/EBM)."""
    term_list = [s(i, n_splines=n_splines, spline_order=spline_order) for i in range(n_features)]
    terms = term_list[0]
    for term in term_list[1:]:
        terms = terms + term
    return terms


def train_and_evaluate_gam(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hyperparameters: Dict[str, Any],
    fixed_hp: Dict[str, Any],
    is_regression: bool,
):
    """Train GAM and return validation score, training time, model, and convergence flag."""
    n_features = X_train.shape[1]
    n_splines = int(hyperparameters["n_splines"])
    lam = hyperparameters["lam"]
    spline_order = int(hyperparameters["spline_order"])

    terms = build_terms(n_features, n_splines=n_splines, spline_order=spline_order)

    if is_regression:
        gam = LinearGAM(terms=terms, lam=lam, fit_intercept=True, **fixed_hp)
    else:
        gam = LogisticGAM(terms=terms, lam=lam, fit_intercept=True, **fixed_hp)

    start_time = time.time()
    gam.fit(X_train, y_train)
    training_time = time.time() - start_time

    converged = getattr(gam, "converged_", True)

    if is_regression:
        y_pred = gam.predict(X_val)
        score = float(np.sqrt(mean_squared_error(y_val, y_pred)))  # RMSE (lower better)
    else:
        y_pred_proba = gam.predict_proba(X_val)
        score = float(roc_auc_score(y_val, y_pred_proba))          # AUC (higher better)

    return score, training_time, gam, converged


def gather_gam_predictions(
    fold_train_indices,
    fold_test_indices,
    data_x: np.ndarray,
    data_y: np.ndarray,
    column_names: List[str],
    best_hp: Dict[str, Any],
    fixed_hp: Dict[str, Any],
    is_regression: bool,
    num_folds: int,
    base_logdir: str | None,
):
    """
    Train or load one GAM per fold, cached as fold_k/gam_model.pkl.
    Returns all_preds_per_fold shaped [fold][1][samples] (single model per fold).
    """
    all_preds_per_fold = [[] for _ in range(num_folds)]

    n_features = data_x.shape[1]
    n_splines = int(best_hp["n_splines"])
    lam = best_hp["lam"]
    spline_order = int(best_hp["spline_order"])
    terms = build_terms(n_features, n_splines=n_splines, spline_order=spline_order)

    for fold in range(1, num_folds + 1):
        fold_idx = fold - 1
        train_indices = fold_train_indices[fold_idx]
        test_indices = fold_test_indices[fold_idx]

        X_train_fold = data_x[train_indices]
        y_train_fold = data_y[train_indices]
        X_test_fold = data_x[test_indices]

        print(f"Processing fold {fold} (train: {len(train_indices)}, test: {len(test_indices)})...", end=" ", flush=True)

        model_path = None
        if base_logdir:
            model_path = os.path.join(base_logdir, f"fold_{fold}", "gam_model.pkl")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if model_path and os.path.exists(model_path):
            with open(model_path, "rb") as f:
                gam = pickle.load(f)
            print("(loaded)", flush=True)
        else:
            if is_regression:
                gam = LinearGAM(terms=terms, lam=lam, fit_intercept=True, **fixed_hp)
            else:
                gam = LogisticGAM(terms=terms, lam=lam, fit_intercept=True, **fixed_hp)
            gam.fit(X_train_fold, y_train_fold)

            if model_path:
                with open(model_path, "wb") as f:
                    pickle.dump(gam, f)

            print("(trained)", flush=True)

        if is_regression:
            preds_test = gam.predict(X_test_fold)
        else:
            preds_test = gam.predict_proba(X_test_fold)

        all_preds_per_fold[fold_idx].append(preds_test)

    return all_preds_per_fold


def main() -> int:
    args = parse_args()

    # --- Project root + src path ---
    project_root = Path(__file__).resolve().parent.parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    import neural_additive_models.data_utils as data_utils
    from neural_additive_models.data_utils import get_train_val_test_split
    from utils import create_fold_indices, save_performance_metrics
    from hp_tuning_utils import sample_hyperparameters

    # --- Config ---
    OPENML_DATASET_ID = args.dataset_id
    TASK_TYPE = args.task_type
    dataset_name = f"OpenML_{OPENML_DATASET_ID}_{TASK_TYPE}"
    is_regression = (TASK_TYPE == "regression")

    hp_search_space = {
        "n_splines": [6, 8, 12, 16, 20, 24],
        "lam": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        "spline_order": [2, 3, 4, 5],
    }
    fixed_hp = {"max_iter": 100, "tol": 1e-4}

    n_trials = args.n_trials
    random_seed = args.random_seed
    NUM_FOLDS = args.num_folds

    results_dir = project_root / "results" / "hyperparameter_tuning" / "gam"
    results_dir.mkdir(parents=True, exist_ok=True)

    safe_ds = _safe_name(dataset_name)
    tag = f"_{args.run_tag}" if args.run_tag else ""
    best_hp_file = results_dir / f"best_hp_{safe_ds}{tag}.json"
    results_file = results_dir / f"tuning_results_{safe_ds}{tag}.json"

    if not args.overwrite and (best_hp_file.exists() or results_file.exists()):
        raise FileExistsError(
            f"Refusing to overwrite:\n  {best_hp_file}\n  {results_file}\n"
            f"Use --overwrite or --run_tag."
        )

    # --- Load dataset + train/val/test split ---
    print(f"Loading dataset: {dataset_name}")
    data_x, data_y, column_names = data_utils.load_dataset(dataset_name)

    if "_regression" in dataset_name:
        is_regression = True
    elif "_classification" in dataset_name:
        is_regression = False

    print(f"Dataset shape: {data_x.shape}")
    print(f"Target shape: {data_y.shape}")
    print(f"Number of features: {data_x.shape[1]}")
    print(f"Task type: {'Regression' if is_regression else 'Classification'}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_train_val_test_split(
        data_x,
        data_y,
        test_size=0.2,
        val_size=0.2,
        stratified=not is_regression,
        random_state=random_seed,
    )
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # --- Hyperparameter configs ---
    hyperparameters = []
    for trial in range(n_trials):
        trial_seed = random_seed + trial
        hp_config = sample_hyperparameters(hp_search_space, trial_seed)
        hyperparameters.append({"trial": trial + 1, "hyperparameters": hp_config})
    print(f"Generated {n_trials} hyperparameter configurations")

    # --- Tuning loop ---
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
            score, training_time, model, converged = train_and_evaluate_gam(
                X_train, y_train, X_val, y_val,
                hyperparameters=hp,
                fixed_hp=fixed_hp,
                is_regression=is_regression,
            )
            if not converged:
                print("did not converge")
            print(f"{metric_name}: {score:.4f} ({training_time:.1f}s)")

            trial_results.append({
                "trial": trial_num,
                "hyperparameters": hp,
                "validation_score": score,
                "training_time": training_time,
                "converged": bool(converged),
                "success": True,
            })
        except Exception as e:
            print(f"Failed: {str(e)[:50]}")
            trial_results.append({
                "trial": trial_num,
                "hyperparameters": hp,
                "validation_score": None,
                "training_time": None,
                "success": False,
                "error": str(e),
            })

    print("\n" + "=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)

    # --- Select best + save ---
    df_results = pd.DataFrame(trial_results)
    df_success = df_results[df_results["success"]].copy()

    if len(df_success) == 0:
        print("No successful trials!")
        with open(results_file, "w") as f:
            json.dump(trial_results, f, indent=2)
        print(f"Saved all results to: {results_file}")
        return 1

    df_converged = df_success[df_success.get("converged", True)].copy() if "converged" in df_success.columns else df_success.copy()
    if len(df_converged) > 0:
        used_converged_only = True
        best_idx = df_converged["validation_score"].idxmin() if is_regression else df_converged["validation_score"].idxmax()
        best_trial = df_converged.loc[best_idx]
    else:
        used_converged_only = False
        print("WARNING: No trials converged! Using best from all trials (may be unstable).")
        best_idx = df_success["validation_score"].idxmin() if is_regression else df_success["validation_score"].idxmax()
        best_trial = df_success.loc[best_idx]

    best_hp = dict(best_trial["hyperparameters"])

    print(f"Best trial: {int(best_trial['trial'])}")
    if "converged" in best_trial and not bool(best_trial["converged"]):
        print("WARNING: Best trial did not converge! Model may be unstable.")
    print(f"Best {metric_name}: {float(best_trial['validation_score']):.4f}")
    print(f"Training time: {float(best_trial['training_time']):.1f}s")
    if "converged" in best_trial:
        print(f"Converged: {bool(best_trial['converged'])}")

    print("\nBest hyperparameters:")
    for k, v in best_hp.items():
        print(f"  {k}: {v}")

    with open(best_hp_file, "w") as f:
        json.dump(best_hp, f, indent=2)
    print(f"\nSaved best hyperparameters to: {best_hp_file}")

    with open(results_file, "w") as f:
        json.dump(trial_results, f, indent=2)
    print(f"Saved all results to: {results_file}")

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Successful trials: {len(df_success)}/{n_trials}")
    if "converged" in df_success.columns:
        n_converged = int(df_success["converged"].sum())
        print(f"Converged trials: {n_converged}/{len(df_success)}")
        if n_converged < len(df_success):
            print(f"Non-converged trials: {len(df_success) - n_converged} (excluded from best HP selection)")
    print(f"Mean {metric_name}: {df_success['validation_score'].mean():.4f}")
    print(f"Std {metric_name}: {df_success['validation_score'].std():.4f}")
    print(f"Mean training time: {df_success['training_time'].mean():.1f}s")

    # --- CV folds + train per fold + save performance ---
    if args.skip_training:
        print("\nSkipping fold-training and performance saving (--skip_training). Done.")
        return 0

    fold_train_indices, fold_test_indices = create_fold_indices(
        data_x, data_y, num_folds=NUM_FOLDS, random_state=42
    )
    print(f"\nCreated {NUM_FOLDS} folds for cross-validation")
    print("Note: GAMs train one model per fold (no splits / early stopping).")

    base_logdir = project_root / "results" / "training" / "gam" / f"openml_{OPENML_DATASET_ID}_{TASK_TYPE}"
    base_logdir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("TRAINING MODELS PER FOLD")
    print("=" * 70)
    print(f"Number of folds: {NUM_FOLDS}")
    print(f"Total models to train: {NUM_FOLDS}")
    print(f"Models will be saved to: {base_logdir}")
    print("=" * 70)
    print("\nGathering predictions (training models as needed)...")
    print("=" * 70)

    all_preds_per_fold = gather_gam_predictions(
        fold_train_indices=fold_train_indices,
        fold_test_indices=fold_test_indices,
        data_x=data_x,
        data_y=data_y,
        column_names=column_names,
        best_hp=best_hp,
        fixed_hp=fixed_hp,
        is_regression=is_regression,
        num_folds=NUM_FOLDS,
        base_logdir=str(base_logdir),
    )

    output_file = save_performance_metrics(
        all_preds_per_fold=all_preds_per_fold,
        fold_test_indices=fold_test_indices,
        data_y=data_y,
        dataset_id=OPENML_DATASET_ID,
        dataset_name=dataset_name,
        task_type=TASK_TYPE,
        model_type="GAM",
        num_folds=NUM_FOLDS,
        num_splits=1,  # GAMs: 1 model per fold
        is_regression=is_regression,
        project_root=project_root,
        verbose=True,
    )

    print(f"\nSaved performance JSON: {output_file}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())