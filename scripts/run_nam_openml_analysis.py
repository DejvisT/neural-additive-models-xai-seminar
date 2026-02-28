from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR

import argparse
import sys
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Helpers: locate project root and ensure src/ is importable
# ---------------------------------------------------------------------------

def find_project_root(start: Path) -> Path:
    """
    Walk up from `start` until we find a folder containing 'src'.
    Falls back to start if not found.
    """
    for p in [start, *start.parents]:
        if (p / "src").is_dir():
            return p
    return start


def ensure_src_on_path(project_root: Path) -> None:
    src_path = project_root / "src"
    if src_path.is_dir() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate NAM on an OpenML dataset (classification or regression), save JSON + plots.",
    )
    parser.add_argument("--dataset_id", type=int, required=True, help="OpenML dataset id, e.g. 31")
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["classification", "regression"],
        help="Task type for dataset naming: OpenML_<id>_<task_type>",
    )
    parser.add_argument("--num_folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--num_splits", type=int, default=20, help="Number of splits/ensembles per fold (default: 20)")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for fold creation (default: 42)")
    parser.add_argument(
        "--y_limits",
        type=float,
        nargs=2,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Optional y-axis limits for contribution plot, e.g. --y_limits -0.1 0.1",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=5,
        help="Print progress every N splits when gathering checkpoints (default: 5)",
    )
    return parser.parse_args()


def main() -> int:
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    args = parse_args()
    OPENML_DATASET_ID = args.dataset_id
    TASK_TYPE = args.task_type
    NUM_FOLDS = args.num_folds
    NUM_SPLITS = args.num_splits
    RANDOM_STATE = args.random_state
    is_regression = (TASK_TYPE == "regression")

    # ============================================================================
    # Imports + path setup
    # ============================================================================
    project_root = find_project_root(Path(__file__).resolve().parent)
    ensure_src_on_path(project_root)

    import neural_additive_models.data_utils as data_utils
    from utils import (
        inverse_min_max_scaler,
        create_fold_indices,
        prepare_feature_arrays,
        gather_nam_predictions_and_hist_data,
        plot_nam_contributions_with_density,
        plot_feature_importance_across_splits,
        evaluate_ensemble_across_folds,
        save_performance_metrics,
    )

    # ============================================================================
    # Load dataset
    # ============================================================================
    dataset_name = f"OpenML_{OPENML_DATASET_ID}_{TASK_TYPE}"
    print(f"Loading dataset: {dataset_name}")
    data_x, data_y, column_names = data_utils.load_dataset(dataset_name)

    print(f"Dataset shape: {data_x.shape}")
    print(f"Target shape: {data_y.shape}")
    print(f"Number of features: {data_x.shape[1]}")
    print(f"Task type: {'Regression' if is_regression else 'Classification'}")

    # Compute per-column min/max in ORIGINAL (inverse-scaled) space
    col_min_max = {}
    for i, col_name in enumerate(column_names):
        vals = data_x[:, i]
        col_min_max[col_name] = (float(np.min(vals)), float(np.max(vals)))

    # ============================================================================
    # Create folds
    # ============================================================================
    fold_train_indices, fold_test_indices = create_fold_indices(
        data_x,
        data_y,
        num_folds=NUM_FOLDS,
        is_regression=is_regression,
        random_state=RANDOM_STATE,
    )

    # ============================================================================
    # Prepare feature arrays + config
    # ============================================================================
    SINGLE_FEATURES_ORIGINAL, UNIQUE_FEATURES_ORIGINAL, UNIQUE_FEATURES = prepare_feature_arrays(
        data_x, column_names, col_min_max, inverse_min_max_scaler
    )

    # Identity mappings (no special label formatting for OpenML features)
    COL_NAMES = {dataset_name: {name: name for name in column_names}}
    FEATURE_LABEL_MAPPING = {dataset_name: {name: name for name in column_names}}
    CATEGORICAL_NAMES = []

    # ============================================================================
    # Gather predictions + hist data
    # ============================================================================
    base_logdir = (
        project_root / "results" / "training" / "nam" / f"openml_{OPENML_DATASET_ID}_{TASK_TYPE}"
    )
    print(f"Using NAM checkpoint base_logdir: {base_logdir}")

    all_preds_per_fold, all_hist_data, all_mean_pred = gather_nam_predictions_and_hist_data(
        fold_test_indices=fold_test_indices,
        data_x=data_x,
        column_names=column_names,
        unique_features=UNIQUE_FEATURES,
        dataset_name=dataset_name,
        base_logdir=str(base_logdir),
        num_folds=NUM_FOLDS,
        num_splits=NUM_SPLITS,
        print_every=args.print_every,
    )

    # ============================================================================
    # Save plots
    # ============================================================================
    plots_dir = project_root / "results" / "evaluation" / "plots" / f"nam_openml_{OPENML_DATASET_ID}_{TASK_TYPE}"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Contributions plot
    contrib_plot_path = plots_dir / "nam_contributions_with_density.png"
    print(f"Saving contribution plot to: {contrib_plot_path}")

    y_limits = tuple(args.y_limits) if args.y_limits is not None else None

    _ = plot_nam_contributions_with_density(
        hist_data=all_hist_data,
        unique_features=UNIQUE_FEATURES_ORIGINAL,
        single_features=SINGLE_FEATURES_ORIGINAL,
        categorical_names=CATEGORICAL_NAMES,
        col_mapping=COL_NAMES[dataset_name],
        feature_mapping=FEATURE_LABEL_MAPPING[dataset_name],
        mean_pred=all_mean_pred,
        feature_to_use=column_names,
        y_limits=y_limits,
        save_path=contrib_plot_path,
    )

    # Feature importance plot
    fi_plot_path = plots_dir / "feature_importance_across_splits.png"
    print(f"Saving feature importance plot to: {fi_plot_path}")

    sorted_features, sorted_mean, sorted_std = plot_feature_importance_across_splits(
        all_hist_data,
        all_mean_pred,
        dataset_name,
        save_path=fi_plot_path,
    )

    # ============================================================================
    # Evaluate ensemble across folds
    # ============================================================================
    fold_metrics, avg_metric, std_metric = evaluate_ensemble_across_folds(
        all_preds_per_fold=all_preds_per_fold,
        fold_test_indices=fold_test_indices,
        data_y=data_y,
        is_regression=is_regression,
        verbose=True,
    )

    # ============================================================================
    # Save performance metrics JSON
    # ============================================================================
    output_file = save_performance_metrics(
        all_preds_per_fold=all_preds_per_fold,
        fold_test_indices=fold_test_indices,
        data_y=data_y,
        dataset_id=OPENML_DATASET_ID,
        dataset_name=dataset_name,
        task_type=TASK_TYPE,
        model_type="NAM",
        num_folds=NUM_FOLDS,
        num_splits=NUM_SPLITS,
        is_regression=is_regression,
        project_root=project_root,
        verbose=True,
    )

    print("\nDone.")
    print(f"Performance JSON: {output_file}")
    print(f"Plots directory:  {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())