from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ----------------------------
# Path setup
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EBM OpenML analysis (evaluation + plots).")
    p.add_argument("--dataset_id", type=int, required=True, help="OpenML dataset ID (e.g., 31, 1464, 41021).")
    p.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["classification", "regression"],
        help="Task type.",
    )
    p.add_argument("--num_folds", type=int, default=5, help="Number of folds (default: 5).")
    p.add_argument("--num_splits", type=int, default=20, help="Number of splits per fold (default: 20).")
    p.add_argument("--random_state", type=int, default=42, help="Random seed used for folds + fixed params (default: 42).")
    p.add_argument("--print_every", type=int, default=5, help="Print progress every N splits (default: 5).")
    p.add_argument("--save_plots", action="store_true", help="Save plots to results/evaluation/plots/")
    p.add_argument(
        "--y_limits",
        type=float,
        nargs=2,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Optional y-axis limits for EBM shape plot, e.g. --y_limits -0.2 0.2",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite plots if they exist (JSON naming is handled by save_performance_metrics).",
    )
    return p.parse_args()


def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def compute_feature_importance_from_shapes(
    all_shape_functions: List[Dict],
    column_names: List[str],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Very simple across-splits importance:
      importance(feature, split) = mean(|y|) of the shape curve for that feature in that split.
    Returns sorted feature names, mean importance, std importance.
    """
    # Collect per-split importances
    split_imps = []
    for shape_data in all_shape_functions:
        imps = []
        for feat in column_names:
            d = shape_data.get(feat)
            if not d or "y" not in d or d["y"] is None or len(d["y"]) == 0:
                imps.append(np.nan)
                continue
            y = np.asarray(d["y"], dtype=float)
            imps.append(float(np.nanmean(np.abs(y))))
        split_imps.append(imps)

    M = np.asarray(split_imps, dtype=float)  # [n_splits_total, n_features]
    mean_imp = np.nanmean(M, axis=0)
    std_imp = np.nanstd(M, axis=0)

    order = np.argsort(-mean_imp)  # descending
    sorted_feats = [column_names[i] for i in order]
    return sorted_feats, mean_imp[order], std_imp[order]


def plot_feature_importance_across_splits_from_shapes(
    sorted_features: List[str],
    sorted_mean: np.ndarray,
    sorted_std: np.ndarray,
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(sorted_features))
    plt.figure(figsize=(max(10, len(sorted_features) * 0.35), 5))
    plt.bar(x, sorted_mean, yerr=sorted_std)
    plt.xticks(x, sorted_features, rotation=90)
    plt.ylabel("Importance (mean |shape| across splits)")
    plt.title("EBM feature importance across splits (from shape functions)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> int:
    args = parse_args()
    dataset_id = args.dataset_id
    task_type = args.task_type
    is_regression = (task_type == "regression")

    import neural_additive_models.data_utils as data_utils
    from utils import (
        create_fold_indices,
        gather_ebm_predictions_and_shape_functions,
        save_performance_metrics,
        plot_ebm_shape_functions,
    )

    dataset_name = f"OpenML_{dataset_id}_{task_type}"
    print(f"Loading dataset: {dataset_name}")
    data_x, data_y, column_names = data_utils.load_dataset(dataset_name)

    print(f"Dataset shape: {data_x.shape}")
    print(f"Target shape: {data_y.shape}")
    print(f"Number of features: {data_x.shape[1]}")
    print(f"Task: {'Regression' if is_regression else 'Classification'}")

    # Folds
    fold_train_indices, fold_test_indices = create_fold_indices(
        data_x,
        data_y,
        num_folds=args.num_folds,
        is_regression=is_regression,
        random_state=42,
    )

    # Load best HP
    best_hp_path = (
        PROJECT_ROOT
        / "results"
        / "hyperparameter_tuning"
        / "ebm"
        / f"best_hp_{_safe_name(dataset_name)}.json"
    )
    if not best_hp_path.exists():
        raise FileNotFoundError(
            f"EBM best hyperparameters not found at {best_hp_path}. "
            f"Run EBM hyperparameter tuning first."
        )

    with open(best_hp_path, "r") as f:
        best_hp = json.load(f)

    fixed_hp = {
        "random_state": 42,  # Hardcoded to match notebook exactly
        "n_jobs": -1,
        "early_stopping_rounds": 50,
        "validation_size": 0.125,
    }

    # Where EBM models are cached
    base_logdir = PROJECT_ROOT / "results" / "training" / "ebm" / f"openml_{dataset_id}_{task_type}"
    base_logdir.mkdir(parents=True, exist_ok=True)
    print(f"EBM model cache dir: {base_logdir}")

    # Gather preds + shape functions
    print("Gathering EBM predictions and shape functions (train/load).")
    all_preds_per_fold, all_shape_functions, all_mean_pred = gather_ebm_predictions_and_shape_functions(
        fold_test_indices=fold_test_indices,
        data_x=data_x,
        data_y=data_y,
        column_names=column_names,
        dataset_name=dataset_name,
        best_hp=best_hp,
        fixed_hp=fixed_hp,
        is_regression=is_regression,
        num_folds=args.num_folds,
        num_splits=args.num_splits,
        print_every=args.print_every,
        base_logdir=str(base_logdir),
    )

    # Save performance JSON
    perf_path = save_performance_metrics(
        all_preds_per_fold=all_preds_per_fold,
        fold_test_indices=fold_test_indices,
        data_y=data_y,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        task_type=task_type,
        model_type="EBM",
        num_folds=args.num_folds,
        num_splits=args.num_splits,
        is_regression=is_regression,
        project_root=PROJECT_ROOT,
        verbose=True,
    )
    print(f"Saved performance JSON: {perf_path}")

    # Save plots
    if args.save_plots:
        plots_dir = PROJECT_ROOT / "results" / "evaluation" / "plots" / f"ebm_OpenML_{dataset_id}_{task_type}"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # 1) Shape functions plot (across all splits)
        shape_plot_path = plots_dir / "ebm_shape_functions.png"
        if shape_plot_path.exists() and not args.overwrite:
            print(f"Plot exists (skip): {shape_plot_path} (use --overwrite to regenerate)")
        else:
            y_limits = tuple(args.y_limits) if args.y_limits is not None else (-0.1, 0.1)
            fig = plot_ebm_shape_functions(
                column_names=column_names,
                all_shape_functions=all_shape_functions,
                y_limits=y_limits,
            )
            fig.savefig(shape_plot_path, dpi=150, bbox_inches="tight")
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass
            print(f"Saved plot: {shape_plot_path}")

        # 2) Simple feature importance across splits (computed from shapes)
        fi_plot_path = plots_dir / "feature_importance_across_splits.png"
        if fi_plot_path.exists() and not args.overwrite:
            print(f"Plot exists (skip): {fi_plot_path} (use --overwrite to regenerate)")
        else:
            sorted_feats, sorted_mean, sorted_std = compute_feature_importance_from_shapes(
                all_shape_functions=all_shape_functions,
                column_names=column_names,
            )
            plot_feature_importance_across_splits_from_shapes(
                sorted_features=sorted_feats,
                sorted_mean=sorted_mean,
                sorted_std=sorted_std,
                save_path=fi_plot_path,
            )
            print(f"Saved plot: {fi_plot_path}")

        print(f"Plots directory: {plots_dir}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())