from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_project_root(start: Path) -> Path:
    """
    Walk upwards until we find 'src' and 'config' directories.
    Falls back to start if not found.
    """
    for p in [start, *start.parents]:
        if (p / "src").is_dir() and (p / "config").is_dir():
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
        description="NAM hyperparameter tuning for OpenML (classification/regression), notebook-equivalent.",
    )
    p.add_argument("--dataset_id", type=int, required=True, help="OpenML dataset ID (e.g., 31, 1462, 44959).")
    p.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["classification", "regression"],
        help="Task type.",
    )
    p.add_argument("--n_trials", type=int, default=50, help="Number of random search trials (default: 50).")
    p.add_argument("--random_seed", type=int, default=42, help="Base random seed (default: 42).")
    p.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip a trial if its results already exist (default: False).",
    )
    p.add_argument(
        "--per_split_timeout_s",
        type=int,
        default=None,
        help="Optional per-split timeout in seconds (default: None).",
    )
    p.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional suffix appended to the tuning logdir name (prevents overwriting).",
    )

    # Optional training stage (cells 8-11)
    p.add_argument(
        "--train_after_tuning",
        action="store_true",
        help="After selecting best_hp.json, run training across folds/splits (like the notebook).",
    )
    p.add_argument("--num_folds", type=int, default=5, help="Num folds for training stage (default: 5).")
    p.add_argument(
        "--num_splits",
        type=int,
        default=None,
        help="Num splits for training stage. If not set, uses training config num_splits or defaults to 20.",
    )
    p.add_argument(
        "--cpu_only",
        action="store_true",
        help="Set CUDA_VISIBLE_DEVICES='' for the training subprocess calls (default: False).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # ============================================================================
    # imports + path setup + dataset params
    # ============================================================================
    args = parse_args()
    OPENML_DATASET_ID = args.dataset_id
    TASK_TYPE = args.task_type

    project_root = find_project_root(Path(__file__).resolve().parent)
    ensure_src_on_path(project_root)

    from hp_tuning_utils import (
        load_hp_tuning_conf,
        sample_hyperparameters,
        run_trials,
        build_validation_scores_df,
        select_and_save_best_hp,
        generate_training_command,
    )

    # ============================================================================
    # load hp tuning config
    # ============================================================================
    config_dir = project_root / "config" / "hp_tuning"
    if TASK_TYPE == "classification":
        CONF_PATH = config_dir / "openml_classification.json"
    else:
        CONF_PATH = config_dir / "openml_regression.json"

    hp_search_space, fixed_hp = load_hp_tuning_conf(CONF_PATH)
    fixed_hp["dataset_name"] = f"OpenML_{OPENML_DATASET_ID}_{TASK_TYPE}"
    print(f"Loaded HP tuning config: {CONF_PATH}")
    print(f"Dataset: {fixed_hp['dataset_name']}")

    # ============================================================================
    # random search parameters + base_logdir
    # ============================================================================
    n_trials = args.n_trials
    random_seed = args.random_seed

    results_dir = project_root / "results" / "hyperparameter_tuning" / "nam"

    logdir_name = f"hp_tuning_openml_{OPENML_DATASET_ID}_{TASK_TYPE}"
    if args.run_tag:
        logdir_name = f"{logdir_name}_{args.run_tag}"

    base_logdir = results_dir / logdir_name
    base_logdir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {base_logdir}")

    # ============================================================================
    # sample hyperparameters
    # ============================================================================
    hyperparameters: List[Dict[str, Any]] = []
    for trial in range(n_trials):
        trial_seed = random_seed + trial
        hp_config = sample_hyperparameters(hp_search_space, trial_seed)
        trial_num = trial + 1
        hyperparameters.append({"trial": trial_num, "hyperparameters": hp_config.copy()})

    # ============================================================================
    # run trials
    # ============================================================================
    num_splits = fixed_hp["num_splits"]
    trials_to_run = range(1, n_trials + 1)

    run_trials(
        fixed_hp=fixed_hp,
        hyperparameters=hyperparameters,
        base_logdir=base_logdir,
        n_trials=n_trials,
        num_splits=num_splits,
        trials_to_run=trials_to_run,
        skip_if_exists=args.skip_if_exists,
        per_split_timeout_s=args.per_split_timeout_s,
    )

    # ============================================================================
    # build validation df
    # ============================================================================
    split_results_files = list(Path(base_logdir).glob("trial_*_split_results.json"))
    if len(split_results_files) == 0:
        raise FileNotFoundError(
            f"No trial_*_split_results.json files found in {base_logdir}. "
            f"Check that run_trials wrote results and that you have permissions."
        )

    df = build_validation_scores_df(split_results_files, base_logdir, fixed_hp)
    print(f"Built validation scores DF with shape: {df.shape}")

    # ============================================================================
    # select best trial and save best_hp.json
    # ============================================================================
    best_trial, best_data = select_and_save_best_hp(df, base_logdir)
    print(f"Best trial: {best_trial}")
    print(f"Saved best_hp.json to: {base_logdir / 'best_hp.json'}")

    # ============================================================================
    # training stage using best_hp
    # ============================================================================
    if args.train_after_tuning:
        import subprocess

        best_hp_path = base_logdir / "best_hp.json"
        with open(best_hp_path, "r") as f:
            best_hp = json.load(f)

        # Load fixed training parameters
        training_config_dir = project_root / "config" / "training"
        if TASK_TYPE == "classification":
            training_config_path = training_config_dir / "nam_training_parameters_openml_classification.json"
        else:
            training_config_path = training_config_dir / "nam_training_parameters_openml_regression.json"

        with open(training_config_path, "r") as f:
            fixed_training_params = json.load(f)

        training_results_dir = project_root / "results" / "training" / "nam"
        training_logdir = training_results_dir / f"openml_{OPENML_DATASET_ID}_{TASK_TYPE}"
        training_logdir.mkdir(parents=True, exist_ok=True)
        print(f"Training results will be saved to: {training_logdir}")

        # Combine fixed training params + best hp
        all_params = {**fixed_training_params, **best_hp}
        all_params["dataset_name"] = fixed_hp["dataset_name"]
        all_params["regression"] = fixed_hp.get("regression", False)

        num_folds = args.num_folds
        train_num_splits = args.num_splits if args.num_splits is not None else fixed_training_params.get("num_splits", 20)

        print("Training configuration:")
        print(f"  Dataset: {all_params['dataset_name']}")
        print(f"  Task: {'Regression' if all_params['regression'] else 'Classification'}")
        print(f"  Folds: {num_folds}")
        print(f"  Splits per fold: {train_num_splits}")
        print(f"  Logdir: {training_logdir}")

        # Set env (PYTHONPATH + optional CPU-only)
        src_path = str(project_root / "src")
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
        if args.cpu_only:
            env["CUDA_VISIBLE_DEVICES"] = ""

        # Run training commands fold/split
        for fold in range(1, num_folds + 1):
            print(f"\n{'='*70}")
            print(f"Training Fold {fold}/{num_folds}")
            print(f"{'='*70}")

            for split in range(1, train_num_splits + 1):
                # Build a command string
                cmd_parts = generate_training_command(
                    all_params=all_params,
                    logdir=str(training_logdir),
                    fold=fold,
                    split=split,
                )

                # In case generate_training_command returns either a list or a string
                if isinstance(cmd_parts, list):
                    cmd = " ".join(cmd_parts)
                else:
                    cmd = str(cmd_parts)

                print(f"\nFold {fold}, Split {split}/{train_num_splits}...")
                print(f"Command: {cmd[:120]}...")

                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    env=env,
                )

                if result.returncode == 0:
                    print("  ✅ Completed successfully")
                else:
                    print(f"  ❌ Failed (return code: {result.returncode})")
                    if result.stderr:
                        err = result.stderr.strip()
                        print(f"  Error (tail): ...{err[-500:]}" if len(err) > 500 else f"  Error: {err}")
                    if result.stdout:
                        out = result.stdout.strip()
                        if ("Error" in out) or ("Traceback" in out):
                            print(f"  Output (tail): ...{out[-500:]}" if len(out) > 500 else f"  Output: {out}")

        print(f"\n{'='*70}")
        print("Training completed!")
        print(f"Results saved to: {training_logdir}")
        print(f"{'='*70}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
