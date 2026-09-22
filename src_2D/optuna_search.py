"""Unified Optuna search, built on the exact training loop of src_2D/train.py.

    python src_2D/optuna_search.py --model UNet2dHLCC --n-trials 50
    python src_2D/optuna_search.py --model SNN --num_layers 12 --n-trials 50

One worker = one GPU: launch several workers on the same study with
``CUDA_VISIBLE_DEVICES=<i>`` (see scripts/launch_optuna_8gpus.sh).

- Objective: validation MSE on the missing wedge at the best epoch (minimised), i.e. the same
  quantity that selects checkpoints in train.py.
- For the graph models, the ablation variables (num_layers, k, graph_type) are NOT searched:
  they are fixed by the CLI so that every depth gets its own study.
- Trials never write model checkpoints, so a search can never overwrite a trained model.
- ``best_params.json`` is rewritten after EVERY completed trial (a killed worker loses nothing).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_2D.models.factory import GRAPH_MODELS, LEARNED_MODELS, default_run_name
from src_2D.train import SELECTION_METRIC, build_parser as build_train_parser, model_config_from_args, run_training, uses_physics


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search (unified protocol)")
    parser.add_argument("--model", type=str, required=True, choices=LEARNED_MODELS)
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials for this worker")
    parser.add_argument("--n-epochs", type=int, default=30, help="Epochs per trial")
    parser.add_argument("--n-samples", type=int, default=500, help="Training samples per trial")
    parser.add_argument("--n-val", type=int, default=100, help="Validation samples per trial")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--physics", type=str, default="auto", choices=["auto", "none", "hlcc"])
    # Ablation variables of the graph models: fixed, never searched.
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--graph_type", type=str, default="knn", choices=["knn", "full"])
    parser.add_argument("--grad_checkpoint", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--storage", type=str, default=None, help="Defaults to sqlite:///db/optuna_<study>.db")
    parser.add_argument("--study-name", type=str, default=None, help="Defaults to <run_name>_parallel")
    return parser.parse_args()


def make_train_args(args, trial: optuna.Trial):
    """Translate one Optuna trial into the argparse namespace consumed by train.run_training."""
    argv = [
        "--model", args.model, "--epochs", str(args.n_epochs), "--n_samples", str(args.n_samples),
        "--n_val", str(args.n_val), "--batch_size", str(args.batch_size), "--physics", args.physics,
        "--num_layers", str(args.num_layers), "--k", str(args.k), "--graph_type", args.graph_type,
    ]
    train_args = build_train_parser().parse_args(argv)
    train_args.grad_checkpoint = args.grad_checkpoint
    train_args.use_wandb = args.use_wandb
    train_args.wandb_project = "dbt-sinogram-optuna-2d"

    train_args.lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    train_args.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    if args.model in GRAPH_MODELS:
        train_args.num_stalks = trial.suggest_categorical("num_stalks", [16, 32, 64])
        train_args.sigma_deg = trial.suggest_categorical("sigma_deg", [2.0, 3.5, 5.0, 10.0])
    else:
        train_args.filters = trial.suggest_categorical("filters", [16, 32, 64])
    if uses_physics(train_args):
        train_args.lambda_m0 = trial.suggest_float("lambda_m0", 1e-3, 1.0, log=True)
        train_args.lambda_m1 = trial.suggest_float("lambda_m1", 1e-3, 1.0, log=True)
        train_args.anneal_epochs = trial.suggest_int("anneal_epochs", 2, max(3, args.n_epochs // 2))

    train_args.run_name = f"{args.study_name}_trial{trial.number}"
    return train_args


def ensure_sqlite_wal(storage_url: str) -> None:
    """WAL mode avoids database locking when several workers share one SQLite study."""
    if storage_url.startswith("sqlite:///"):
        db_path = Path(storage_url.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), timeout=60)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=60000;")
        conn.commit()
        conn.close()


def main():
    args = parse_args()
    base_config = model_config_from_args(build_train_parser().parse_args(
        ["--model", args.model, "--num_layers", str(args.num_layers), "--k", str(args.k)]))
    run_name = default_run_name(args.model, base_config)
    args.study_name = args.study_name or f"{run_name}_parallel"
    args.storage = args.storage or f"sqlite:///{PROJECT_ROOT / 'db' / f'optuna_{args.study_name}.db'}"
    best_params_path = PROJECT_ROOT / "outputs/2d/optuna" / args.study_name / "best_params.json"

    ensure_sqlite_wal(args.storage)
    storage = optuna.storages.RDBStorage(url=args.storage, engine_kwargs={"connect_args": {"timeout": 60}})
    study = optuna.create_study(
        direction="minimize", study_name=args.study_name, storage=storage, load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=5),
    )

    def objective(trial: optuna.Trial) -> float:
        best = run_training(make_train_args(args, trial), trial=trial, save=False)
        for key, value in best.items():
            trial.set_user_attr(key, value)
        return best[SELECTION_METRIC]

    def write_best_params(study: optuna.Study, _trial) -> None:
        try:
            best = study.best_trial
        except ValueError:
            return
        best_params_path.parent.mkdir(parents=True, exist_ok=True)
        best_params_path.write_text(json.dumps({
            "model": args.model, "study_name": args.study_name, f"best_{SELECTION_METRIC}": best.value,
            "trial_number": best.number, "user_attrs": best.user_attrs, "params": best.params,
            "fixed": {"num_layers": args.num_layers, "k": args.k, "graph_type": args.graph_type,
                      "physics": args.physics, "batch_size": args.batch_size},
        }, indent=2))

    print(f"Optuna study '{args.study_name}' ({args.storage})")
    study.optimize(objective, n_trials=args.n_trials, callbacks=[write_best_params])
    print(f"Best {SELECTION_METRIC}: {study.best_value:.6f} with {study.best_params}\nSaved to {best_params_path}")


if __name__ == "__main__":
    main()
