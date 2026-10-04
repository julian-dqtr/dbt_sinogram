#!/usr/bin/env python3
"""Print (or run with --run) the full training command matching the best Optuna trial.

    python scripts/launch_best_training.py SNN_L12_parallel
    python scripts/launch_best_training.py UNet2dHLCC_parallel --epochs 200 --n_samples 2000 --run
    python scripts/launch_best_training.py GCN_L6_parallel_optimizer_only --run
    python scripts/launch_best_training.py GCN_L12_parallel_optimizer_only --model SNN --run   # same configuration, SNN
    python scripts/launch_best_training.py GCN_L18_parallel_optimizer_only --seed 1 --run      # replica: run GCN_L18_S1

Every searched parameter AND every fixed one (num_layers, k, graph_type, physics, batch size,
plus the architecture recorded by the best trial: num_stalks, sigma_deg, grad_checkpoint...)
is forwarded, so the final run is the configuration that Optuna actually evaluated.
Note: with N GPUs the effective batch size is N x batch_size; the learning rate found with a
single GPU is NOT rescaled automatically (see --lr-scale).
"""
import argparse
import json
import shlex
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_MODELS = ("GCN", "SNN")  # as in src_2D/models/factory.py (not imported: this script needs no torch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study_name")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--nproc", type=int, default=8, help="GPUs for torchrun (1 = plain python)")
    parser.add_argument("--lr-scale", type=float, default=1.0, help="Multiply the searched lr (e.g. for a larger effective batch)")
    parser.add_argument("--run", action="store_true", help="Actually start the training (default: only print the command)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Another seed (initialisation and batch order; the phantoms stay the same) gives a replica "
                             "saved as <run>_S<seed>, to measure the run-to-run noise")
    parser.add_argument("--model", choices=GRAPH_MODELS, default=None,
                        help="Train the other graph model with this study's configuration (e.g. SNN with a GCN study): "
                             "same graph, depth, parameters and optimiser, only the transport differs")
    args = parser.parse_args()

    best = json.loads((PROJECT_ROOT / "outputs/2d/optuna" / args.study_name / "best_params.json").read_text())
    params, fixed = dict(best["params"]), best["fixed"]
    model = args.model or best["model"]
    if args.model and best["model"] not in GRAPH_MODELS:
        parser.error(f"--model swaps graph models only, but '{args.study_name}' is a {best['model']} study.")
    if "lr" in params:
        params["lr"] = params["lr"] * args.lr_scale
    elif args.lr_scale != 1.0:
        parser.error("--lr-scale needs a searched lr, but this study kept the default lr of train.py (--physics-only).")
    launcher = [".venv/bin/torchrun", f"--nproc_per_node={args.nproc}"] if args.nproc > 1 else [".venv/bin/python"]
    cmd = launcher + ["src_2D/train.py", "--model", model, "--epochs", str(args.epochs),
                      "--n_samples", str(args.n_samples), "--n_val", str(args.n_val),
                      "--batch_size", str(fixed["batch_size"]), "--physics", fixed["physics"], "--use-wandb"]
    if model in GRAPH_MODELS:
        cmd += ["--num_layers", str(fixed["num_layers"]), "--k", str(fixed["k"]), "--graph_type", fixed["graph_type"]]
    if args.seed != 0:  # same names as factory.run_name_of("GCN_L18_S1")
        run_name = f"{model}_L{fixed['num_layers']}" if model in GRAPH_MODELS else model
        cmd += ["--seed", str(args.seed), "--run_name", f"{run_name}_S{args.seed}"]
    if fixed.get("hlcc_max_order", 1) != 1:  # older studies only knew the orders 0 and 1
        cmd += ["--hlcc_max_order", str(fixed["hlcc_max_order"])]
    for key, value in params.items():
        cmd += [f"--{key}", str(value)]
    # Architecture of the best trial (recorded by optuna_search.py): a later change of a train.py
    # default can then never change the final model. Older studies do not record it.
    forwarded = set(params) | {"num_layers", "k", "graph_type"}
    for key, value in best["user_attrs"].get("model_config", {}).items():
        if key in forwarded:
            continue
        if isinstance(value, bool):
            cmd += [f"--{key}"] if value else []
        else:
            cmd += [f"--{key}", str(value)]

    print(f"Best trial #{best['trial_number']} of '{args.study_name}': {best}")
    if model != best["model"]:
        print(f"Model swapped: {best['model']} -> {model} (same configuration)")
    print("\n" + " ".join(shlex.quote(c) for c in cmd) + "\n")
    if args.run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
