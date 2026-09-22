#!/usr/bin/env python3
"""Print (or run with --run) the full training command matching the best Optuna trial.

    python scripts/launch_best_training.py SNN_L12_parallel
    python scripts/launch_best_training.py UNet2dHLCC_parallel --epochs 200 --n_samples 2000 --run

Every searched parameter AND every fixed one (num_layers, k, graph_type, physics, batch size)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study_name")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--nproc", type=int, default=8, help="GPUs for torchrun (1 = plain python)")
    parser.add_argument("--lr-scale", type=float, default=1.0, help="Multiply the searched lr (e.g. for a larger effective batch)")
    parser.add_argument("--run", action="store_true", help="Actually start the training (default: only print the command)")
    args = parser.parse_args()

    best = json.loads((PROJECT_ROOT / "outputs/2d/optuna" / args.study_name / "best_params.json").read_text())
    params, fixed = dict(best["params"]), best["fixed"]
    params["lr"] = params["lr"] * args.lr_scale
    launcher = [".venv/bin/torchrun", f"--nproc_per_node={args.nproc}"] if args.nproc > 1 else [".venv/bin/python"]
    cmd = launcher + ["src_2D/train.py", "--model", best["model"], "--epochs", str(args.epochs),
                      "--n_samples", str(args.n_samples), "--n_val", str(args.n_val),
                      "--batch_size", str(fixed["batch_size"]), "--physics", fixed["physics"], "--use-wandb"]
    if best["model"] in ("GCN", "SNN"):
        cmd += ["--num_layers", str(fixed["num_layers"]), "--k", str(fixed["k"]), "--graph_type", fixed["graph_type"]]
    for key, value in params.items():
        cmd += [f"--{key}", str(value)]

    print(f"Best trial #{best['trial_number']} of '{args.study_name}': {best}")
    print("\n" + " ".join(shlex.quote(c) for c in cmd) + "\n")
    if args.run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
