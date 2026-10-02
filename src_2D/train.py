"""Unified training script: ONE protocol for every model of the thesis.

    python src_2D/train.py --model UNet2D
    python src_2D/train.py --model UNet2dHLCC
    python src_2D/train.py --model GCN --num_layers 12
    torchrun --nproc_per_node=8 src_2D/train.py --model SNN --num_layers 18 --use-wandb

Shared by all models: data (seeded train / val splits), data consistency, metrics
(src_2D/utils/metrics.py), checkpoint selection (validation MSE on the missing wedge, never
the annealed total loss), self-describing checkpoints and DDP-exact metric reduction.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.models.baselines import linear_interpolation, zero_filling
from src_2D.models.factory import GRAPH_MODELS, LEARNED_MODELS, build_model, default_run_name
from src_2D.models.SinoSheavesNN.physics_loss import AnnealedLoss
from src_2D.utils.checkpoint import save_checkpoint
from src_2D.utils.evaluation import generate_example_figure, save_random_dataset_preview, save_training_curve
from src_2D.utils.metrics import METRIC_KEYS, sinogram_metrics

# The Tesla K80 (Kepler) is not supported by the cuDNN shipped with recent PyTorch builds.
torch.backends.cudnn.enabled = False

SELECTION_METRIC = "mse_wedge"  # lower is better
CALIBRATION_SAMPLES = 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a sinogram completion model (unified protocol).")
    parser.add_argument("--model", type=str, required=True, choices=LEARNED_MODELS)
    # Optimisation
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4, help="Per-process batch size")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience in epochs (0 disables)")
    # Data
    parser.add_argument("--n_samples", type=int, default=2000, help="Size of the fixed training set")
    parser.add_argument("--n_val", type=int, default=200, help="Size of the fixed validation set")
    parser.add_argument("--noise_level", type=float, default=1e5, help="Poisson I0 (<= 0 disables noise)")
    parser.add_argument("--seed", type=int, default=0)
    # U-Net
    parser.add_argument("--filters", type=int, default=32)
    # Graph models (GCN / SNN)
    parser.add_argument("--num_stalks", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--sigma_deg", type=float, default=5.0)
    parser.add_argument("--graph_type", type=str, default="knn", choices=["knn", "full"])
    parser.add_argument("--normalization", type=str, default="sym", choices=["sym", "rw"])
    parser.add_argument("--grad_checkpoint", action="store_true", help="Save memory in deep graph models")
    # Physics (HLCC) loss. "auto": on for UNet2dHLCC, off for every other model.
    parser.add_argument("--physics", type=str, default="auto", choices=["auto", "none", "hlcc"])
    parser.add_argument("--lambda_m0", type=float, default=0.1)
    parser.add_argument("--lambda_m1", type=float, default=0.1)
    parser.add_argument("--anneal_epochs", type=int, default=20)
    # Outputs / logging
    parser.add_argument("--run_name", type=str, default=None, help="Defaults to <model> or <model>_L<num_layers>")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="dbt-sinogram-completion-2D")
    parser.add_argument("--wandb-mode", type=str, choices=("online", "offline", "disabled"), default="online")
    # Smoke testing
    parser.add_argument("--max_train_batches", type=int, default=0, help="Truncate every epoch (0 = full epoch)")
    return parser


def model_config_from_args(args) -> Dict:
    if args.model in GRAPH_MODELS:
        return {
            "num_stalks": args.num_stalks, "num_layers": args.num_layers, "k": args.k,
            "sigma_deg": args.sigma_deg, "graph_type": args.graph_type,
            "normalization": args.normalization, "grad_checkpoint": args.grad_checkpoint,
        }
    return {"filters": args.filters}


def uses_physics(args) -> bool:
    if args.physics == "auto":
        return args.model == "UNet2dHLCC"
    return args.physics == "hlcc"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _reduce_sum(values: torch.Tensor, is_distributed: bool) -> torch.Tensor:
    if is_distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


@torch.no_grad()
def evaluate(predict_fn, loader, missing_views: np.ndarray, device, is_distributed: bool) -> Dict[str, float]:
    """Average ``sinogram_metrics`` over the samples actually processed (exact under DDP)."""
    sums = torch.zeros(len(METRIC_KEYS) + 1, dtype=torch.float64, device=device)
    for incomplete, target, _ in loader:
        pred = predict_fn(incomplete.to(device)).float().cpu().numpy()
        target_np = target.numpy()
        for i in range(pred.shape[0]):
            m = sinogram_metrics(pred[i, 0], target_np[i, 0], missing_views)
            sums[:-1] += torch.tensor([m[key] for key in METRIC_KEYS], dtype=torch.float64, device=device)
            sums[-1] += 1
    sums = _reduce_sum(sums, is_distributed)
    count = max(1.0, sums[-1].item())
    return {key: (sums[i] / count).item() for i, key in enumerate(METRIC_KEYS)}


def calibrate_physics(loss_fn: AnnealedLoss, val_dataset, device) -> Dict[str, float]:
    """Deterministic calibration on the zero-filled inputs of the first validation samples."""
    n = min(CALIBRATION_SAMPLES, len(val_dataset))
    reference = torch.stack([val_dataset[i][0][0] for i in range(n)]).to(device)  # [n, V, D]
    loss_fn.calibrate(reference)
    return {
        "scale_m0": loss_fn.physics_loss_fn.scale_m0.item(),
        "scale_m1": loss_fn.physics_loss_fn.scale_m1.item(),
        "calibration_samples": n,
    }


def run_training(args, trial=None, save: bool = True) -> Dict[str, float]:
    """Train one model. ``trial`` (Optuna) enables pruning on the selection metric.

    Returns the validation metrics of the best epoch (plus "best_epoch").
    """
    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    if device.type == "cuda":
        import astra
        astra.set_gpu_index(device.index or 0)  # keep the ASTRA projector on this process' GPU

    seed_everything(args.seed)

    geometry_config = DBTGeometryConfig()
    geom = DBTGeometry.from_config(geometry_config)
    missing_views = ~geom.acquired_view_mask

    model_config = model_config_from_args(args)
    run_name = args.run_name or default_run_name(args.model, model_config)
    checkpoint_dir = args.checkpoint_dir or PROJECT_ROOT / "outputs/2d/checkpoints" / run_name
    args.figures_dir = args.figures_dir or PROJECT_ROOT / "outputs/2d/visualisation" / run_name

    # --- Data: fixed, seeded splits shared by every model ---
    train_dataset = SinogramCompletionDataset(args.n_samples, split="train", noise_level=args.noise_level, geometry_config=geometry_config)
    val_dataset = SinogramCompletionDataset(args.n_val, split="val", noise_level=args.noise_level, geometry_config=geometry_config)

    train_sampler = DistributedSampler(train_dataset, seed=args.seed) if is_distributed else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_sampler is None, sampler=train_sampler, drop_last=True)
    # Validation is sharded WITHOUT padding so that the reduced metrics are exact.
    val_shard = Subset(val_dataset, range(rank, len(val_dataset), world_size))
    val_loader = DataLoader(val_shard, batch_size=args.batch_size, shuffle=False)

    # --- Model, loss, optimiser ---
    model, model_config = build_model(args.model, geometry_config, **model_config)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ddp_model = DDP(model, device_ids=[device.index]) if is_distributed else model

    physics = uses_physics(args)
    loss_fn = AnnealedLoss(geom, args.lambda_m0, args.lambda_m1, args.anneal_epochs).to(device)
    calibration = calibrate_physics(loss_fn, val_dataset, device) if physics else {}
    mse_fn = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    run = None
    if is_main:
        print(f"[{run_name}] {num_params:,} parameters | physics={physics} {calibration} | world_size={world_size} | device={device}")
        if hasattr(model, "angular_reach_deg"):
            print(f"[{run_name}] angular reach = {model.angular_reach_deg:.1f} deg "
                  f"(farthest missing view: {np.rad2deg(np.abs(geom.angles[missing_views]).max()) - geometry_config.angle_max_deg:.1f} deg from the acquired window)")
        if save:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_random_dataset_preview(train_dataset, geometry_config, args)
        if args.use_wandb:
            import wandb
            run = wandb.init(project=args.wandb_project, name=run_name, mode=args.wandb_mode,
                             config={**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                     "num_params": num_params, **calibration}, reinit=True)

    # Reference scores of the non-learned baselines on the very same validation samples.
    baselines = {
        "ZeroFilling": evaluate(zero_filling, val_loader, missing_views, device, is_distributed),
        "LinearInterp": evaluate(linear_interpolation, val_loader, missing_views, device, is_distributed),
    }
    if is_main:
        for name, m in baselines.items():
            print(f"[baseline] {name:13s} mse_wedge={m['mse_wedge']:.5f} ssim_wedge={m['ssim_wedge']:.4f} ssim={m['ssim']:.4f}")

    best, history, epochs_without_improvement = None, [], 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ---- Train ----
        ddp_model.train()
        sums = torch.zeros(5, dtype=torch.float64, device=device)  # total, mse, m0, m1, batches
        iterator = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False, disable=not is_main)
        for step, (incomplete, target, _) in enumerate(iterator):
            if args.max_train_batches and step >= args.max_train_batches:
                break
            incomplete, target = incomplete.to(device), target.to(device)
            pred = ddp_model(incomplete)
            if physics:
                loss, l_mse, l_m0, l_m1 = loss_fn(pred[:, 0], target[:, 0], epoch - 1)
            else:
                loss = l_mse = mse_fn(pred, target)
                l_m0 = l_m1 = torch.zeros((), device=device)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            sums += torch.tensor([loss.item(), l_mse.item(), float(l_m0), float(l_m1), 1.0], dtype=torch.float64, device=device)

        # The scheduler must step on EVERY rank, otherwise the replicas train with different learning rates.
        scheduler.step()
        sums = _reduce_sum(sums, is_distributed)
        n_batches = max(1.0, sums[4].item())
        train_log = {"train/loss": sums[0].item() / n_batches, "train/mse": sums[1].item() / n_batches,
                     "train/hlcc_m0": sums[2].item() / n_batches, "train/hlcc_m1": sums[3].item() / n_batches}

        # ---- Validate ----
        ddp_model.eval()
        val = evaluate(model, val_loader, missing_views, device, is_distributed)
        history.append({"epoch": epoch, **train_log, **{f"val/{k}": v for k, v in val.items()}})

        improved = best is None or val[SELECTION_METRIC] < best[SELECTION_METRIC]
        if improved:
            best = {**val, "best_epoch": epoch}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if is_main:
            print(f"Epoch {epoch:03d} | train mse {train_log['train/mse']:.5f} | val mse_wedge {val['mse_wedge']:.5f} "
                  f"ssim_wedge {val['ssim_wedge']:.4f} ssim {val['ssim']:.4f} psnr {val['psnr']:.2f} "
                  f"| lr {optimizer.param_groups[0]['lr']:.2e}{'  *' if improved else ''}")
            if run is not None:
                run.log({**history[-1], "lr": optimizer.param_groups[0]["lr"], "physics_alpha": loss_fn.alpha(epoch - 1) if physics else 0.0})
            if improved and save:
                save_checkpoint(
                    checkpoint_dir / "best_model.pt", model, args.model, model_config, geometry_config,
                    meta={"epoch": epoch, "val_metrics": val, "baselines": baselines, "physics": physics,
                          "calibration": calibration, "num_params": num_params,
                          "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
                )

        if trial is not None:
            import optuna
            trial.report(val[SELECTION_METRIC], epoch)
            if trial.should_prune():
                if run is not None:
                    run.finish()
                raise optuna.exceptions.TrialPruned()

        if args.patience and epochs_without_improvement >= args.patience:
            if is_main:
                print(f"No improvement of {SELECTION_METRIC} for {args.patience} epochs: stopping at epoch {epoch}.")
            break

    if best is None:
        raise RuntimeError("No epoch was run: --n_epochs must be >= 1.")

    if is_main and save:
        stats = {"training_time_seconds": time.time() - start_time, "num_params": num_params, "best": best,
                 "baselines": baselines, "epochs_run": len(history)}
        (checkpoint_dir / "training_stats.json").write_text(json.dumps(stats, indent=2))
        (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2))
        save_training_curve([h["train/loss"] for h in history], args)
        generate_example_figure(model, val_dataset, device, geometry_config, args, run)
        print(f"Done. Best epoch {best['best_epoch']}: " + ", ".join(f"{k}={best[k]:.5f}" for k in METRIC_KEYS))

    if run is not None:
        run.finish()
    if is_distributed:
        dist.destroy_process_group()
    return best


def main(argv: Optional[list] = None) -> None:
    run_training(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
