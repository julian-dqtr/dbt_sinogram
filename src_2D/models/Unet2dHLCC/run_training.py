import os
import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm import tqdm

torch.backends.cudnn.enabled = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.models.Unet2dHLCC.unet_hlcc import Unet2dHLCC
from src_2D.models.SinoSheavesNN.physics_loss import AnnealedLoss
from src_2D.utils.evaluation import (
    generate_example_figure,
    save_training_curve,
    save_random_dataset_preview,
    get_soft_acquired_mask,
)


# ---------------------------------------------------------------------------
# Training / Validation helpers
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, loss_fn, acquired_mask, epoch, device, local_rank, is_distributed):
    model.train()
    total_loss = total_data = total_m0 = total_m1 = 0.0

    batch_iter = dataloader
    if local_rank == 0:
        batch_iter = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

    for incomplete_sino, full_sino, _ in batch_iter:
        inc_sino = incomplete_sino.to(device)
        target_sino = full_sino[0, 0].to(device)

        optimizer.zero_grad()
        out = model(inc_sino, acquired_mask)
        pred_sino = out[0, 0]

        loss, l_data, l_m0, l_m1 = loss_fn(pred_sino, target_sino, epoch)
        loss.backward()

        # Gradient clipping prevents exploding gradients when physics terms kick in
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += loss.item()
        total_data  += l_data.item()
        total_m0    += l_m0.item()
        total_m1    += l_m1.item()

    n = max(1, len(dataloader))
    avg = dict(loss=total_loss / n, data=total_data / n, m0=total_m0 / n, m1=total_m1 / n)

    if is_distributed:
        for key in avg:
            t = torch.tensor([avg[key]], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg[key] = (t / dist.get_world_size()).item()

    return avg


def val_epoch(model, dataloader, loss_fn, acquired_mask, epoch, device, local_rank, is_distributed):
    model.eval()
    total_loss = total_data = total_m0 = total_m1 = 0.0

    batch_iter = dataloader
    if local_rank == 0:
        batch_iter = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)

    with torch.no_grad():
        for incomplete_sino, full_sino, _ in batch_iter:
            inc_sino = incomplete_sino.to(device)
            target_sino = full_sino[0, 0].to(device)

            out = model(inc_sino, acquired_mask)
            pred_sino = out[0, 0]
            loss, l_data, l_m0, l_m1 = loss_fn(pred_sino, target_sino, epoch)

            total_loss  += loss.item()
            total_data  += l_data.item()
            total_m0    += l_m0.item()
            total_m1    += l_m1.item()

    n = max(1, len(dataloader))
    avg = dict(loss=total_loss / n, data=total_data / n, m0=total_m0 / n, m1=total_m1 / n)

    if is_distributed:
        for key in avg:
            t = torch.tensor([avg[key]], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg[key] = (t / dist.get_world_size()).item()

    return avg


# ---------------------------------------------------------------------------
# Loss calibration: run a few batches before training to set normalisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_loss(loss_fn, dataloader, model, acquired_mask, device, n_batches: int = 5, is_distributed: bool = False, local_rank: int = 0):
    """
    Feed a few sinograms through the model to measure the raw physics loss
    magnitudes, so they can be normalised to the same scale as the MSE.
    """
    model.eval()
    ref_sinos = []
    for i, (inc, full, _) in enumerate(dataloader):
        if i >= n_batches:
            break
        ref_sinos.append(full[0, 0].to(device))

    stacked = torch.stack(ref_sinos)  # [n_batches, V, D]
    loss_fn.calibrate(stacked)

    if is_distributed:
        dist.broadcast(loss_fn.physics_loss_fn.scale_m0, src=0)
        dist.broadcast(loss_fn.physics_loss_fn.scale_m1, src=0)

    if local_rank == 0:
        print(
            f"[Calibration] scale_m0={loss_fn.physics_loss_fn.scale_m0.item():.4e}  "
            f"scale_m1={loss_fn.physics_loss_fn.scale_m1.item():.4e}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Unet2dHLCC")
    parser.add_argument("--epochs",         type=int,   default=300)
    parser.add_argument("--batch_size",     type=int,   default=1)
    parser.add_argument("--lr",             type=float, default=0.0001826052610920645)
    parser.add_argument("--n_samples",      type=int,   default=2000)
    parser.add_argument("--filters",        type=int,   default=32)
    # HLCC weights (relative, now that losses are normalised at calibration time)
    parser.add_argument("--lambda_m0",      type=float, default=0.6118940652183877)
    parser.add_argument("--lambda_m1",      type=float, default=0.0014802448790780431)
    # Cosine annealing: physics loss ramps from 0 → full over this many epochs
    # Rule of thumb: ~10-20% of total epochs so network learns basic MSE first
    parser.add_argument("--anneal_epochs",  type=int,   default=50)
    parser.add_argument("--wandb_project",  type=str,   default="dbt-sinogram-completion-2D")
    parser.add_argument("--wandb_name",     type=str,   default="Unet2dHLCC-v2")
    parser.add_argument("--use-wandb",      action="store_true", help="Log metrics to W&B")
    parser.add_argument(
        "--checkpoint_dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "2d" / "checkpoints_unet2dhlcc",
    )
    parser.add_argument(
        "--figures_dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "2d" / "figures_unet2dhlcc",
    )
    args = parser.parse_args()

    # --- Distributed setup ---
    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if local_rank == 0 and args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    # --- Geometry ---
    config = DBTGeometryConfig()
    geom = DBTGeometry(
        angles=config.full_angles,
        src_radius=config.src_radius_mm,
        det_radius=config.det_radius_mm,
        det_col_count=config.det_col_count,
        det_pixel_size=config.det_pixel_size_mm,
    )
    acquired_mask = get_soft_acquired_mask(geom, device)

    # --- Dataset ---
    train_dataset = SinogramCompletionDataset(n_samples=args.n_samples, device=str(device))
    val_dataset   = SinogramCompletionDataset(
        n_samples=max(1, args.n_samples // 5), device=str(device), is_validation_or_test=True
    )

    train_sampler = DistributedSampler(train_dataset)          if is_distributed else None
    val_sampler   = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
    )

    # --- Model ---
    model = Unet2dHLCC(in_channels=1, out_channels=1, filters=args.filters).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # --- Optimiser & scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # CosineAnnealingLR decays LR smoothly to eta_min; better than ReduceLROnPlateau for long runs
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # --- Loss with HLCC ---
    loss_fn = AnnealedLoss(
        geom,
        lambda_m0=args.lambda_m0,
        lambda_m1=args.lambda_m1,
        anneal_epochs=args.anneal_epochs,
    ).to(device)

    # Calibrate physics loss scales BEFORE training begins
    if local_rank == 0:
        print("Calibrating physics loss normalisation...")
    calibrate_loss(
        loss_fn, train_loader, model, acquired_mask, device,
        is_distributed=is_distributed, local_rank=local_rank
    )

    if local_rank == 0:
        print(f"Starting training for {args.epochs} epochs. Distributed: {is_distributed}")
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    train_losses  = []

    for epoch in range(1, args.epochs + 1):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        t = train_epoch(model, train_loader, optimizer, loss_fn, acquired_mask, epoch, device, local_rank, is_distributed)
        v = val_epoch  (model, val_loader,   loss_fn, acquired_mask, epoch, device, local_rank, is_distributed)

        if local_rank == 0:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:03d} | "
                f"T-Loss:{t['loss']:.4f}  T-MSE:{t['data']:.4f}  T-M0:{t['m0']:.4f}  T-M1:{t['m1']:.4f} | "
                f"V-Loss:{v['loss']:.4f}  V-MSE:{v['data']:.4f} | "
                f"LR:{current_lr:.2e}"
            )
            train_losses.append(t["loss"])

            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "lr": current_lr,
                    "train/loss": t["loss"], "train/mse": t["data"],
                    "train/m0": t["m0"],     "train/m1": t["m1"],
                    "val/loss":  v["loss"],  "val/mse": v["data"],
                    "val/m0":    v["m0"],    "val/m1":  v["m1"],
                })

            if v["loss"] < best_val_loss:
                best_val_loss = v["loss"]
                model_to_save = model.module if is_distributed else model
                torch.save(
                    model_to_save.state_dict(),
                    args.checkpoint_dir / "best_unet2dhlcc_model.pt",
                )
                print(f"  --> New best model (Val Loss: {best_val_loss:.4f})")

    if local_rank == 0:
        save_training_curve(train_losses, args)

        model_to_eval = model.module if is_distributed else model
        state = torch.load(
            args.checkpoint_dir / "best_unet2dhlcc_model.pt", map_location=device, weights_only=True
        )
        model_to_eval.load_state_dict(state)

        generate_example_figure(
            model_to_eval, val_dataset, device, config, args,
            wandb.run if wandb.run is not None else None,
            acquired_mask=acquired_mask,
        )
        print("Training completed.")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
