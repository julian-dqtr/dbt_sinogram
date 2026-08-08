from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_3D.conf.geometry_conf_3d import DBTGeometryConfig
from src_3D.data.dataset_3d import SinogramCompletionDataset
from src_3D.models.evaluate_3d import (
    build_full_astra_geometries,
    plot_qualitative_example,
    reconstruct_volume_sirt,
)
from src_3D.models.unet_3d import SinogramUNet
from src_3D.models.pipeline_3d import UNet25DWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 3D sinogram completion model.")
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "outputs/3d/checkpoints")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "outputs/3d/visualisation")
    parser.add_argument("--reconstruct-iters", type=int, default=15)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="dbt-sinogram-completion-3d")
    parser.add_argument("--wandb-mode", type=str, choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = resolve_compute_device(local_rank)
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        device = resolve_compute_device()

    geometry = DBTGeometryConfig()

    train_dataset = SinogramCompletionDataset(n_samples=args.n_samples, phantom_type="ellipses", device=str(device))
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler)

    val_dataset = SinogramCompletionDataset(n_samples=max(1, args.n_samples // 5), phantom_type="shepp_logan", device=str(device))
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler)

    test_dataset = SinogramCompletionDataset(n_samples=max(1, args.n_samples // 5), phantom_type="mixed", device=str(device))
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if is_distributed else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, sampler=test_sampler)

    if local_rank == 0:
        save_random_dataset_preview(train_dataset, geometry, args)

    unet = SinogramUNet(in_channels=1, out_channels=1).to(device)
    model = UNet25DWrapper(unet).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    run = None
    if local_rank == 0 and args.use_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, config=vars(args), mode=args.wandb_mode)

    if local_rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float("inf")
    epochs_without_improvement = 0
    start_time = time.time()
    train_losses = []

    model.train()
    
    epoch_iter = range(args.max_epochs)
    if local_rank == 0:
        epoch_iter = tqdm(epoch_iter, desc="training", unit="epoch")
        
    try:
        for epoch in epoch_iter:
            if is_distributed:
                train_sampler.set_epoch(epoch)
                
            running_loss = 0.0
            
            batch_iter = train_loader
            if local_rank == 0:
                batch_iter = tqdm(batch_iter, desc=f"epoch {epoch + 1}", leave=False, unit="batch")
                
            for incomplete, target, _ in batch_iter:
                incomplete = incomplete.to(device)
                target = target.to(device)

                # Randomly sample 16 slices along Y (dim=3) to prevent OOM
                Y = incomplete.shape[3]
                num_slices = 16
                if Y > num_slices:
                    slice_indices = torch.randperm(Y, device=device)[:num_slices]
                    incomplete_train = incomplete[:, :, :, slice_indices, :]
                    target_train = target[:, :, :, slice_indices, :]
                else:
                    incomplete_train = incomplete
                    target_train = target

                refined_output = model(incomplete_train)
                loss = criterion(refined_output, target_train)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if local_rank == 0:
                    batch_iter.set_postfix(unet=f"{loss.item():.4f}")

            train_loss = running_loss / max(1, len(train_loader))
            
            if is_distributed:
                loss_tensor = torch.tensor([train_loss], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                train_loss = (loss_tensor[0] / dist.get_world_size()).item()
            
            # Validation loop
            model.eval()
            val_loss = 0.0
            val_psnr = 0.0
            val_ssim = 0.0
            with torch.no_grad():
                for incomplete, target, _ in val_loader:
                    incomplete = incomplete.to(device)
                    target = target.to(device)
                    refined_output = model(incomplete)
                    val_loss += criterion(refined_output, target).item()
                    
                    # Compute SSIM/PSNR on CPU. For 3D we compute slice-by-slice along Y
                    target_np = target.cpu().numpy()
                    output_np = refined_output.cpu().numpy()
                    
                    batch_psnr = 0.0
                    batch_ssim = 0.0
                    total_slices = 0
                    for b in range(target_np.shape[0]):
                        for y in range(target_np.shape[3]):
                            t = target_np[b, 0, :, y, :]
                            o = output_np[b, 0, :, y, :]
                            batch_psnr += psnr(t, o, data_range=2.0)
                            batch_ssim += ssim(t, o, data_range=2.0)
                            total_slices += 1
                            
                    val_psnr += batch_psnr / total_slices
                    val_ssim += batch_ssim / total_slices
                    
            val_loss /= max(1, len(val_loader))
            val_psnr /= max(1, len(val_loader))
            val_ssim /= max(1, len(val_loader))
            
            if is_distributed:
                val_tensor = torch.tensor([val_loss, val_psnr, val_ssim], device=device)
                dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
                val_loss = (val_tensor[0] / dist.get_world_size()).item()
                val_psnr = (val_tensor[1] / dist.get_world_size()).item()
                val_ssim = (val_tensor[2] / dist.get_world_size()).item()
                
            model.train()

            if local_rank == 0:
                train_losses.append(train_loss)
                epoch_iter.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", ssim=f"{val_ssim:.4f}")

                if run is not None:
                    import wandb
                    wandb.log({
                        "epoch": epoch + 1,
                        "loss/train_unet": train_loss,
                        "loss/val_unet": val_loss,
                        "metrics/val_psnr": val_psnr,
                        "metrics/val_ssim": val_ssim,
                        "loss/best_val": min(best_loss, val_loss),
                    })

                if val_loss < best_loss - args.min_delta:
                    best_loss = val_loss
                    epochs_without_improvement = 0
                    model_to_save = model.module if is_distributed else model
                    torch.save(model_to_save.state_dict(), args.checkpoint_dir / "best_model.pt")
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= args.patience:
                        print(f"\nNo improvement for {args.patience} epochs, stopping early at epoch {epoch + 1}.")
            
            if is_distributed:
                break_flag = torch.tensor(1 if epochs_without_improvement >= args.patience else 0, device=device)
                dist.broadcast(break_flag, src=0)
                if break_flag.item() == 1:
                    break
            elif epochs_without_improvement >= args.patience:
                break

    except KeyboardInterrupt:
        if local_rank == 0:
            print("\nTraining interrupted by user, saving current model state.")
            model_to_save = model.module if is_distributed else model
            torch.save(model_to_save.state_dict(), args.checkpoint_dir / "interrupted_model.pt")

    if local_rank == 0:
        elapsed = time.time() - start_time
        print(f"Training finished in {elapsed / 60:.1f} min. Best loss (U-Net): {best_loss:.4f}")
        
        # Reload best model for test
        model_to_eval = model.module if is_distributed else model
        if (args.checkpoint_dir / "best_model.pt").exists():
            model_to_eval.load_state_dict(torch.load(args.checkpoint_dir / "best_model.pt", map_location=device, weights_only=True))
            
        save_training_curve(train_losses, args)
        generate_example_figure(model_to_eval, test_dataset, device, geometry, args, run)
        
        if run is not None:
            run.finish()

    if is_distributed:
        dist.destroy_process_group()


def resolve_compute_device(local_rank: int = -1) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    device = torch.device(f"cuda:{local_rank}") if local_rank != -1 else torch.device("cuda")
    
    probe = torch.randn(1, 1, 8, 8, 8, device=device)
    conv = torch.nn.Conv3d(1, 1, kernel_size=3).to(device)

    try:
        _ = conv(probe)
        return device
    except RuntimeError as exc:
        if "CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH" in str(exc):
            torch.backends.cudnn.enabled = False
            try:
                _ = conv(probe)
                return device
            except RuntimeError:
                torch.backends.cudnn.enabled = True
                return torch.device("cpu")
        return torch.device("cpu")


def generate_example_figure(model, dataset, device, geometry, args, run) -> None:
    model.eval()
    idx = random.randrange(len(dataset))
    incomplete, full, phantom = dataset[idx]
    with torch.no_grad():
        refined_out = model(incomplete.unsqueeze(0).to(device))

    Z = phantom.shape[1]
    slice_idx = Z // 2
    phantom_slice = phantom.squeeze(0)[slice_idx].cpu()
    
    Det_Y = full.shape[2]
    sino_idx = Det_Y // 2
    full_slice = full.squeeze(0)[:, sino_idx, :].cpu()
    incomplete_slice = incomplete.squeeze(0)[:, sino_idx, :].cpu()
    refined_out_slice = refined_out.squeeze(0).squeeze(0)[:, sino_idx, :].cpu()

    reconstructed_volume_slice = None
    try:
        proj_geom, vol_geom = build_full_astra_geometries(geometry, tuple(phantom.shape[1:]))
        reconstructed_vol = reconstruct_volume_sirt(
            refined_out.squeeze(0).squeeze(0).cpu().numpy(), proj_geom, vol_geom, n_iterations=args.reconstruct_iters
        )
        if reconstructed_vol is not None:
            reconstructed_volume_slice = reconstructed_vol[slice_idx, :, :]
    except Exception as exc:
        print(f"Skipping volume reconstruction panel ({exc}).")

    save_path = args.figures_dir / "example_after_training.png"
    dummy_baseline = torch.zeros_like(incomplete_slice)
    fig = plot_qualitative_example(
        phantom_slice, full_slice, incomplete_slice, dummy_baseline, refined_out_slice, reconstructed_volume_slice, geometry, save_path
    )
    print(f"Saved qualitative example to {save_path}")

    if run is not None:
        import wandb
        wandb.log({"example": wandb.Image(fig)})


def save_random_dataset_preview(dataset, geometry, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = random.randrange(len(dataset))
    incomplete, full, phantom = dataset[idx]
    
    Z = phantom.shape[1]
    slice_idx = Z // 2
    phantom_slice = phantom.squeeze(0)[slice_idx].cpu()
    
    Det_Y = full.shape[2]
    sino_idx = Det_Y // 2
    full_slice = full.squeeze(0)[:, sino_idx, :].cpu()
    incomplete_slice = incomplete.squeeze(0)[:, sino_idx, :].cpu()

    row_min, row_max = geometry.image_extent_mm[1]
    col_min, col_max = geometry.image_extent_mm[2]
    image_extent = [col_min, col_max, row_min, row_max]

    detector_half_width = geometry.det_col_count * geometry.det_pixel_size_mm / 2.0
    sino_extent = [geometry.full_angle_min_deg, geometry.full_angle_max_deg, -detector_half_width, detector_half_width]

    def plot_sinogram(ax, sinogram: torch.Tensor, title: str) -> None:
        image = sinogram.detach().cpu().numpy().T
        ax.imshow(image, cmap="bone", origin="lower", aspect="auto", extent=sino_extent)
        ax.set_title(title)
        ax.set_xlabel(r"Angle $\phi$ (degrees)")
        ax.set_ylabel("Detector width u (mm)")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(phantom_slice.detach().cpu().numpy(), cmap="gray", origin="lower", extent=image_extent, vmin=-1.0, vmax=1.0)
    axes[0].set_title(f"Random sample #{idx} - Middle Slice")
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Y (mm)")

    plot_sinogram(axes[1], full_slice, "Full sinogram (Middle Det Row)")
    plot_sinogram(axes[2], incomplete_slice, "Incomplete sinogram")

    plt.tight_layout()
    save_path = args.figures_dir / "dataset_preview.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved random dataset preview to {save_path}")


def save_training_curve(train_losses: list[float], args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not train_losses:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="U-Net loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_path = args.figures_dir / "training_losses.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
