from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conf.geometry import DBTGeometryConfig
from data.dataset import SinogramCompletionDataset
from models.evaluate import (
    build_full_astra_geometries,
    plot_qualitative_example,
    reconstruct_volume_sirt,
)
from models.Unet import SinogramUNet
from models.interpolator import SinusoidalViewInterpolator
from models.pipeline import SinogramCompletionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the sinogram completion model.")
    parser.add_argument("--max-epochs", type=int, default=500, help="Upper bound on epochs (safety net).")
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Stop early after this many epochs without loss improvement.",
    )
    parser.add_argument("--min-delta", type=float, default=1e-5, help="Minimum loss improvement to reset patience.")
    parser.add_argument("--n-samples", type=int, default=100, help="Samples generated per epoch.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models"))
    parser.add_argument("--figures-dir", type=Path, default=Path("models/visualisation"))
    parser.add_argument("--reconstruct-iters", type=int, default=30, help="SIRT iterations for the example figure.")
    parser.add_argument("--use-wandb", action="store_true", help="Log metrics/figures to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="dbt-sinogram-completion")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=("online", "offline", "disabled"),
        default="online",
        help="Weights & Biases mode. Use offline on a cluster without network access.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_compute_device()
    geometry = DBTGeometryConfig()

    # No custom projector_fn here: the dataset builds a real ASTRA cone-beam projector by
    # default. Overriding it with a cheap placeholder would make the incomplete/full
    # sinograms near-identical (trivial task) and defeat the point of comparing models.
    dataset = SinogramCompletionDataset(
        n_samples=args.n_samples,
        device=str(device),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    save_random_dataset_preview(dataset, geometry, args)

    interpolator = SinusoidalViewInterpolator(angles_rad=geometry.full_angles).to(device)
    refiner = SinogramUNet(in_channels=2, out_channels=1).to(device)
    model = SinogramCompletionPipeline(interpolator, refiner).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    run = None
    if args.use_wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, config=vars(args), mode=args.wandb_mode)
        print(f"Weights & Biases run started: {run.url}")
    else:
        print("Weights & Biases is disabled. Re-run with --use-wandb to track losses remotely.")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    epoch_baseline_loss = float("nan")
    epochs_without_improvement = 0
    start_time = time.time()
    train_losses: list[float] = []
    baseline_losses: list[float] = []

    model.train()
    epoch_bar = tqdm(range(args.max_epochs), desc="training", unit="epoch")
    try:
        for epoch in epoch_bar:
            running_loss = 0.0
            running_baseline_loss = 0.0
            batch_bar = tqdm(loader, desc=f"epoch {epoch + 1}", leave=False, unit="batch")
            for incomplete, target, _ in batch_bar:
                incomplete = incomplete.to(device)
                target = target.to(device)

                baseline_output, refined_output = model(incomplete)
                loss = criterion(refined_output, target)
                with torch.no_grad():
                    baseline_loss = criterion(baseline_output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_baseline_loss += baseline_loss.item()
                batch_bar.set_postfix(unet=f"{loss.item():.4f}", interp=f"{baseline_loss.item():.4f}")

            epoch_loss = running_loss / max(1, len(loader))
            epoch_baseline_loss = running_baseline_loss / max(1, len(loader))
            train_losses.append(epoch_loss)
            baseline_losses.append(epoch_baseline_loss)
            epoch_bar.set_postfix(unet=f"{epoch_loss:.4f}", interp=f"{epoch_baseline_loss:.4f}", best=f"{best_loss:.4f}")

            if run is not None:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "loss/unet": epoch_loss,
                        "loss/interpolation_baseline": epoch_baseline_loss,
                        "loss/best_unet": min(best_loss, epoch_loss),
                    }
                )

            if epoch_loss < best_loss - args.min_delta:
                best_loss = epoch_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), args.checkpoint_dir / "best_model.pt")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"\nNo improvement for {args.patience} epochs, stopping early at epoch {epoch + 1}.")
                    break
    except KeyboardInterrupt:
        print("\nTraining interrupted by user, saving current model state.")
        torch.save(model.state_dict(), args.checkpoint_dir / "interrupted_model.pt")

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed / 60:.1f} min. Best loss (U-Net): {best_loss:.4f}")
    print(f"Interpolation baseline loss (last epoch): {epoch_baseline_loss:.4f}")

    save_training_curve(train_losses, baseline_losses, args)
    generate_example_figure(model, dataset, device, geometry, args, run)

    if run is not None:
        run.finish()


def resolve_compute_device() -> torch.device:
    """Prefer CUDA, but fall back if the local PyTorch/cuDNN build cannot run there."""
    if not torch.cuda.is_available():
        print("CUDA is not available; using CPU.")
        return torch.device("cpu")

    cuda_device = torch.device("cuda")
    probe = torch.randn(1, 1, 8, 8, device=cuda_device)
    conv = torch.nn.Conv2d(1, 1, kernel_size=3).to(cuda_device)

    try:
        _ = conv(probe)
        return cuda_device
    except RuntimeError as exc:
        if "CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH" in str(exc):
            print("CUDA is available but cuDNN does not support this GPU architecture; disabling cuDNN.")
            torch.backends.cudnn.enabled = False
            try:
                _ = conv(probe)
                print("Using CUDA with cuDNN disabled.")
                return cuda_device
            except RuntimeError as fallback_exc:
                print(f"CUDA still failed after disabling cuDNN ({fallback_exc}); using CPU.")
                torch.backends.cudnn.enabled = True
                return torch.device("cpu")

        print(f"CUDA probe failed ({exc}); using CPU.")
        return torch.device("cpu")


def generate_example_figure(model, dataset, device, geometry, args, run) -> None:
    """Pick a random sample and save a GT/full-sino/limited-sino/U-Net/reconstruction figure."""
    model.eval()
    idx = random.randrange(len(dataset))
    incomplete, full, phantom = dataset[idx]
    with torch.no_grad():
        baseline_out, refined_out = model(incomplete.unsqueeze(0).to(device))

    phantom = phantom.squeeze(0).cpu()
    full = full.squeeze(0).cpu()
    incomplete = incomplete.squeeze(0).cpu()
    baseline_out = baseline_out.squeeze(0).squeeze(0).cpu()
    refined_out = refined_out.squeeze(0).squeeze(0).cpu()

    reconstructed_volume = None
    try:
        proj_geom, vol_geom = build_full_astra_geometries(geometry, tuple(phantom.shape))
        reconstructed_volume = reconstruct_volume_sirt(
            refined_out.numpy(), proj_geom, vol_geom, n_iterations=args.reconstruct_iters
        )
    except Exception as exc:  # pragma: no cover - environment dependent (ASTRA/CUDA)
        print(f"Skipping volume reconstruction panel ({exc}).")

    save_path = args.figures_dir / "example_after_training.png"
    fig = plot_qualitative_example(
        phantom, full, incomplete, baseline_out, refined_out, reconstructed_volume, geometry, save_path
    )
    print(f"Saved qualitative example to {save_path}")

    if run is not None:
        import wandb

        wandb.log({"example": wandb.Image(fig)})


def save_random_dataset_preview(dataset, geometry, args) -> None:
    """Save a PNG preview of a random dataset sample for cluster workflows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = random.randrange(len(dataset))
    incomplete, full, phantom = dataset[idx]
    phantom = phantom.squeeze(0).cpu()
    full = full.squeeze(0).cpu()
    incomplete = incomplete.squeeze(0).cpu()

    row_min, row_max = geometry.image_extent_mm[0]
    col_min, col_max = geometry.image_extent_mm[1]
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
    axes[0].imshow(phantom.detach().cpu().numpy(), cmap="gray", origin="lower", extent=image_extent)
    axes[0].set_title(f"Random dataset sample #{idx} - Phantom")
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Z (mm)")

    plot_sinogram(axes[1], full, "Full sinogram")
    plot_sinogram(axes[2], incomplete, "Incomplete sinogram")

    plt.tight_layout()
    save_path = args.figures_dir / "dataset_preview.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved random dataset preview to {save_path}")


def save_training_curve(train_losses: list[float], baseline_losses: list[float], args) -> None:
    """Save a PNG with the training and interpolation losses."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not train_losses:
        print("Skipping training curve plot because no epochs were completed.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="U-Net loss", linewidth=2)
    ax.plot(epochs, baseline_losses, label="Interpolation baseline loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training losses")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_path = args.figures_dir / "training_losses.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved training loss curve to {save_path}")


if __name__ == "__main__":
    main()


