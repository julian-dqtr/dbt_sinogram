import argparse
import sys
import time
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_2D.conf.geometry import DBTGeometryConfig
from src_2D.data.dataset import SinogramCompletionDataset
from src_2D.models.Unet import SinogramUNet
from src_2D.models.evaluate import (
    build_full_astra_geometries,
    plot_qualitative_example,
    reconstruct_volume_sirt,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 2D sinogram completion model.")
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.007875808314275142)
    parser.add_argument("--filters", type=int, default=16)
    parser.add_argument("--optimizer", type=str, choices=("Adam", "AdamW"), default="Adam")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models_2D"))
    parser.add_argument("--figures-dir", type=Path, default=Path("models_2D/visualisation"))
    parser.add_argument("--reconstruct-iters", type=int, default=15)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="dbt-sinogram-completion-2D")
    parser.add_argument("--wandb-mode", type=str, choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_compute_device()
    geometry = DBTGeometryConfig()

    train_dataset = SinogramCompletionDataset(n_samples=args.n_samples, phantom_type="ellipses", device=str(device))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = SinogramCompletionDataset(n_samples=max(1, args.n_samples // 5), phantom_type="shepp_logan", device=str(device))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = SinogramCompletionDataset(n_samples=max(1, args.n_samples // 5), phantom_type="mixed", device=str(device))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    save_random_dataset_preview(train_dataset, geometry, args)

    model = SinogramUNet(in_channels=1, out_channels=1, filters=args.filters).to(device)

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    run = None
    if args.use_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, config=vars(args), mode=args.wandb_mode)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float("inf")
    epochs_without_improvement = 0
    start_time = time.time()
    train_losses = []

    model.train()
    
    epoch_iter = tqdm(range(args.max_epochs), desc="training", unit="epoch")
    try:
        for epoch in epoch_iter:
            running_loss = 0.0
            
            batch_iter = tqdm(train_loader, desc=f"epoch {epoch + 1}", leave=False, unit="batch")
            for incomplete, target, _ in batch_iter:
                incomplete = incomplete.to(device)
                target = target.to(device)

                refined_output = model(incomplete)
                loss = criterion(refined_output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batch_iter.set_postfix(unet=f"{loss.item():.4f}")

            train_loss = running_loss / max(1, len(train_loader))
            
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
                    
                    # Calculate metrics on CPU
                    target_np = target.cpu().numpy()
                    output_np = refined_output.cpu().numpy()
                    
                    batch_psnr = 0.0
                    batch_ssim = 0.0
                    for i in range(target_np.shape[0]):
                        t = target_np[i, 0]
                        o = output_np[i, 0]
                        batch_psnr += psnr(t, o, data_range=2.0)
                        batch_ssim += ssim(t, o, data_range=2.0)
                    
                    val_psnr += batch_psnr / target_np.shape[0]
                    val_ssim += batch_ssim / target_np.shape[0]

            val_loss /= max(1, len(val_loader))
            val_psnr /= max(1, len(val_loader))
            val_ssim /= max(1, len(val_loader))
            model.train()

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
    
    # Reload best model for final evaluation
    if (args.checkpoint_dir / "best_model.pt").exists():
        model.load_state_dict(torch.load(args.checkpoint_dir / "best_model.pt", map_location=device, weights_only=True))

    print("\nEvaluating best model on test set...")
    model.eval()
    test_loss = 0.0
    test_psnr = 0.0
    test_ssim = 0.0
    with torch.no_grad():
        for incomplete, target, _ in test_loader:
            incomplete = incomplete.to(device)
            target = target.to(device)
            refined_output = model(incomplete)

            test_loss += criterion(refined_output, target).item()

            target_np = target.cpu().numpy()
            output_np = refined_output.cpu().numpy()

            batch_psnr = 0.0
            batch_ssim = 0.0
            for i in range(target_np.shape[0]):
                t = target_np[i, 0]
                o = output_np[i, 0]
                batch_psnr += psnr(t, o, data_range=2.0)
                batch_ssim += ssim(t, o, data_range=2.0)

            test_psnr += batch_psnr / target_np.shape[0]
            test_ssim += batch_ssim / target_np.shape[0]

    test_loss /= max(1, len(test_loader))
    test_psnr /= max(1, len(test_loader))
    test_ssim /= max(1, len(test_loader))
    print(f"Test Loss: {test_loss:.4f} | Test PSNR: {test_psnr:.4f} | Test SSIM: {test_ssim:.4f}")

    if run is not None:
        import wandb
        wandb.log({
            "metrics/test_loss": test_loss,
            "metrics/test_psnr": test_psnr,
            "metrics/test_ssim": test_ssim,
        })

    save_training_curve(train_losses, args)
    generate_example_figure(model, test_dataset, device, geometry, args, run)
        
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
    model.eval()
    idx = random.randrange(len(dataset))
    incomplete, full, phantom = dataset[idx]
    with torch.no_grad():
        refined_out = model(incomplete.unsqueeze(0).to(device))

    reconstructed_image = None
    try:
        proj_geom, vol_geom = build_full_astra_geometries(geometry, tuple(phantom.shape[1:]))
        reconstructed_image = reconstruct_volume_sirt(
            refined_out.squeeze(0).squeeze(0).cpu().numpy(), proj_geom, vol_geom, n_iterations=args.reconstruct_iters
        )
    except Exception as exc:
        print(f"Skipping volume reconstruction panel ({exc}).")

    save_path = args.figures_dir / "example_after_training.png"
    # Create a dummy baseline to not break the plotting function
    dummy_baseline = torch.zeros_like(incomplete.squeeze(0))
    fig = plot_qualitative_example(
        phantom.squeeze(0), full.squeeze(0), incomplete.squeeze(0), dummy_baseline, refined_out.squeeze(0).squeeze(0).cpu(), reconstructed_image, geometry, save_path
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

    row_min, row_max = geometry.image_extent_mm[0]
    col_min, col_max = geometry.image_extent_mm[1]
    image_extent = [col_min, col_max, row_min, row_max]

    detector_half_width = geometry.det_col_count * geometry.det_pixel_size_mm / 2.0
    sino_extent = [geometry.full_angle_min_deg, geometry.full_angle_max_deg, -detector_half_width, detector_half_width]

    def plot_sinogram(ax, sinogram: torch.Tensor, title: str) -> None:
        image = sinogram.detach().cpu().squeeze(0).numpy().T
        ax.imshow(image, cmap="bone", origin="lower", aspect="auto", extent=sino_extent)
        ax.set_title(title)
        ax.set_xlabel(r"Angle $\phi$ (degrees)")
        ax.set_ylabel("Detector width u (mm)")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(phantom.detach().cpu().squeeze(0).numpy(), cmap="gray", origin="lower", extent=image_extent, vmin=-1.0, vmax=1.0)
    axes[0].set_title(f"Random sample #{idx} (Ground truth phantom)")
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Z (mm)")

    plot_sinogram(axes[1], full, "Full sinogram (180 views)")
    plot_sinogram(axes[2], incomplete, "Incomplete sinogram (Limited views)")

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
