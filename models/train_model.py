from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dbt_sinogram_completion.conf.geometry import DBTGeometryConfig
from dbt_sinogram_completion.data.dataset import SinogramCompletionDataset
from dbt_sinogram_completion.models.evaluate import (
    build_full_astra_geometries,
    plot_qualitative_example,
    reconstruct_volume_sirt,
)
from dbt_sinogram_completion.models.Unet import SinogramUNet
from dbt_sinogram_completion.models.interpolator import SinusoidalViewInterpolator
from dbt_sinogram_completion.models.pipeline import SinogramCompletionPipeline


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geometry = DBTGeometryConfig()

    # No custom projector_fn here: the dataset builds a real ASTRA cone-beam projector by
    # default. Overriding it with a cheap placeholder would make the incomplete/full
    # sinograms near-identical (trivial task) and defeat the point of comparing models.
    dataset = SinogramCompletionDataset(
        n_samples=args.n_samples,
        device=str(device),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    interpolator = SinusoidalViewInterpolator(angles_rad=geometry.full_angles).to(device)
    refiner = SinogramUNet(in_channels=2, out_channels=1).to(device)
    model = SinogramCompletionPipeline(interpolator, refiner).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    run = None
    if args.use_wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, config=vars(args))

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    epoch_baseline_loss = float("nan")
    epochs_without_improvement = 0
    start_time = time.time()

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

    generate_example_figure(model, dataset, device, geometry, args, run)

    if run is not None:
        run.finish()


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


if __name__ == "__main__":
    main()


