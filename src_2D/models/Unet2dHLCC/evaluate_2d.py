from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.geometry.dbt_geometry_2d import DBTGeometry


def build_full_astra_geometries(geometry_config: DBTGeometryConfig, image_shape: Tuple[int, int]):
    """Build the ASTRA (proj_geom, vol_geom) pair for the full-arc (-90..+90deg) sinogram."""
    import astra

    geometry = DBTGeometry(
        angles=geometry_config.full_angles,
        src_radius=geometry_config.src_radius_mm,
        det_radius=geometry_config.det_radius_mm,
        det_col_count=geometry_config.det_col_count,
        det_pixel_size=geometry_config.det_pixel_size_mm,
    )
    proj_geom = geometry.get_astra_proj_geom()

    rows, cols = image_shape
    row_min, row_max = geometry_config.image_extent_mm[0]
    col_min, col_max = geometry_config.image_extent_mm[1]
    vol_geom = astra.create_vol_geom(rows, cols, col_min, col_max, row_min, row_max)
    return proj_geom, vol_geom


def reconstruct_volume_fbp(
    sinogram_view_col: np.ndarray,
    proj_geom,
    vol_geom,
) -> Optional[np.ndarray]:
    """Reconstruct a 2D image (FBP_CUDA) from a sinogram shaped [views, cols]."""
    try:
        import astra
    except ImportError:
        return None

    sinogram_astra = np.ascontiguousarray(sinogram_view_col.astype(np.float32))

    projector_id = astra.create_projector("cuda", proj_geom, vol_geom)
    sino_id = astra.data2d.create("-sino", proj_geom, sinogram_astra)
    reco_id = astra.data2d.create("-vol", vol_geom)
    try:
        cfg = astra.astra_dict("BP_CUDA")
        cfg["ProjectorId"] = projector_id
        cfg["ProjectionDataId"] = sino_id
        cfg["ReconstructionDataId"] = reco_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        reconstruction = np.asarray(astra.data2d.get(reco_id))
        astra.algorithm.delete(alg_id)
    finally:
        astra.data2d.delete(reco_id)
        astra.data2d.delete(sino_id)
        astra.projector.delete(projector_id)
    return reconstruction


def plot_qualitative_example(
    phantom: torch.Tensor,
    full_sinogram: torch.Tensor,
    incomplete_sinogram: torch.Tensor,
    baseline_sinogram: torch.Tensor,
    refined_sinogram: torch.Tensor,
    reconstructed_image: Optional[np.ndarray],
    geometry_config: DBTGeometryConfig,
    save_path: Path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Compute robust global min/max for consistent scaling across plots
    # This avoids ValueError: minvalue > maxvalue if a prediction goes out of bounds
    sino_vmin = min(full_sinogram.min().item(), incomplete_sinogram.min().item(), baseline_sinogram.min().item(), refined_sinogram.min().item())
    sino_vmax = max(full_sinogram.max().item(), incomplete_sinogram.max().item(), baseline_sinogram.max().item(), refined_sinogram.max().item(), 1.0)
    
    # Ensure vmin is strictly less than vmax
    if sino_vmin >= sino_vmax:
        sino_vmin = sino_vmax - 0.1

    row_min, row_max = geometry_config.image_extent_mm[0]
    col_min, col_max = geometry_config.image_extent_mm[1]
    image_extent = [col_min, col_max, row_min, row_max]

    angle_min, angle_max = geometry_config.full_angle_min_deg, geometry_config.full_angle_max_deg
    detector_half_width = geometry_config.det_col_count * geometry_config.det_pixel_size_mm / 2.0
    sino_extent = [angle_min, angle_max, -detector_half_width, detector_half_width]

    def plot_sinogram(ax, sinogram: torch.Tensor, title: str):
        image = sinogram.detach().cpu().numpy().T
        ax.imshow(image, cmap="bone", origin="lower", aspect="auto", extent=sino_extent, vmin=sino_vmin, vmax=sino_vmax)
        ax.set_xlim(angle_min, angle_max)
        ax.set_title(title)
        ax.set_xlabel(r"Angle $\phi$ (degrees)")
        ax.set_ylabel("Detector width u (mm)")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(phantom.detach().cpu().numpy(), cmap="gray", origin="lower", extent=image_extent)
    axes[0, 0].set_title("1. Original Phantom")
    axes[0, 0].set_xlabel("X (mm)")
    axes[0, 0].set_ylabel("Z (mm)")

    plot_sinogram(axes[0, 1], full_sinogram, "2. Full Sinogram (-90 deg to +90 deg)")

    plot_sinogram(
        axes[0, 2],
        incomplete_sinogram,
        f"3. Limited Sinogram (Stationary DBT, "
        f"{geometry_config.angle_min_deg:.0f} deg to {geometry_config.angle_max_deg:.0f} deg)",
    )

    plot_sinogram(axes[1, 0], baseline_sinogram, "4. Baseline Reconstruction (Linear Fit)")

    plot_sinogram(axes[1, 1], refined_sinogram, "5. Unet2dHLCC Reconstruction (Data Consistent)")

    if reconstructed_image is not None:
        vmax = max(float(phantom.max()), float(reconstructed_image.max()), 1.0)
        vmin = min(float(phantom.min()), float(reconstructed_image.min()), 0.0)
        if vmin >= vmax:
            vmin = vmax - 0.1
        axes[1, 2].imshow(
            reconstructed_image, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, extent=image_extent
        )
        axes[1, 2].set_title("6. FBP Reconstruction (from U-Net Sinogram)")
        axes[1, 2].set_xlabel("X (mm)")
        axes[1, 2].set_ylabel("Z (mm)")
    else:
        axes[1, 2].axis("off")
        axes[1, 2].set_title("6. FBP Reconstruction unavailable (ASTRA not found)")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    return fig


def generate_example_figure(model, dataset, device, geometry, args, run) -> None:
    model.eval()
    idx = random.randrange(len(dataset))
    incomplete, full, phantom = dataset[idx]
    
    # Need to generate the acquired mask for Unet2dHLCC
    config = DBTGeometryConfig()
    full_angles_deg = np.rad2deg(geometry.angles)
    in_window = (full_angles_deg >= config.angle_min_deg) & (full_angles_deg <= config.angle_max_deg)
    acquired_mask_1d = torch.zeros(geometry.num_views, device=device)
    acquired_mask_1d[in_window] = 1.0
    acquired_mask = acquired_mask_1d.view(1, 1, geometry.num_views, 1)
    
    with torch.no_grad():
        inc_sino = incomplete.unsqueeze(0).to(device)
        refined_out = model(inc_sino, acquired_mask)

    reconstructed_image = None
    try:
        proj_geom, vol_geom = build_full_astra_geometries(config, tuple(phantom.shape[1:]))
        reconstructed_image = reconstruct_volume_fbp(
            refined_out.squeeze(0).squeeze(0).cpu().numpy() * 100.0, proj_geom, vol_geom
        )
    except Exception as exc:
        print(f"Skipping volume reconstruction panel ({exc}).")

    if not hasattr(args, 'figures_dir'):
        save_dir = Path("outputs/2d/figures_unet2dhlcc")
    else:
        save_dir = args.figures_dir
        
    save_path = save_dir / "example_after_training.png"
    dummy_baseline = torch.zeros_like(incomplete.squeeze(0))
    fig = plot_qualitative_example(
        phantom.squeeze(0), full.squeeze(0), incomplete.squeeze(0), dummy_baseline, refined_out.squeeze(0).squeeze(0).cpu(), reconstructed_image, config, save_path
    )
    print(f"Saved qualitative example to {save_path}")

    if run is not None:
        import wandb
        wandb.log({"example": wandb.Image(fig)})


def save_training_curve(train_losses: list[float], args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not train_losses:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Loss totale", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss (MSE + HLCC)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if not hasattr(args, 'figures_dir'):
        save_dir = Path("outputs/2d/figures_unet2dhlcc")
    else:
        save_dir = args.figures_dir
        
    save_path = save_dir / "training_losses.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)

if __name__ == "__main__":
    from src_2D.models.Unet2dHLCC.unet_hlcc import Unet2dHLCC
    from src_2D.data.dataset_2d import SinogramCompletionDataset
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Unet2dHLCC")
    parser.add_argument("--figures_dir", type=Path, default=Path("outputs/2d/figures_unet2dhlcc"))
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DBTGeometryConfig()
    geom = DBTGeometry(
        angles=config.full_angles,
        src_radius=config.src_radius_mm,
        det_radius=config.det_radius_mm,
        det_col_count=config.det_col_count,
        det_pixel_size=config.det_pixel_size_mm,
    )
    dataset = SinogramCompletionDataset(n_samples=args.n_samples, geometry_config=config, device=str(device))
    
    model = Unet2dHLCC(in_channels=1, out_channels=1, filters=16).to(device)
    weights_path = Path("outputs/2d/checkpoints_unet2dhlcc/best_unet2dhlcc_model.pt")
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("Loaded weights successfully.")
    else:
        print("Warning: weights not found, using untrained model.")
        
    generate_example_figure(model, dataset, device, geom, args, None)
