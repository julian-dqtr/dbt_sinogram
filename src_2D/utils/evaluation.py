from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.geometry.dbt_geometry_2d import DBTGeometry


def get_soft_acquired_mask(geom: DBTGeometry, device: torch.device, blend_width_deg: float = 5.0) -> torch.Tensor:
    """
    Creates a Soft Data Consistency mask.
    Returns a tensor of shape [1, 1, num_views, 1] for broadcasting.
    Values are 1.0 inside the acquired angular window, and taper to 0.0 smoothly 
    using a cosine window over `blend_width_deg` degrees on both sides.
    """
    config = DBTGeometryConfig()
    angles_deg = np.rad2deg(geom.angles)
    mask = np.zeros_like(angles_deg, dtype=np.float32)
    
    min_deg, max_deg = config.angle_min_deg, config.angle_max_deg
    
    for i, angle in enumerate(angles_deg):
        if angle < min_deg - blend_width_deg:
            mask[i] = 0.0
        elif angle > max_deg + blend_width_deg:
            mask[i] = 0.0
        elif min_deg <= angle <= max_deg:
            mask[i] = 1.0
        elif angle < min_deg:
            # Taper from 0 to 1 over blend_width_deg
            x = (angle - (min_deg - blend_width_deg)) / blend_width_deg
            mask[i] = 0.5 * (1 - np.cos(np.pi * x))
        else: 
            # Taper from 1 to 0 over blend_width_deg
            x = (angle - max_deg) / blend_width_deg
            mask[i] = 0.5 * (1 + np.cos(np.pi * x))
            
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
    return mask_tensor.view(1, 1, geom.num_views, 1)


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
    """Reconstruct a 2D image (FBP_CUDA) from a sinogram shaped [views, cols].

    Returns ``None`` if ASTRA is unavailable, so callers can skip this panel gracefully.
    """
    try:
        import astra
    except ImportError:  # pragma: no cover - environment dependent
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
    refined_sinogram: torch.Tensor,
    reconstructed_image: Optional[np.ndarray],
    geometry_config: DBTGeometryConfig,
    save_path: Path,
):
    """Save (and return) a figure comparing the GT phantom, sinograms, and the U-Net output.

    Layout and style follow ``geometry/archive/smallgeometryV2.py``: physical units on the
    axes (mm for the image, degrees for sinogram views, mm for detector width), numbered
    panel titles, and the same x-axis range enforced across all sinogram panels for a direct
    visual comparison.

    Panels: (1) GT phantom, (2) full sinogram (-90..+90deg), (3) limited/incomplete
    sinogram (the acquired +/-angle window, black elsewhere), (4) baseline (linear
    interpolation) reconstruction, (5) U-Net reconstruction, and (6, if ASTRA is available)
    the image reconstructed via FBP from the U-Net's sinogram.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sino_vmax = full_sinogram.max().item() or 1.0

    row_min, row_max = geometry_config.image_extent_mm[0]
    col_min, col_max = geometry_config.image_extent_mm[1]
    image_extent = [col_min, col_max, row_min, row_max]

    angle_min, angle_max = geometry_config.full_angle_min_deg, geometry_config.full_angle_max_deg
    detector_half_width = geometry_config.det_col_count * geometry_config.det_pixel_size_mm / 2.0
    sino_extent = [angle_min, angle_max, -detector_half_width, detector_half_width]

    def plot_sinogram(ax, sinogram: torch.Tensor, title: str):
        # [views, cols] -> transpose to [cols, views] so columns map to the angle axis.
        image = sinogram.detach().cpu().numpy().T
        vmin_val = float(np.nanmin(image)) if not np.isnan(image).all() else 0.0
        vmax_val = max(float(np.nanmax(image)) if not np.isnan(image).all() else 1.0, vmin_val + 1e-4)
        ax.imshow(image, cmap="bone", origin="lower", aspect="auto", extent=sino_extent, vmin=vmin_val, vmax=vmax_val)
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

    plot_sinogram(axes[1, 0], refined_sinogram, "4. U-Net Reconstruction")

    if reconstructed_image is not None:
        rec_min = float(np.nanmin(reconstructed_image)) if not np.isnan(reconstructed_image).all() else 0.0
        rec_max = max(float(np.nanmax(reconstructed_image)) if not np.isnan(reconstructed_image).all() else 1.0, rec_min + 1e-4)
        axes[1, 1].imshow(
            reconstructed_image, cmap="gray", origin="lower", vmin=rec_min, vmax=rec_max, extent=image_extent
        )
        axes[1, 1].set_title("5. FBP Reconstruction (from U-Net Sinogram)")
        axes[1, 1].set_xlabel("X (mm)")
        axes[1, 1].set_ylabel("Z (mm)")
    else:
        axes[1, 1].axis("off")
        axes[1, 1].set_title("5. FBP Reconstruction unavailable (ASTRA not found)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    return fig


import random


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


def generate_example_figure(model, dataset, device, geometry, args, wandb_run=None, acquired_mask=None):
    try:
        model.eval()
        idx = random.randrange(len(dataset))
        incomplete, full, phantom = dataset[idx]
        with torch.no_grad():
            if acquired_mask is not None:
                refined_out = model(incomplete.unsqueeze(0).to(device), acquired_mask)
            else:
                refined_out = model(incomplete.unsqueeze(0).to(device))

        reconstructed_image = None
        try:
            proj_geom, vol_geom = build_full_astra_geometries(geometry, tuple(phantom.shape[1:]))
            reconstructed_image = reconstruct_volume_fbp(
                refined_out.squeeze(0).squeeze(0).cpu().numpy() * 100.0, proj_geom, vol_geom
            )
        except Exception as exc:
            print(f"Skipping volume reconstruction panel ({exc}).")

        save_path = args.figures_dir / "example_after_training.png"
        fig = plot_qualitative_example(
            phantom.squeeze(0), full.squeeze(0), incomplete.squeeze(0), refined_out.squeeze(0).squeeze(0).cpu(), reconstructed_image, geometry, save_path
        )
        print(f"Saved qualitative example to {save_path}")

        if wandb_run is not None:
            import wandb
            wandb_run.log({"example_reconstruction": wandb.Image(str(save_path))})
    except Exception as exc:
        print(f"Warning: Failed to generate example figure ({exc}). Continuing...")


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


