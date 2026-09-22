from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.geometry.dbt_geometry_2d import DBTGeometry


def get_soft_acquired_mask(geom: DBTGeometry, device: torch.device, blend_width_deg: float = 5.0) -> torch.Tensor:
    """
    Soft Data Consistency (DC) mask, shape [1, 1, num_views, 1] for broadcasting.

    The mask is used as ``out = mask * measured + (1 - mask) * predicted``. Since the
    measured sinogram is ZERO outside the acquired window, the mask must be exactly 0 on
    every missing view, otherwise the prediction would be blended with zeros (i.e.
    attenuated). The cosine taper therefore lives INSIDE the acquired window:

    - 0.0 on every missing view,
    - rises smoothly over ``blend_width_deg`` degrees starting from the first missing view,
    - 1.0 in the core of the acquired window.

    Every acquired view keeps a strictly positive weight, and the defining invariant is
    ``DC(ground_truth) == ground_truth`` (checked in tests/test_dc_mask.py).
    """
    angles_deg = np.rad2deg(np.asarray(geom.angles, dtype=np.float64))
    acquired = np.asarray(geom.acquired_view_mask, dtype=bool)
    mask = np.zeros(geom.num_views, dtype=np.float64)

    if acquired.any():
        idx = np.flatnonzero(acquired)
        first, last = idx[0], idx[-1]
        # Angular distance to the nearest missing view on each side. A side without any
        # missing view (window touching the end of the full range) needs no taper.
        dist_left = angles_deg - angles_deg[first - 1] if first > 0 else np.full_like(angles_deg, np.inf)
        dist_right = angles_deg[last + 1] - angles_deg if last < geom.num_views - 1 else np.full_like(angles_deg, np.inf)
        dist = np.minimum(dist_left, dist_right)

        if blend_width_deg > 0:
            ramp = np.clip(dist / blend_width_deg, 0.0, 1.0)
            taper = 0.5 * (1.0 - np.cos(np.pi * ramp))
        else:
            taper = np.ones_like(dist)
        mask[acquired] = taper[acquired]

    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
    return mask_tensor.view(1, 1, geom.num_views, 1)


def apply_data_consistency(measured: torch.Tensor, predicted: torch.Tensor, soft_mask: torch.Tensor) -> torch.Tensor:
    """Blend the measured views back into a predicted sinogram (see get_soft_acquired_mask)."""
    return soft_mask * measured + (1.0 - soft_mask) * predicted


def build_full_astra_geometries(geometry_config: DBTGeometryConfig, image_shape: Tuple[int, int]):
    """Build the ASTRA (proj_geom, vol_geom) pair of the full-range ground-truth sinogram."""
    import astra

    geometry = DBTGeometry.from_config(geometry_config)
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
    sirt_iterations: int = 100,
) -> Optional[np.ndarray]:
    """Reconstruct a 2D image from a sinogram shaped [views, cols] (raw, un-normalised units).

    Parallel-beam geometries use a true filtered back-projection (FBP_CUDA). ASTRA's FBP
    does not support the legacy 'fanflat_vec' geometry, for which SIRT_CUDA is used instead
    (an unfiltered back-projection is NOT a reconstruction).

    Returns ``None`` if ASTRA is unavailable, so callers can skip this panel gracefully.
    """
    try:
        import astra
    except ImportError:  # pragma: no cover - environment dependent
        return None

    sinogram_astra = np.ascontiguousarray(sinogram_view_col.astype(np.float32))
    use_fbp = proj_geom["type"] == "parallel"

    projector_id = astra.create_projector("cuda", proj_geom, vol_geom)
    sino_id = astra.data2d.create("-sino", proj_geom, sinogram_astra)
    reco_id = astra.data2d.create("-vol", vol_geom)
    try:
        cfg: Dict[str, Any] = astra.astra_dict("FBP_CUDA" if use_fbp else "SIRT_CUDA")
        cfg["ProjectorId"] = projector_id
        cfg["ProjectionDataId"] = sino_id
        cfg["ReconstructionDataId"] = reco_id
        if use_fbp:
            cfg["option"] = {"FilterType": "ram-lak"}
        else:
            cfg["option"] = {"MinConstraint": 0.0}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1 if use_fbp else sirt_iterations)
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
        ax.set_ylabel("Detector coordinate s (mm)")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(phantom.detach().cpu().numpy(), cmap="gray", origin="lower", extent=image_extent)
    axes[0, 0].set_title("1. Original Phantom")
    axes[0, 0].set_xlabel("X (mm)")
    axes[0, 0].set_ylabel("Z (mm)")

    plot_sinogram(axes[0, 1], full_sinogram, "2. Full Sinogram (-90 deg to +90 deg, parallel-beam)")

    plot_sinogram(
        axes[0, 2],
        incomplete_sinogram,
        f"3. Limited-angle Sinogram ("
        f"{geometry_config.angle_min_deg:.0f} deg to {geometry_config.angle_max_deg:.0f} deg)",
    )

    plot_sinogram(axes[1, 0], refined_sinogram, "4. Completed Sinogram (model output)")

    if reconstructed_image is not None:
        rec_min = float(np.nanmin(reconstructed_image)) if not np.isnan(reconstructed_image).all() else 0.0
        rec_max = max(float(np.nanmax(reconstructed_image)) if not np.isnan(reconstructed_image).all() else 1.0, rec_min + 1e-4)
        axes[1, 1].imshow(
            reconstructed_image, cmap="gray", origin="lower", vmin=rec_min, vmax=rec_max, extent=image_extent
        )
        axes[1, 1].set_title("5. FBP Reconstruction (from completed sinogram)")
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


def generate_example_figure(model, dataset, device, geometry, args, wandb_run=None, sample_index: int = 0):
    """Save a qualitative figure for one sample. Every model shares the ``model(incomplete)`` signature."""
    try:
        model.eval()
        incomplete, full, phantom = dataset[sample_index % len(dataset)]
        with torch.no_grad():
            refined_out = model(incomplete.unsqueeze(0).to(device))

        reconstructed_image = None
        try:
            proj_geom, vol_geom = build_full_astra_geometries(geometry, tuple(phantom.shape[1:]))
            reconstructed_image = reconstruct_volume_fbp(
                refined_out.squeeze(0).squeeze(0).cpu().numpy() * geometry.sino_norm, proj_geom, vol_geom
            )
        except Exception as exc:
            print(f"Skipping volume reconstruction panel ({exc}).")

        save_path = args.figures_dir / "example_after_training.png"
        plot_qualitative_example(
            phantom.squeeze(0).cpu(), full.squeeze(0).cpu(), incomplete.squeeze(0).cpu(),
            refined_out.squeeze(0).squeeze(0).cpu(), reconstructed_image, geometry, save_path
        )
        print(f"Saved qualitative example to {save_path}")

        if wandb_run is not None:
            import wandb
            wandb_run.log({"example_reconstruction": wandb.Image(str(save_path))})
    except Exception as exc:
        # A plotting failure must never cost a finished training run, but it must be visible.
        import traceback
        traceback.print_exc()
        print(f"Warning: Failed to generate example figure ({exc}). Continuing...")


def save_random_dataset_preview(dataset, geometry, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = 0
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
        ax.set_ylabel("Detector coordinate s (mm)")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(phantom.detach().cpu().squeeze(0).numpy(), cmap="gray", origin="lower", extent=image_extent, vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Sample #{idx} (Ground truth phantom)")
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


