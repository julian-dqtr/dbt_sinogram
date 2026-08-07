from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from conf.geometry import DBTGeometryConfig
from geometry.DBT_Geometry import DBTGeometry


def build_full_astra_geometries(geometry_config: DBTGeometryConfig, image_shape: Tuple[int, int]):
    """Build the ASTRA (proj_geom, vol_geom) pair for the full-arc (-90..+90deg) sinogram."""
    import astra

    geometry = DBTGeometry(
        angles=geometry_config.full_angles,
        src_radius=geometry_config.src_radius_mm,
        det_radius=geometry_config.det_radius_mm,
        det_row_count=geometry_config.det_row_count,
        det_col_count=geometry_config.det_col_count,
        det_pixel_size=geometry_config.det_pixel_size_mm,
    )
    proj_geom = geometry.get_astra_proj_geom()

    depth, rows, cols = image_shape
    z_min, z_max = geometry_config.image_extent_mm[0]
    y_min, y_max = geometry_config.image_extent_mm[1]
    x_min, x_max = geometry_config.image_extent_mm[2]
    
    vol_geom = astra.create_vol_geom(rows, cols, depth, x_min, x_max, y_min, y_max, z_min, z_max)
    return proj_geom, vol_geom


def reconstruct_volume_sirt(
    sinogram_view_col: np.ndarray,
    proj_geom,
    vol_geom,
    n_iterations: int = 30,
) -> Optional[np.ndarray]:
    """Reconstruct a 2D image (SIRT_CUDA) from a sinogram shaped [views, cols].

    Returns ``None`` if ASTRA is unavailable, so callers can skip this panel gracefully.
    """
    try:
        import astra
    except ImportError:  # pragma: no cover - environment dependent
        return None

    sinogram_astra = np.ascontiguousarray(sinogram_view_col.astype(np.float32))
    # Transpose for ASTRA 3D SIRT (Views, Det_Y, Det_X) -> (Det_Y, Views, Det_X)
    sinogram_astra = np.transpose(sinogram_astra, (1, 0, 2))

    projector_id = astra.create_projector("cuda3d", proj_geom, vol_geom)
    sino_id = astra.data3d.create("-sino", proj_geom, sinogram_astra)
    reco_id = astra.data3d.create("-vol", vol_geom)
    try:
        cfg = astra.astra_dict("SIRT3D_CUDA")
        cfg["ProjectorId"] = projector_id
        cfg["ProjectionDataId"] = sino_id
        cfg["ReconstructionDataId"] = reco_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, n_iterations)
        reconstruction = np.asarray(astra.data3d.get(reco_id))
        astra.algorithm.delete(alg_id)
    finally:
        astra.data3d.delete(reco_id)
        astra.data3d.delete(sino_id)
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
    """Save (and return) a figure comparing the GT phantom, sinograms, and the U-Net output.

    Layout and style follow ``geometry/archive/smallgeometryV2.py``: physical units on the
    axes (mm for the image, degrees for sinogram views, mm for detector width), numbered
    panel titles, and the same x-axis range enforced across all sinogram panels for a direct
    visual comparison.

    Panels: (1) GT phantom, (2) full sinogram (-90..+90deg), (3) limited/incomplete
    sinogram (the acquired +/-angle window, black elsewhere), (4) baseline (linear
    interpolation) reconstruction, (5) U-Net reconstruction, and (6, if ASTRA is available)
    the image reconstructed via SIRT from the U-Net's sinogram.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sino_vmax = full_sinogram.max().item() or 1.0

    row_min, row_max = geometry_config.image_extent_mm[1]
    col_min, col_max = geometry_config.image_extent_mm[2]
    image_extent = [col_min, col_max, row_min, row_max]

    angle_min, angle_max = geometry_config.full_angle_min_deg, geometry_config.full_angle_max_deg
    detector_half_width = geometry_config.det_col_count * geometry_config.det_pixel_size_mm / 2.0
    sino_extent = [angle_min, angle_max, -detector_half_width, detector_half_width]

    def plot_sinogram(ax, sinogram: torch.Tensor, title: str):
        # [views, cols] -> transpose to [cols, views] so columns map to the angle axis.
        image = sinogram.detach().cpu().numpy().T
        ax.imshow(image, cmap="bone", origin="lower", aspect="auto", extent=sino_extent, vmin=0, vmax=sino_vmax)
        ax.set_xlim(angle_min, angle_max)
        ax.set_title(title)
        ax.set_xlabel(r"Angle $\phi$ (degrees)")
        ax.set_ylabel("Detector width u (mm)")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(phantom.detach().cpu().numpy(), cmap="gray", origin="lower", extent=image_extent, vmin=-1.0, vmax=1.0)
    axes[0, 0].set_title("1. Original Phantom Slice")
    axes[0, 0].set_xlabel("X (mm)")
    axes[0, 0].set_ylabel("Y (mm)")

    plot_sinogram(axes[0, 1], full_sinogram, "2. Full Sinogram (-90 deg to +90 deg)")

    plot_sinogram(
        axes[0, 2],
        incomplete_sinogram,
        f"3. Limited Sinogram (Stationary DBT, "
        f"{geometry_config.angle_min_deg:.0f} deg to {geometry_config.angle_max_deg:.0f} deg)",
    )

    plot_sinogram(axes[1, 0], baseline_sinogram, "4. Baseline Reconstruction (Sinusoidal Fit)")

    plot_sinogram(axes[1, 1], refined_sinogram, "5. U-Net Reconstruction")

    if reconstructed_image is not None:
        vmax = max(float(phantom.max()), 1.0)
        axes[1, 2].imshow(
            reconstructed_image, cmap="gray", origin="lower", vmin=-1.0, vmax=vmax, extent=image_extent
        )
        axes[1, 2].set_title("6. SIRT Reconstruction Slice")
        axes[1, 2].set_xlabel("X (mm)")
        axes[1, 2].set_ylabel("Y (mm)")
    else:
        axes[1, 2].axis("off")
        axes[1, 2].set_title("6. SIRT Reconstruction unavailable (ASTRA not found)")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    return fig

