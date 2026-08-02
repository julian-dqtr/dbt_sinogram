from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DBTGeometryConfig:
    """Central configuration for a simplified digital breast tomosynthesis acquisition.

    Two angular ranges are defined:
    - The "full" range models a hypothetical wide-arc sinogram (ground truth) spanning
      ``full_angle_min_deg`` to ``full_angle_max_deg`` over ``num_views_full`` views.
    - The "limited" range (``num_views`` over ``angle_min_deg``/``angle_max_deg``) is the
      actual DBT acquisition window, centered inside the full range.

    This describes a single 2D slice (fan-beam) acquisition: images are 2D (rows x
    columns) and sinograms are 2D (views x detector pixels). It is the exact 2D
    equivalent of the previous 3D cone-beam geometry: the axis that used to be
    stationary/orthogonal to the rotation plane (``det_row_count`` / the volume's Y
    axis) is dropped entirely, since each of its slices was an independent fan-beam
    problem anyway. The remaining plane keeps the same source/detector radii, angles,
    detector width and pixel size as before.
    """

    num_views: int = 25
    angle_min_deg: float = -25.0
    angle_max_deg: float = 25.0
    num_views_full: int = 180
    full_angle_min_deg: float = -90.0
    full_angle_max_deg: float = 90.0
    src_radius_mm: float = 590.0
    det_radius_mm: float = 60.0
    det_col_count: int = 128
    det_pixel_size_mm: float = 2.5
    image_shape: tuple[int, int] = (128, 128)
    # ASTRA's CUDA 2D algorithms (FP_CUDA/SIRT_CUDA) require square pixels, so the
    # extent must scale with image_shape to keep both axes at the same mm/pixel.
    # +/-70mm keeps the phantom well inside this stationary-detector geometry's field
    # of view even at the extreme +/-90deg full-arc angles; going much beyond that
    # (e.g. +/-120mm) makes rays graze past the detector at those angles and produces
    # a wrapped-around, streaky sinogram instead of the expected sinusoidal trace.
    image_extent_mm: tuple[tuple[float, float], ...] = (
        (-70.0, 70.0),
        (-70.0, 70.0),
    )

    @property
    def angles(self) -> np.ndarray:
        """Angles actually acquired by the limited-angle DBT geometry."""
        return np.linspace(
            np.deg2rad(self.angle_min_deg),
            np.deg2rad(self.angle_max_deg),
            self.num_views,
        )

    @property
    def full_angles(self) -> np.ndarray:
        """Angles of the hypothetical full-arc sinogram used as ground truth."""
        return np.linspace(
            np.deg2rad(self.full_angle_min_deg),
            np.deg2rad(self.full_angle_max_deg),
            self.num_views_full,
        )

