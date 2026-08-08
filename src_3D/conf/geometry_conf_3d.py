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

    """

    num_views: int = 25
    angle_min_deg: float = -25.0
    angle_max_deg: float = 25.0
    num_views_full: int = 180
    full_angle_min_deg: float = -90.0
    full_angle_max_deg: float = 90.0
    src_radius_mm: float = 590.0
    det_radius_mm: float = 60.0
    det_row_count: int = 128
    det_col_count: int = 128
    det_pixel_size_mm: float = 2.5
    image_shape: tuple[int, int, int] = (32, 128, 128)

    image_extent_mm: tuple[tuple[float, float], ...] = (
        (0.0, 60.0),
        (-70.0, 70.0),
        (-70.0, 70.0),
    )

    @property
    def angles(self) -> np.ndarray:
        """Angles acquired by the limited-angle DBT geometry."""
        return np.linspace(
            np.deg2rad(self.angle_min_deg),
            np.deg2rad(self.angle_max_deg),
            self.num_views,
        )

    @property
    def full_angles(self) -> np.ndarray:
        """Angles of the full-arc sinogram used as ground truth."""
        return np.linspace(
            np.deg2rad(self.full_angle_min_deg),
            np.deg2rad(self.full_angle_max_deg),
            self.num_views_full,
        )

