from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DBTGeometryConfig:
    """Central configuration of the 2D limited-angle sinogram completion problem.

    Two angular ranges are defined:

    - The "full" range is the ground-truth sinogram: ``num_views_full`` views uniformly
      spread over [``full_angle_min_deg``, ``full_angle_max_deg``). The upper bound is
      EXCLUDED: in parallel-beam the views at -90 and +90 degrees are the same line
      integrals (detector flipped), so [-90, 90) is exactly one period of the Radon
      transform, sampled with a clean 1 degree step.

    - The "acquired" window [``angle_min_deg``, ``angle_max_deg``] (bounds included) is the
      limited-angle acquisition. Every full-range view outside of it must be completed.

    ``beam`` selects the ASTRA projection model:

    - ``"parallel"`` (default): rotating parallel-beam. The ground truth then satisfies
      the Helgason-Ludwig consistency conditions (HLCC) up to discretisation error.
    - ``"fanflat_vec"``: legacy stationary-detector fan-beam (DBT-like). Kept only to
      document why it was abandoned: its ground truth violates the parallel-beam HLCC by
      35 % (order 0) and 42 % (order 1) over the full arc.

    The detector (128 x 1.6 mm = 204.8 mm) is just wider than the diagonal of the
    140 mm x 140 mm object (198 mm), so no view is truncated.
    """

    beam: str = "parallel"
    angle_min_deg: float = -25.0
    angle_max_deg: float = 25.0
    num_views_full: int = 180
    full_angle_min_deg: float = -90.0
    full_angle_max_deg: float = 90.0
    det_col_count: int = 128
    det_pixel_size_mm: float = 1.6
    image_shape: tuple[int, int] = (128, 128)

    image_extent_mm: tuple[tuple[float, float], ...] = (
        (-70.0, 70.0),
        (-70.0, 70.0),
    )

    # Only used by the legacy "fanflat_vec" beam.
    src_radius_mm: float = 590.0
    det_radius_mm: float = 60.0

    # Divisor applied to raw line integrals (mm) to bring sinograms to O(1) values.
    sino_norm: float = 100.0

    @property
    def full_angles(self) -> np.ndarray:
        """Angles (radians) of the full-range ground-truth sinogram, upper bound excluded."""
        return np.linspace(
            np.deg2rad(self.full_angle_min_deg),
            np.deg2rad(self.full_angle_max_deg),
            self.num_views_full,
            endpoint=False,
        )

    @property
    def angle_step_deg(self) -> float:
        return (self.full_angle_max_deg - self.full_angle_min_deg) / self.num_views_full

    @property
    def acquired_view_mask(self) -> np.ndarray:
        """Boolean mask [num_views_full]: True for the views inside the acquired window."""
        angles_deg = np.rad2deg(self.full_angles)
        eps = 1e-6
        return (angles_deg >= self.angle_min_deg - eps) & (angles_deg <= self.angle_max_deg + eps)

    @property
    def angles(self) -> np.ndarray:
        """Angles (radians) of the acquired views (subset of ``full_angles``)."""
        return self.full_angles[self.acquired_view_mask]

    @property
    def num_views(self) -> int:
        """Number of acquired views."""
        return int(self.acquired_view_mask.sum())

    def to_dict(self) -> dict:
        """JSON-serialisable description, stored inside every checkpoint."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DBTGeometryConfig":
        data = dict(data)
        data["image_shape"] = tuple(data["image_shape"])
        data["image_extent_mm"] = tuple(tuple(extent) for extent in data["image_extent_mm"])
        return cls(**data)
