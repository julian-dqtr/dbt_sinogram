from __future__ import annotations

import numpy as np


class DBTGeometry:
    """2D projection geometry of the full-range (ground-truth) sinogram.

    Angle convention (shared by both beams): at angle ``theta`` the source sits in the
    direction (sin theta, cos theta), i.e. on the +y axis at theta = 0, and the rays travel
    towards (-sin theta, -cos theta).

    - ``beam="parallel"`` (default): rotating parallel-beam. The detector axis is
      u(theta) = (cos theta, -sin theta), perpendicular to the rays, so the measured data
      is the Radon transform p(theta, s) with s = x . u(theta).
    - ``beam="fanflat_vec"``: legacy stationary-detector fan-beam (source on an arc of
      radius ``src_radius``, flat detector fixed at y = -``det_radius``). Its sinogram is
      NOT a Radon transform in (theta, s) and violates the parallel-beam HLCC.
    """

    def __init__(
        self,
        angles,
        src_radius=None,
        det_radius=None,
        det_col_count=128,
        det_pixel_size=1.6,
        beam="parallel",
        acquired_window_deg=(-25.0, 25.0),
    ):
        if beam not in ("parallel", "fanflat_vec"):
            raise ValueError(f"Unknown beam '{beam}'. Use 'parallel' or 'fanflat_vec'.")
        if beam == "fanflat_vec" and (src_radius is None or det_radius is None):
            raise ValueError("The 'fanflat_vec' beam requires src_radius and det_radius.")

        self.beam = beam
        self.angles = np.asarray(angles, dtype=np.float64)
        self.num_views = len(self.angles)
        self.src_radius = src_radius
        self.det_radius = det_radius
        self.det_col_count = det_col_count
        self.det_pixel_size = det_pixel_size
        self.acquired_window_deg = tuple(acquired_window_deg)

        # Only the legacy beam needs explicit per-view vectors.
        self.vectors = self._generate_fan_vectors() if beam == "fanflat_vec" else None

    @classmethod
    def from_config(cls, config) -> "DBTGeometry":
        """Build the full-range geometry described by a ``DBTGeometryConfig``."""
        return cls(
            angles=config.full_angles,
            src_radius=config.src_radius_mm,
            det_radius=config.det_radius_mm,
            det_col_count=config.det_col_count,
            det_pixel_size=config.det_pixel_size_mm,
            beam=config.beam,
            acquired_window_deg=(config.angle_min_deg, config.angle_max_deg),
        )

    @property
    def angle_step_deg(self) -> float:
        return float(np.rad2deg(np.mean(np.diff(self.angles))))

    @property
    def acquired_view_mask(self) -> np.ndarray:
        """Boolean mask [num_views]: True for the views inside the acquired window."""
        angles_deg = np.rad2deg(self.angles)
        lo, hi = self.acquired_window_deg
        eps = 1e-6
        return (angles_deg >= lo - eps) & (angles_deg <= hi + eps)

    def _generate_fan_vectors(self):
        if self.src_radius is None or self.det_radius is None:
            raise ValueError("The 'fanflat_vec' beam requires src_radius and det_radius.")
        # Row format expected by ASTRA 'fanflat_vec': [src_x, src_y, det_x, det_y, u_x, u_y]
        vectors = np.zeros((self.num_views, 6))
        vectors[:, 0] = self.src_radius * np.sin(self.angles)
        vectors[:, 1] = self.src_radius * np.cos(self.angles)
        # Stationary detector below the isocenter, U scaled by the pixel size.
        vectors[:, 2] = 0.0
        vectors[:, 3] = -self.det_radius
        vectors[:, 4] = self.det_pixel_size
        vectors[:, 5] = 0.0
        return vectors

    def get_astra_proj_geom(self):
        import astra

        if self.beam == "parallel":
            # ASTRA's 'parallel' convention is ray = (sin a, -cos a), u = (cos a, sin a).
            # Passing a = -theta yields ray = (-sin theta, -cos theta) and
            # u = (cos theta, -sin theta), i.e. the convention documented above.
            return astra.create_proj_geom(
                "parallel", self.det_pixel_size, self.det_col_count, -self.angles
            )
        return astra.create_proj_geom("fanflat_vec", self.det_col_count, self.vectors)
