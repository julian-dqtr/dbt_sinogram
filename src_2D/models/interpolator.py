from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn


class SinusoidalViewInterpolator(nn.Module):
    """Baseline model that fills missing views by fitting a sinusoid along the angle axis.

    For a fixed detector pixel, the projection value as a function of the rotation angle
    theta is classically well approximated by a single sinusoid
    ``a*cos(theta) + b*sin(theta) + c`` (this is exactly the sinogram trace of a point at a
    fixed radius from the isocenter, and a good first-order model for extended objects too).
    This fits that 3-parameter model per detector pixel using only the acquired (non-zero)
    views via ridge-regularized least squares, then evaluates it at every angle (acquired
    and missing) to fill the incomplete sinogram.

    The acquired views only span a narrow window (e.g. +/-25deg out of the full +/-90deg
    arc), where ``cos(theta)`` is nearly constant and thus nearly collinear with the
    constant term: an unregularized fit is ill-conditioned and its coefficients explode
    when evaluated far outside that window. ``ridge_lambda`` penalizes large coefficients
    to keep the extrapolation to the full angular range well-behaved.
    """

    def __init__(
        self,
        angles_rad: Optional[np.ndarray] = None,
        fill_value: float = 0.0,
        ridge_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        if angles_rad is None:
            from src_2D.conf.geometry import DBTGeometryConfig

            angles_rad = DBTGeometryConfig().full_angles
        angles_rad = np.asarray(angles_rad, dtype=np.float64)

        self.fill_value = fill_value
        self.ridge_lambda = ridge_lambda
        design = np.stack([np.cos(angles_rad), np.sin(angles_rad), np.ones_like(angles_rad)], axis=1)
        self.register_buffer("_design", torch.from_numpy(design).float(), persistent=False)

    def forward(self, incomplete_sinogram: torch.Tensor) -> torch.Tensor:
        if incomplete_sinogram.dim() != 4:
            raise ValueError("Expected a tensor of shape [B, C, views, detector_pixels]")
        if incomplete_sinogram.shape[2] != self._design.shape[0]:
            raise ValueError(
                f"Expected {self._design.shape[0]} views (one per configured angle), "
                f"got {incomplete_sinogram.shape[2]}"
            )

        arr = incomplete_sinogram.detach().cpu().numpy()
        design = self._design.cpu().numpy()
        n_params = design.shape[1]
        ridge = self.ridge_lambda * np.eye(n_params, dtype=design.dtype)
        out = np.zeros_like(arr, dtype=np.float32)

        for b in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                for w in range(arr.shape[3]):
                    values = arr[b, c, :, w]
                    valid = values != 0.0
                    if valid.sum() < n_params:
                        out[b, c, :, w] = self.fill_value
                        continue

                    a_valid = design[valid]
                    y_valid = values[valid]
                    coeffs = np.linalg.solve(a_valid.T @ a_valid + ridge, a_valid.T @ y_valid)
                    out[b, c, :, w] = design @ coeffs

        return torch.from_numpy(out).to(device=incomplete_sinogram.device, dtype=incomplete_sinogram.dtype)
