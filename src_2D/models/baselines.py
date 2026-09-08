"""
Non-learned baselines for sinogram completion.

These serve as mandatory references in the comparative evaluation:
- ZeroFilling: the "non-method" (missing views stay at zero).
- LinearInterpolation: 1D interpolation along the angular axis for each detector pixel.
"""
from __future__ import annotations

import numpy as np
import torch

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig


def zero_filling(incomplete_sino: torch.Tensor) -> torch.Tensor:
    """
    Returns the incomplete sinogram as-is.
    Missing views remain at zero — the simplest possible baseline.

    Args:
        incomplete_sino: [B, 1, V, D] or [V, D] incomplete sinogram.
    Returns:
        Same tensor unchanged.
    """
    return incomplete_sino.clone()


def linear_interpolation(incomplete_sino: torch.Tensor) -> torch.Tensor:
    """
    Performs 1D linear interpolation along the angular axis for each detector pixel,
    using the acquired (non-zero) views as support points.

    Args:
        incomplete_sino: [B, 1, V, D] or [V, D] incomplete sinogram.
    Returns:
        Completed sinogram with the same shape.
    """
    squeeze_batch = False
    if incomplete_sino.dim() == 2:
        incomplete_sino = incomplete_sino.unsqueeze(0).unsqueeze(0)
        squeeze_batch = True
    elif incomplete_sino.dim() == 3:
        incomplete_sino = incomplete_sino.unsqueeze(0)
        squeeze_batch = True

    B, C, V, D = incomplete_sino.shape
    result = incomplete_sino.clone()

    for b in range(B):
        sino = result[b, 0].cpu().numpy()  # [V, D]

        # Determine which views are acquired (non-zero rows)
        row_sums = np.abs(sino).sum(axis=1)
        acquired_mask = row_sums > 1e-6
        acquired_indices = np.where(acquired_mask)[0]
        missing_indices = np.where(~acquired_mask)[0]

        if len(acquired_indices) < 2 or len(missing_indices) == 0:
            continue

        # For each detector pixel, interpolate across the angular dimension
        for d in range(D):
            acquired_values = sino[acquired_indices, d]
            # Use numpy interp (linear, with extrapolation at boundaries)
            sino[missing_indices, d] = np.interp(
                missing_indices,
                acquired_indices,
                acquired_values,
            )

        result[b, 0] = torch.from_numpy(sino).to(result.device)

    if squeeze_batch:
        result = result.squeeze(0)

    return result
