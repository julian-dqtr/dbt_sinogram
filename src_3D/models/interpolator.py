from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn


class SinusoidalViewInterpolator(nn.Module):
    """Baseline model that fills missing views by fitting a sinusoid along the angle axis.

    Vectorized version for 3D sinograms.
    Input shape: [B, C, Views, Det_Y, Det_X]
    """

    def __init__(
        self,
        angles_rad: Optional[np.ndarray] = None,
        fill_value: float = 0.0,
        ridge_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        if angles_rad is None:
            from conf.geometry import DBTGeometryConfig
            angles_rad = DBTGeometryConfig().full_angles
        angles_rad = np.asarray(angles_rad, dtype=np.float64)

        self.fill_value = fill_value
        self.ridge_lambda = ridge_lambda
        design = np.stack([np.cos(angles_rad), np.sin(angles_rad), np.ones_like(angles_rad)], axis=1)
        self.register_buffer("_design", torch.from_numpy(design).float(), persistent=False)

    def forward(self, incomplete_sinogram: torch.Tensor) -> torch.Tensor:
        if incomplete_sinogram.dim() != 5:
            raise ValueError(f"Expected a tensor of shape [B, C, views, det_Y, det_X], got {incomplete_sinogram.shape}")
        
        V = incomplete_sinogram.shape[2]
        if V != self._design.shape[0]:
            raise ValueError(
                f"Expected {self._design.shape[0]} views (one per configured angle), "
                f"got {V}"
            )

        B, C, V, Y, X = incomplete_sinogram.shape

        # Permute to put Views last: [B, C, Y, X, V] -> [N, V]
        flat_arr = incomplete_sinogram.permute(0, 1, 3, 4, 2).reshape(-1, V)
        
        # Identify which views were acquired (non-zero anywhere)
        valid_views_mask = (flat_arr.abs().sum(dim=0) > 0)
        num_valid = valid_views_mask.sum().item()
        
        # If not enough views to fit 3 parameters, just return fill_value
        if num_valid < 3:
            return torch.full_like(incomplete_sinogram, self.fill_value)
            
        a_valid = self._design[valid_views_mask]  # [num_valid, 3]
        y_valid = flat_arr[:, valid_views_mask]   # [N, num_valid]
        
        # A_T_A: [3, 3]
        A_T_A = a_valid.T @ a_valid + self.ridge_lambda * torch.eye(3, device=a_valid.device, dtype=a_valid.dtype)
        # A_T_Y: [3, N]
        A_T_Y = a_valid.T @ y_valid.T
        
        # coeffs_T: [3, N]
        coeffs_T = torch.linalg.solve(A_T_A, A_T_Y)
        
        # out_flat_T: [V, 3] @ [3, N] -> [V, N]
        out_flat_T = self._design @ coeffs_T
        
        # Transpose to [N, V]
        out_flat = out_flat_T.T
        
        # Reshape back to [B, C, Y, X, V]
        out = out_flat.reshape(B, C, Y, X, V)
        
        # Permute back to [B, C, V, Y, X]
        out = out.permute(0, 1, 4, 2, 3)
        
        return out
