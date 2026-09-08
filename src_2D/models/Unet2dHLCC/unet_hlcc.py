from __future__ import annotations

import torch
import torch.nn as nn
from src_2D.models.utils.custom_unet import CustomUNet


class Unet2dHLCC(nn.Module):
    """
    2D U-Net (CustomUNet backbone) operating on the sinogram domain,
    trained with Helgason-Ludwig Consistency Conditions (HLCC).

    This is the direct counterpart of SinogramUNet (classic UNet2D) with two additions:
    1. Residual learning: the network predicts a *delta* to add to the incomplete sinogram.
    2. Soft Data Consistency: acquired views are blended back using a cosine-tapered mask,
       preserving measured data while avoiding hard boundary discontinuities that would
       violate HLCC (sharp edges generate spurious moments).

    Backbone: CustomUNet
    - Bilinear upsample + Conv (no ConvTranspose) → zero checkerboard artifacts
    - Built-in dynamic padding → works for any sinogram size
    - Same architecture as SinogramUNet → apples-to-apples comparison
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, filters: int = 16) -> None:
        super().__init__()
        if filters <= 0:
            raise ValueError("filters must be a positive integer")

        self.network = CustomUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            filters=filters,
        )

    def forward(self, x: torch.Tensor, acquired_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:             [B, C, Angles, Detectors]  — incomplete sinogram (zeros outside FOV)
            acquired_mask: [1, 1, Angles, 1]           — soft mask: 1.0 inside FOV, tapers to 0 outside
        Returns:
            out_sino:      [B, C, Angles, Detectors]  — completed sinogram
        """
        if x.dim() != 4:
            raise ValueError(f"Expected input shape [B, C, Angles, Detectors], got {x.shape}")

        # Network predicts residual (padding handled inside CustomUNet)
        pred_residual = self.network(x)

        # Residual learning: add predicted residual to the input
        out_sino = x + pred_residual

        # Soft Data Consistency:
        # In acquired regions   → keep measured values (weight = acquired_mask ≈ 1)
        # In missing regions    → use network prediction (weight = 1 - acquired_mask ≈ 1)
        # At boundaries         → smooth cosine blend, no hard discontinuity
        out_sino = acquired_mask * x + (1.0 - acquired_mask) * out_sino

        return out_sino
