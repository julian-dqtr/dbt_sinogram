from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DynUNet

class Unet2dHLCC(nn.Module):
    """
    Classic 2D U-Net operating on the sinogram domain, 
    intended to be trained with Helgason-Ludwig Consistency Conditions (HLCC).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, filters: int = 16) -> None:
        super().__init__()
        if filters <= 0:
            raise ValueError("filters must be a positive integer")

        strides = [[1, 1], [2, 2], [2, 2], [2, 2]]
        self.network = DynUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[[3, 3], [3, 3], [3, 3], [3, 3]],
            filters=[filters, filters * 2, filters * 4, filters * 8],
            strides=strides,
            upsample_kernel_size=[[2, 2], [2, 2], [2, 2]],
            norm_name="instance",
            deep_supervision=False,
        )
        
        # DynUNet downsamples spatially by the product of strides along each axis
        # The input must be padded to a multiple of this factor.
        self._divisor = [1, 1]
        for stride in strides:
            for axis, s in enumerate(stride):
                self._divisor[axis] *= s

    def forward(self, x: torch.Tensor, acquired_mask: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected input shape [B, C, Angles, Detectors], got {x.shape}")

        original_shape = x.shape[-2:]
        padded_x = self._pad_to_divisor(x)
        out = self.network(padded_x)
        pred_full = self._crop_to_shape(out, original_shape)
        
        # Residual learning: the network predicts the missing data 
        # and smooths the transitions. We simply add the prediction to the input.
        # Since x is 0 in the missing region, pred_full provides the missing data.
        # The network can also learn to predict small negative values inside the
        # acquired region to smooth out noise, avoiding sharp "walls".
        out_sino = x + pred_full
        
        return out_sino

    def _pad_to_divisor(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = (-h) % self._divisor[0]
        pad_w = (-w) % self._divisor[1]
        if pad_h or pad_w:
            # F.pad takes padding from the last dimension backwards: (W, H).
            # Using 'replicate' instead of default zeros to prevent boundary artifacts (checkerboard)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x

    @staticmethod
    def _crop_to_shape(x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        h, w = shape
        return x[..., :h, :w]
