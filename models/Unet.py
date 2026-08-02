from __future__ import annotations

import torch
import torch.nn.functional as F
from monai.networks.nets import DynUNet


class SinogramUNet(torch.nn.Module):
    """Learned refiner for limited-angle sinogram completion."""

    def __init__(self, in_channels: int = 2, out_channels: int = 1) -> None:
        super().__init__()
        strides = [[1, 1], [2, 2], [2, 2], [2, 2]]
        self.network = DynUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[[3, 3], [3, 3], [3, 3], [3, 3]],
            filters=[16, 32, 64, 128],
            strides=strides,
            upsample_kernel_size=[[2, 2], [2, 2], [2, 2]],
            norm_name="instance",
            deep_supervision=False,
        )
        # DynUNet downsamples spatially by the product of strides along each axis
        # (here 2*2*2=8 for H/W). The input must be a multiple of this factor so
        # that encoder/decoder feature maps line up at the skip connections.
        self._divisor = [1, 1]
        for stride in strides:
            for axis, s in enumerate(stride):
                self._divisor[axis] *= s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("Expected input shape [B, C, H, W]")

        original_shape = x.shape[-2:]
        x = self._pad_to_divisor(x)
        out = self.network(x)
        return self._crop_to_shape(out, original_shape)

    def _pad_to_divisor(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = (-h) % self._divisor[0]
        pad_w = (-w) % self._divisor[1]
        if pad_h or pad_w:
            # F.pad takes padding from the last dimension backwards: (W, H).
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

    @staticmethod
    def _crop_to_shape(x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        h, w = shape
        return x[..., :h, :w]


