from __future__ import annotations

import torch
import torch.nn as nn
from src_2D.models.utils.custom_unet import CustomUNet

class SinogramUNet(nn.Module):
    """2D U-Net wrapper using CustomUNet (eliminates checkerboard)."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, filters: int = 16) -> None:
        super().__init__()
        self.network = CustomUNet(in_channels=in_channels, out_channels=out_channels, filters=filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)