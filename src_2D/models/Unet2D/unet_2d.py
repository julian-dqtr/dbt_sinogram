from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.models.utils.custom_unet import CustomUNet
from src_2D.utils.evaluation import apply_data_consistency, get_soft_acquired_mask


class SinogramUNet(nn.Module):
    """2D U-Net on the sinogram (CustomUNet backbone: Resize+Conv, no checkerboard).

    Same interface as every other model of the repo: ``completed = model(incomplete)`` with
    [B, 1, Views, Detectors] tensors. The network predicts a residual on top of its input
    and, if ``data_consistency`` is set, the measured views are blended back with the soft
    mask of ``get_soft_acquired_mask`` (taper inside the acquired window).
    """

    # Buffer registered in __init__, declared here so static checkers know its type.
    soft_mask: torch.Tensor

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        filters: int = 16,
        data_consistency: bool = True,
        blend_width_deg: float = 5.0,
        geometry_config: Optional[DBTGeometryConfig] = None,
    ) -> None:
        super().__init__()
        if filters <= 0:
            raise ValueError("filters must be a positive integer")

        self.network = CustomUNet(in_channels=in_channels, out_channels=out_channels, filters=filters)
        self.data_consistency = data_consistency

        geom = DBTGeometry.from_config(geometry_config or DBTGeometryConfig())
        # Derived from the geometry stored in the checkpoint, hence not persistent.
        self.register_buffer(
            "soft_mask", get_soft_acquired_mask(geom, torch.device("cpu"), blend_width_deg), persistent=False
        )

    def forward(self, x: torch.Tensor, apply_dc: Optional[bool] = None) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected input shape [B, C, Views, Detectors], got {tuple(x.shape)}")

        out = x + self.network(x)  # residual learning (padding handled inside CustomUNet)

        use_dc = self.data_consistency if apply_dc is None else apply_dc
        if use_dc:
            out = apply_data_consistency(x, out, self.soft_mask)
        return out
