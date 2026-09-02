from __future__ import annotations

import astra
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DynUNet

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.models.Unet2dRNO.astra_autograd import AstraProjection, AstraBackProjection


class RadonInformedUNet(nn.Module):
    """
    Dual-domain network utilizing differentiable ASTRA projections.
    Takes an incomplete sinogram, back-projects it to the spatial domain,
    applies a U-Net to refine the image, and forward-projects it back to a full sinogram.
    """

    def __init__(
        self,
        geometry_config: DBTGeometryConfig,
        in_channels: int = 1,
        out_channels: int = 1,
        filters: int = 16,
    ) -> None:
        super().__init__()
        if filters <= 0:
            raise ValueError("filters must be a positive integer")

        # 1. Initialize ASTRA geometries ONCE to save overhead
        self._setup_astra_geometries(geometry_config)

        # 2. Setup the spatial U-Net (DynUNet from MONAI)
        strides = [[1, 1], [2, 2], [2, 2], [2, 2]]
        self.spatial_unet = DynUNet(
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
        
        # DynUNet padding requirement
        self._divisor = [1, 1]
        for stride in strides:
            for axis, s in enumerate(stride):
                self._divisor[axis] *= s

    def _setup_astra_geometries(self, config: DBTGeometryConfig):
        """Pre-compute the projector geometries for the forward/backward passes."""
        rows, cols = config.image_shape
        row_min, row_max = config.image_extent_mm[0]
        col_min, col_max = config.image_extent_mm[1]
        
        self.vol_geom = astra.create_vol_geom(rows, cols, col_min, col_max, row_min, row_max)
        
        # Create full projection geometry
        full_geometry = DBTGeometry(
            angles=config.full_angles,
            src_radius=config.src_radius_mm,
            det_radius=config.det_radius_mm,
            det_col_count=config.det_col_count,
            det_pixel_size=config.det_pixel_size_mm,
        )
        self.proj_geom_full = full_geometry.get_astra_proj_geom()
        self.projector_id_full = astra.create_projector("cuda", self.proj_geom_full, self.vol_geom)

    def _pad_to_divisor(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = (-h) % self._divisor[0]
        pad_w = (-w) % self._divisor[1]
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

    @staticmethod
    def _crop_to_shape(x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        h, w = shape
        return x[..., :h, :w]

    def forward(self, limited_sinogram: torch.Tensor) -> torch.Tensor:
        if limited_sinogram.dim() != 4:
            raise ValueError("Expected input shape [B, C, Angles, Detectors]")

        # Step 1: Analytical transformation from Sinogram to Image space
        # limited_sinogram has the full (180, 128) shape but with zeroes outside the limited window.
        # So we can use the full projection geometry directly.
        artifacted_image = AstraBackProjection.apply(
            limited_sinogram, 
            self.vol_geom, 
            self.proj_geom_full, 
            self.projector_id_full
        )

        # Step 2: Spatial correction using the U-Net
        original_shape = artifacted_image.shape[-2:]
        padded_image = self._pad_to_divisor(artifacted_image)
        cleaned_image = self.spatial_unet(padded_image)
        cleaned_image = self._crop_to_shape(cleaned_image, original_shape)

        # Step 3: Analytical projection back to full Sinogram space
        full_sinogram_pred = AstraProjection.apply(
            cleaned_image, 
            self.vol_geom, 
            self.proj_geom_full, 
            self.projector_id_full
        )

        # Match dynamic range (min-max normalization to [0, 1])
        B = full_sinogram_pred.shape[0]
        v_min = full_sinogram_pred.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        v_max = full_sinogram_pred.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        full_sinogram_pred = (full_sinogram_pred - v_min) / (v_max - v_min + 1e-8)

        return full_sinogram_pred
