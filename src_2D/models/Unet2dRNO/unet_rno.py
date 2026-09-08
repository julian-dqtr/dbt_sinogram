from __future__ import annotations

import astra
import torch
import torch.nn as nn

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.models.Unet2dRNO.astra_autograd import AstraProjection, AstraBackProjection
from src_2D.models.utils.custom_unet import CustomUNet


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

        # 2. Setup the spatial U-Net (CustomUNet with Resize+Conv)
        self.spatial_unet = CustomUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            filters=filters
        )

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

    def forward(self, limited_sinogram: torch.Tensor) -> torch.Tensor:
        if limited_sinogram.dim() != 4:
            raise ValueError("Expected input shape [B, C, Angles, Detectors]")

        # Step 1: Analytical transformation from Sinogram to Image space
        artifacted_image = AstraBackProjection.apply(
            limited_sinogram, 
            self.vol_geom, 
            self.proj_geom_full, 
            self.projector_id_full
        )

        # Step 2: Spatial correction using the U-Net
        # Padding is handled automatically inside CustomUNet
        cleaned_image = self.spatial_unet(artifacted_image)

        # Step 3: Analytical projection back to full Sinogram space
        full_sinogram_pred = AstraProjection.apply(
            cleaned_image, 
            self.vol_geom, 
            self.proj_geom_full, 
            self.projector_id_full
        )

        # Note: Min-Max normalization was removed to preserve true scale.
        return full_sinogram_pred
