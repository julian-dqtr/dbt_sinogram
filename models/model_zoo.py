from __future__ import annotations

from torch import nn

from dbt_sinogram_completion.conf.geometry import DBTGeometryConfig
from dbt_sinogram_completion.models.Unet import SinogramUNet
from dbt_sinogram_completion.models.interpolator import SinusoidalViewInterpolator
from dbt_sinogram_completion.models.pipeline import SinogramCompletionPipeline


def build_models(device) -> dict[str, nn.Module]:
    """Create distinct baselines for comparison."""
    interpolator = SinusoidalViewInterpolator(angles_rad=DBTGeometryConfig().full_angles).to(device)
    unet = SinogramUNet(in_channels=2, out_channels=1).to(device)

    return {
        "interpolator": SinogramCompletionPipeline(interpolator, refiner=None).to(device),
        "unet": SinogramCompletionPipeline(interpolator, unet).to(device),
    }
