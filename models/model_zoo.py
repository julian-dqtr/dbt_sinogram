from __future__ import annotations

from torch import nn

from conf.geometry import DBTGeometryConfig
from models.Unet import SinogramUNet
from models.interpolator import SinusoidalViewInterpolator
from models.pipeline import SinogramCompletionPipeline


def build_models(device) -> dict[str, nn.Module]:
    """Create distinct baselines for comparison."""
    interpolator = SinusoidalViewInterpolator(angles_rad=DBTGeometryConfig().full_angles).to(device)
    unet = SinogramUNet(in_channels=2, out_channels=1).to(device)

    return {
        "interpolator": SinogramCompletionPipeline(interpolator, refiner=None).to(device),
        "unet": SinogramCompletionPipeline(interpolator, unet).to(device),
    }
