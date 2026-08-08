from __future__ import annotations

import torch
from torch import nn


class SinogramCompletionPipeline(nn.Module):
    """Common interface for comparing a non-learned interpolator and a learned U-Net."""

    def __init__(self, baseline_model: nn.Module, refiner: nn.Module | None = None) -> None:
        super().__init__()
        self.baseline_model = baseline_model
        self.refiner = refiner

    def forward(self, incomplete_sinogram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        baseline_output = self.baseline_model(incomplete_sinogram)

        if self.refiner is None:
            return baseline_output, baseline_output

        refiner_input = torch.cat([incomplete_sinogram, baseline_output], dim=1)
        refined_output = self.refiner(refiner_input)
        return baseline_output, refined_output
