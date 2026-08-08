from __future__ import annotations

import torch
from torch import nn


class UNet25DWrapper(nn.Module):
    """Wraps a 2D U-Net to process 5D tensors (B, C, Views, Det_Y, Det_X) slice by slice along Det_Y."""

    def __init__(self, unet: nn.Module) -> None:
        super().__init__()
        self.unet = unet

    def forward(self, incomplete_sinogram: torch.Tensor) -> torch.Tensor:
        if incomplete_sinogram.dim() == 5:
            B, C, V, Y, X = incomplete_sinogram.shape
            
            # Reshape to [B * Y, C, V, X] for the 2D UNet
            unet_input_2d = incomplete_sinogram.permute(0, 3, 1, 2, 4).reshape(B * Y, C, V, X)
            
            # If not in training mode (inference), we process in chunks to avoid OOM
            if not self.training and unet_input_2d.shape[0] > 16:
                chunk_size = 16
                refined_chunks = []
                for i in range(0, unet_input_2d.shape[0], chunk_size):
                    chunk = unet_input_2d[i:i + chunk_size]
                    refined_chunks.append(self.unet(chunk))
                refined_output_2d = torch.cat(refined_chunks, dim=0)
            else:
                refined_output_2d = self.unet(unet_input_2d)
                
            # Reshape back to [B, Y, C, V, X] and permute to [B, C, V, Y, X]
            refined_output = refined_output_2d.reshape(B, Y, refined_output_2d.shape[1], V, X).permute(0, 2, 3, 1, 4)
        else:
            refined_output = self.unet(incomplete_sinogram)

        return refined_output
