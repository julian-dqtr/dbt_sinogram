import torch
import torch.nn as nn
import torch.nn.functional as F
from src_2D.models.SinoSheavesNN.snn_layers import SinoSheafBlock

class SinoSheafNet(nn.Module):
    """
    Convolutional Sheaf Neural Network for Sinogram completion (GLM-like architecture).
    Nodes: Views (180).
    Features: 1D projection values (128) + mask (128) -> mapped to F 2D stalks.
    Unlike previous versions, this preserves the spatial (detector) dimension throughout the network.

    Architecture improvements:
    - InstanceNorm1d in SinoSheafBlocks for training stability
    - Learned SO(2) restriction maps (residual corrections on fixed geometric maps)
    - 3-layer decoder for more expressivity
    """
    def __init__(self, num_stalks=16, num_layers=4):
        super(SinoSheafNet, self).__init__()

        # Lift 2 channels (signal + mask) to (num_stalks * 2) channels using Conv1d
        self.encoder = nn.Conv1d(2, num_stalks * 2, kernel_size=3, padding=1)

        # Hybrid Convolutional Sheaf Blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(SinoSheafBlock(channels=num_stalks))

        # Deeper decoder: 3-layer projection from stalk space to sinogram space
        self.decoder = nn.Sequential(
            nn.Conv1d(num_stalks * 2, num_stalks, kernel_size=3, padding=1),
            nn.InstanceNorm1d(num_stalks),
            nn.ReLU(),
            nn.Conv1d(num_stalks, num_stalks // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(num_stalks // 2, 1, kernel_size=3, padding=1),
        )

    def forward(self, data):
        # data.x shape: [N, 129] in original graph.
        # But wait! We need signal and mask separated and reshaped.
        N = data.x.size(0)

        # Split signal and mask
        # Signal is the first 128 elements. Mask is the 129th element.
        signal = data.x[:, :128].unsqueeze(1) # [N, 1, 128]
        mask_val = data.x[:, 128:] # [N, 1]

        # We expand the single mask value to span all 128 detector pixels
        # so that it matches the spatial dimension.
        mask = mask_val.expand(N, 128).unsqueeze(1) # [N, 1, 128]

        # Re-concatenate along channel dimension
        x = torch.cat([signal, mask], dim=1) # [N, 2, 128]

        # Encoding
        x = self.encoder(x)
        x = F.relu(x)

        # SinoSheaf Blocks
        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr, data.edge_weight)

        # Decoding (3-layer decoder)
        out = self.decoder(x) # [N, 1, 128]

        out = out.view(N, 128)

        return out
