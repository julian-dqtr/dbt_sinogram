import torch
import torch.nn as nn
import torch.nn.functional as F
from src_2D.models.SinoSheavesNN.snn_layers import SheafConv

class SinoSheafNet(nn.Module):
    """
    Sheaf Neural Network for Sinogram completion/processing.
    Nodes: Sinogram pixels.
    Features: 1D projection values -> mapped to F 2D stalks.
    """
    def __init__(self, num_stalks=16, num_layers=4):
        super(SinoSheafNet, self).__init__()
        
        # Lift 1D input to (num_stalks * 2) dimensions
        self.encoder = nn.Linear(1, num_stalks * 2)
        
        # Sheaf Convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SheafConv(in_channels=num_stalks, out_channels=num_stalks))
            
        # Decoder back to 1D output
        self.decoder = nn.Linear(num_stalks * 2, 1)
        
    def forward(self, data):
        # x shape: [N, 1]
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Lifting
        x = self.encoder(x)
        x = F.relu(x)
        
        # Sheaf Message Passing
        for conv in self.convs:
            x_new = conv(x, edge_index, edge_attr)
            x = F.relu(x_new)
            
        # Decoding
        out = self.decoder(x)
        return out
