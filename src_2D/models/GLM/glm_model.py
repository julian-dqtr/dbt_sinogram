import torch
import torch.nn as nn
import torch.nn.functional as F

from src_2D.models.GLM.glm_layer import GLM_Module

class GLMNet(nn.Module):
    """
    Graph Neural Network for Line Manifolds (GLM) adapted to the DBT geometry.
    This architecture matches the settings described in the GLM paper for 'GLM-24' 
    (3 GLM layers, 24 channels, kernel size 7).
    """
    def __init__(self, num_channels=24, num_layers=3, kernel_size=7):
        super(GLMNet, self).__init__()
        
        # Initial projection: lifts 2 channels (signal + mask) to num_channels
        self.encoder = nn.Conv1d(2, num_channels, kernel_size=3, padding=1)
        
        # GLM modules stack
        padding = kernel_size // 2
        self.glm_layers = nn.ModuleList([
            GLM_Module(
                in_channels=num_channels, 
                out_channels=num_channels, 
                kernel_size=kernel_size, 
                padding=padding
            ) for _ in range(num_layers)
        ])
        
        # Final projection: map back to 1 channel (the predicted sinogram)
        self.decoder = nn.Conv1d(num_channels, 1, kernel_size=3, padding=1)

    def forward(self, data):
        # data.x shape: [N, 129] 
        N = data.x.size(0)
        
        # Split signal and mask
        signal = data.x[:, :128].unsqueeze(1) # [N, 1, 128]
        mask_val = data.x[:, 128:]            # [N, 1]
        
        # Expand the single mask value to span all 128 detector pixels
        mask = mask_val.expand(N, 128).unsqueeze(1) # [N, 1, 128]
        
        # Re-concatenate along channel dimension
        x = torch.cat([signal, mask], dim=1) # [N, 2, 128]
        
        # 1. Encoding
        x = F.relu(self.encoder(x))
        
        # 2. GLM Layers (Graph Convolution + Spatial Convolution)
        for layer in self.glm_layers:
            x = layer(x, data.edge_index, data.edge_weight)
            
        # 3. Decoding
        out = self.decoder(x) # [N, 1, 128]
        
        # Flatten back to original view shape
        out = out.view(N, 128)
        
        return out
