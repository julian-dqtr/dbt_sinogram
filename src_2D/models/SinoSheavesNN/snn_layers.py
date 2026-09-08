import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class LearnedRestrictionMap(nn.Module):
    """
    Learns residual corrections to the fixed SO(2) restriction maps.

    The fixed geometric SO(2) maps encode the angular difference between views.
    This module predicts a small (delta_cos, delta_sin) correction conditioned on
    the source and destination node feature statistics (mean, std over detectors),
    so the restriction maps can adapt to signal content while staying initialized
    at the physically-motivated SO(2) values.

    Input:
        edge_attr:  [E, 4] fixed SO(2) maps [cos, -sin, sin, cos]
        x_i_stats:  [E, 2] destination node stats (mean, std)
        x_j_stats:  [E, 2] source node stats (mean, std)
    Output:
        refined_attr: [E, 4] learned restriction maps
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        # Input: 4 (fixed SO2) + 2 (src stats) + 2 (dst stats) = 8
        self.mlp = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # predicts (delta_cos, delta_sin)
        )
        # Initialize output layer to near-zero so initial maps = fixed SO(2)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, edge_attr, x_i_stats, x_j_stats):
        # edge_attr: [E, 4], x_i_stats: [E, 2], x_j_stats: [E, 2]
        inp = torch.cat([edge_attr, x_i_stats, x_j_stats], dim=1)  # [E, 8]
        delta = self.mlp(inp)  # [E, 2] → (delta_cos, delta_sin)

        cos_base = edge_attr[:, 0]  # original cos(Δθ)
        sin_base = edge_attr[:, 2]  # original sin(Δθ)

        cos_new = cos_base + delta[:, 0]
        sin_new = sin_base + delta[:, 1]

        # Rebuild the [cos, -sin, sin, cos] structure
        refined = torch.stack([cos_new, -sin_new, sin_new, cos_new], dim=1)
        return refined


class SheafConv1D(MessagePassing):
    """
    A Custom Sheaf Convolution layer for sequence data (1D detector arrays).
    Assumes node features have dimension [N, in_channels * 2, L],
    representing `in_channels` independent 2D stalks along a sequence of length L.
    Integrates Heat Kernel edge weights and uses learned SO(2) restriction maps.
    """
    def __init__(self, in_channels, out_channels, aggr='mean'):
        super().__init__(aggr=aggr, node_dim=0)
        # in_channels and out_channels refer to the number of 2D stalks (F)
        # We use a 1D Convolution instead of an MLP to update node states.
        # This preserves and processes the spatial information.
        self.conv_update = nn.Conv1d(in_channels * 4, out_channels * 2, kernel_size=1)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Learned restriction map module
        self.learned_restriction = LearnedRestrictionMap(hidden_dim=16)

    def forward(self, x, edge_index, edge_attr, edge_weight):
        # x: [num_nodes, in_channels * 2, L]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, 4] representing flattened 2x2 matrices
        # edge_weight: [num_edges]

        # Compute per-node summary statistics for the learned restriction maps
        # x shape: [N, C, L] → stats shape: [N, 2] (mean, std over L)
        node_mean = x.mean(dim=(1, 2))  # [N]
        node_std = x.std(dim=(1, 2))    # [N]
        node_stats = torch.stack([node_mean, node_std], dim=1)  # [N, 2]

        # Gather stats for source (j) and destination (i) nodes
        x_i_stats = node_stats[edge_index[1]]  # [E, 2]
        x_j_stats = node_stats[edge_index[0]]  # [E, 2]

        # Compute learned restriction maps
        learned_edge_attr = self.learned_restriction(edge_attr, x_i_stats, x_j_stats)

        # Start message passing with learned maps
        out = self.propagate(edge_index, x=x, edge_attr=learned_edge_attr, edge_weight=edge_weight)

        # out has shape [num_nodes, in_channels * 2, L] after aggregation
        # Concatenate current features with aggregated messages along the channel dim
        cat = torch.cat([x, out], dim=1)

        return self.conv_update(cat)

    def message(self, x_j, edge_attr, edge_weight):
        # x_j is the features of the source nodes, shape: [num_edges, in_channels * 2, L]
        num_edges = x_j.size(0)
        L = x_j.size(2)

        # Reshape to apply the 2x2 rotation to each 2D stalk
        # x_j: [num_edges, in_channels, 2, L]
        x_j_reshaped = x_j.view(num_edges, self.in_channels, 2, L)

        # edge_attr: [num_edges, 1, 2, 2] so it broadcasts across in_channels
        R = edge_attr.view(num_edges, 1, 2, 2)

        # Matrix multiplication: R (2x2) @ x_j_reshaped (2xL)
        msg = torch.matmul(R, x_j_reshaped)

        # Flatten back to [num_edges, in_channels * 2, L]
        msg = msg.view(num_edges, self.in_channels * 2, L)

        # Weight the message by the Heat Kernel edge weight
        msg = msg * edge_weight.view(-1, 1, 1)

        return msg


class SinoSheafBlock(nn.Module):
    """
    Hybrid Convolutional Sheaf Block with Instance Normalization.
    1. Spatial processing (Conv1d + InstanceNorm1d)
    2. Angular aggregation (SheafConv1D + InstanceNorm1d)
    3. Spatial refinement (Conv1d) with residual connection
    """
    def __init__(self, channels):
        # channels is the number of 2D stalks. So actual feature maps = channels * 2
        super().__init__()
        self.conv1 = nn.Conv1d(channels * 2, channels * 2, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm1d(channels * 2)
        self.sheaf_conv = SheafConv1D(in_channels=channels, out_channels=channels)
        self.norm2 = nn.InstanceNorm1d(channels * 2)
        self.conv2 = nn.Conv1d(channels * 2, channels * 2, kernel_size=3, padding=1)

    def forward(self, x, edge_index, edge_attr, edge_weight):
        # x: [num_nodes, channels * 2, L]
        res = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.sheaf_conv(x, edge_index, edge_attr, edge_weight)
        x = F.relu(self.norm2(x))
        x = self.conv2(x)
        return F.relu(x + res)
