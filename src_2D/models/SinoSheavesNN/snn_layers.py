import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class SheafConv(MessagePassing):
    """
    A custom Sheaf Convolution layer for 2D stalks with SO(2) restriction maps.
    Assumes that the node features have dimension (F * 2), representing F independent 2D stalks.
    """
    def __init__(self, in_channels, out_channels, aggr='mean'):
        super(SheafConv, self).__init__(aggr=aggr)
        # in_channels and out_channels refer to the number of 2D stalks (F)
        # So the actual tensor dimension is in_channels * 2
        
        # Linear transformations for the node itself and the aggregated messages
        # We process the F*2 vector directly
        self.lin_node = nn.Linear(in_channels * 2, out_channels * 2)
        self.lin_msg = nn.Linear(in_channels * 2, out_channels * 2)
        
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, in_channels * 2]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, 4] representing flattened 2x2 matrices
        
        # Start message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # out has shape [num_nodes, in_channels * 2] after aggregation
        # Now apply the update rule
        node_update = self.lin_node(x)
        msg_update = self.lin_msg(out)
        
        return node_update + msg_update

    def message(self, x_j, edge_attr):
        # x_j is the features of the source nodes, shape: [num_edges, in_channels * 2]
        # edge_attr is the SO(2) matrices, shape: [num_edges, 4]
        num_edges = x_j.size(0)
        
        # Reshape to apply the 2x2 rotation to each 2D stalk
        # x_j: [num_edges, in_channels, 2, 1]
        x_j_reshaped = x_j.view(num_edges, self.in_channels, 2, 1)
        
        # edge_attr: [num_edges, 1, 2, 2] so it broadcasts across in_channels
        R = edge_attr.view(num_edges, 1, 2, 2)
        
        # Matrix multiplication: R * x_j
        # R is [num_edges, 1, 2, 2], x_j is [num_edges, in_channels, 2, 1]
        # Result msg: [num_edges, in_channels, 2, 1]
        msg = torch.matmul(R, x_j_reshaped)
        
        # Flatten back to [num_edges, in_channels * 2]
        msg = msg.view(num_edges, self.in_channels * 2)
        
        return msg
