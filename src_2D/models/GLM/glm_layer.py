import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor
from torch_geometric.utils import add_remaining_self_loops, scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes

def gcn_norm(
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    fill_value = 2. if improved else 1.

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


class GLM_Module(MessagePassing):
    """
    GLM Module ported from the official GLM repository.
    Combines 1D spatial convolution over the detector dimension
    and Message Passing over the angular dimension.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        padding: int = 3,
        normalize: bool = True,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        aggr: str = 'add',
        **kwargs,
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, 1, padding)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, 1, padding)
        
        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        nn.init.constant_(self.conv2.bias, 0.1)
        self._cached_edge_index = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        # x is expected to be of shape [n_nodes, in_channels, n_pixels]
        n_nodes = x.size(0)
        n_pixels = x.size(-1)

        if self.normalize:
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(
                    edge_index, edge_weight, n_nodes,
                    self.improved, self.add_self_loops, self.flow, x.dtype)
                if self.cached:
                    self._cached_edge_index = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = cache[0], cache[1]

        x = self.relu(self.conv1(x))

        # Flatten features for propagation: MessagePassing expects [N, F]
        x_flat = x.view(n_nodes, self.out_channels * n_pixels)
        
        out_flat = self.propagate(edge_index, x=x_flat, edge_weight=edge_weight)
        
        out = out_flat.view(n_nodes, self.out_channels, n_pixels)

        return out + self.conv2(out)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
