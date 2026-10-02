import torch
import numpy as np
from torch_geometric.data import Data

def build_glm_view_graph_topology(geom):
    """
    Builds the graph adjacency matrix for a sinogram where each VIEW is a node,
    connecting only immediate neighbors with cosine similarity weights, as described
    in the GLM paper.
    
    Args:
        geom: Geometry object with .num_views and .angles attributes.
        
    Returns:
        edge_index: [2, num_edges] tensor.
        edge_weight: [num_edges] weights based on cosine similarity of angles.
    """
    num_views = geom.num_views
    angles_rad = np.array(geom.angles) * np.pi / 180.0
    
    edges = []
    
    for i in range(num_views):
        # The GLM paper also uses self loops, but GCN normalization adds them internally if requested.
        # We will add them explicitly here to match typical PyG behavior when not relying on on-the-fly normalization,
        # or we can rely on gcn_norm in GLM_Module. Let's not add self loops here as gcn_norm handles it.
        if i > 0:
            edges.append((i, i - 1))
        if i < num_views - 1:
            edges.append((i, i + 1))
            
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Calculate cosine weighting
    src_angles = angles_rad[edge_index[0].numpy()]
    dst_angles = angles_rad[edge_index[1].numpy()]
    
    delta_theta = dst_angles - src_angles
    edge_weight = torch.tensor(np.cos(delta_theta), dtype=torch.float32)
    
    return edge_index, edge_weight

_cached_edge_index = None
_cached_edge_weight = None

def create_glm_sinogram_data(sinogram_tensor, geom):
    """
    Converts a standard 2D sinogram tensor [num_views, num_detectors]
    into a PyTorch Geometric Data object where each node is a view.
    """
    global _cached_edge_index, _cached_edge_weight
    
    assert sinogram_tensor.shape == (geom.num_views, geom.det_col_count), "Sinogram shape mismatch"
    
    view_sums = torch.sum(torch.abs(sinogram_tensor), dim=1)
    is_acquired = (view_sums > 1e-6).float().unsqueeze(-1)
    
    x = torch.cat([sinogram_tensor.float(), is_acquired], dim=1)
    
    if _cached_edge_index is None:
        _cached_edge_index, _cached_edge_weight = build_glm_view_graph_topology(geom)
        
    data = Data(x=x, edge_index=_cached_edge_index, edge_weight=_cached_edge_weight)
    return data
