import torch
import numpy as np
from torch_geometric.data import Data

def build_view_graph_topology(geom, sigma_deg=10.0, graph_type="knn", k=5):
    """
    Builds the graph adjacency matrix for a sinogram where each VIEW is a node.

    Args:
        geom: Geometry object with .num_views and .angles attributes.
        sigma_deg: Standard deviation for the Heat Kernel edge weights (in degrees).
        graph_type: "knn" for k-nearest angular neighbors (default), "full" for fully connected.
        k: Number of nearest neighbors per node (only used when graph_type="knn").

    Returns:
        edge_index: [2, num_edges] tensor.
        edge_weight: [num_edges] Heat Kernel weights.
        edge_attr: [num_edges, 4] flattened SO(2) restriction maps.
    """
    num_views = geom.num_views
    angles_tensor = torch.tensor(geom.angles, dtype=torch.float32)

    if graph_type == "full":
        # Fully connected graph (legacy behavior)
        edges = []
        for i in range(num_views):
            for j in range(num_views):
                edges.append((i, j))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    elif graph_type == "knn":
        # k-nearest angular neighbors + self-loops
        # Compute pairwise angular distances
        angle_diffs = torch.abs(angles_tensor.unsqueeze(0) - angles_tensor.unsqueeze(1))  # [V, V]

        # For each node, find the k nearest neighbors (excluding self initially)
        # We set diagonal to inf so self is not selected as a neighbor
        angle_diffs_no_self = angle_diffs.clone()
        angle_diffs_no_self.fill_diagonal_(float('inf'))

        # Get k nearest neighbors for each node
        _, knn_indices = torch.topk(angle_diffs_no_self, k=min(k, num_views - 1), dim=1, largest=False)

        edges = []
        for i in range(num_views):
            # Self-loop
            edges.append((i, i))
            # k-NN edges
            for j_idx in range(knn_indices.size(1)):
                j = knn_indices[i, j_idx].item()
                edges.append((i, j))

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}. Use 'knn' or 'full'.")

    src_angles = angles_tensor[edge_index[0]]
    dst_angles = angles_tensor[edge_index[1]]

    delta_theta = dst_angles - src_angles

    # Heat Kernel weights
    sigma_rad = sigma_deg * np.pi / 180.0
    edge_weight = torch.exp(-(delta_theta ** 2) / (sigma_rad ** 2))

    # SO(2) restriction maps
    cos_t = torch.cos(delta_theta)
    sin_t = torch.sin(delta_theta)

    # Flattened representation: [cos, -sin, sin, cos]
    edge_attr = torch.stack([cos_t, -sin_t, sin_t, cos_t], dim=1)

    return edge_index, edge_weight, edge_attr

_cached_edge_index = None
_cached_edge_weight = None
_cached_edge_attr = None
_cached_graph_type = None
_cached_k = None
_cached_sigma_deg = None

def create_sinogram_data(sinogram_tensor, geom, sigma_deg=10.0, graph_type="knn", k=5):
    """
    Converts a standard 2D sinogram tensor [num_views, num_detectors]
    into a PyTorch Geometric Data object where each node is a view.
    """
    global _cached_edge_index, _cached_edge_weight, _cached_edge_attr, _cached_graph_type, _cached_k, _cached_sigma_deg

    assert sinogram_tensor.shape == (geom.num_views, geom.det_col_count), "Sinogram shape mismatch"

    # Infer acquired mask: if a view is entirely 0, it wasn't acquired.
    # sinogram_tensor shape: [180, 128]
    view_sums = torch.sum(torch.abs(sinogram_tensor), dim=1)
    is_acquired = (view_sums > 1e-6).float().unsqueeze(-1) # Shape: [180, 1]

    # Node features: concatenate the 128 detector values with the is_acquired flag
    # Shape: [180, 129]
    x = torch.cat([sinogram_tensor.float(), is_acquired], dim=1)

    # Invalidate cache if graph parameters changed
    if (_cached_edge_index is None or _cached_edge_attr is None
            or _cached_graph_type != graph_type or _cached_k != k
            or _cached_sigma_deg != sigma_deg):
        _cached_edge_index, _cached_edge_weight, _cached_edge_attr = build_view_graph_topology(
            geom, sigma_deg, graph_type=graph_type, k=k
        )
        _cached_graph_type = graph_type
        _cached_k = k
        _cached_sigma_deg = sigma_deg

    data = Data(x=x, edge_index=_cached_edge_index, edge_attr=_cached_edge_attr, edge_weight=_cached_edge_weight)
    return data
