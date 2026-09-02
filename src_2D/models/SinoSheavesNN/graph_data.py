import torch
import numpy as np
from torch_geometric.data import Data

def build_sinogram_graph_topology(geom, grid_res=20, object_radius=50.0):
    """
    Builds the graph adjacency matrix for a sinogram based on its geometry.
    Nodes: (view, detector) combinations.
    Edges: Connect adjacent views if they belong to the same physical point's sinusoidal path.
    """
    num_views = geom.num_views
    num_det = geom.det_col_count
    num_nodes = num_views * num_det

    edges = []
    
    # 1. Local horizontal edges (same view, adjacent detectors)
    # This helps local smoothness across the detector array
    for v in range(num_views):
        for d in range(num_det - 1):
            n1 = v * num_det + d
            n2 = v * num_det + d + 1
            edges.append((n1, n2))
            edges.append((n2, n1))
            
    # 2. Sinusoidal edges across views
    # Trace calibration points to connect nodes that lie on the same trajectory
    xs = np.linspace(-object_radius, object_radius, grid_res)
    ys = np.linspace(-object_radius, object_radius, grid_res)
    
    for x in xs:
        for y in ys:
            prev_node = None
            for v, angle in enumerate(geom.angles):
                src_x = geom.vectors[v, 0]
                src_y = geom.vectors[v, 1]
                
                # Intersect ray from source to (x,y) with detector plane (y = -det_radius)
                if y == src_y: 
                    continue
                t = (-geom.det_radius - src_y) / (y - src_y)
                x_det = src_x + t * (x - src_x)
                
                # Map physical x_det to detector index
                d_float = (x_det / geom.det_pixel_size) + (num_det / 2.0) - 0.5
                d = int(round(d_float))
                
                if 0 <= d < num_det:
                    curr_node = v * num_det + d
                    if prev_node is not None and curr_node != prev_node:
                        edges.append((prev_node, curr_node))
                        edges.append((curr_node, prev_node))
                    prev_node = curr_node
                else:
                    prev_node = None
                    
    # Remove duplicates
    edges = list(set(edges))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return edge_index

def build_so2_restriction_maps(edge_index, geom):
    """
    Constructs SO(2) rotation matrices for each edge based on the angle difference.
    Returns edge_attr of shape [num_edges, 4] representing flattened 2x2 matrices.
    """
    num_det = geom.det_col_count
    
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    
    # Extract the view indices for the source and destination nodes
    src_views = src_nodes // num_det
    dst_views = dst_nodes // num_det
    
    # Calculate the angle difference for each edge
    # angles shape: [num_views]
    angles_tensor = torch.tensor(geom.angles, dtype=torch.float32)
    src_angles = angles_tensor[src_views]
    dst_angles = angles_tensor[dst_views]
    
    delta_theta = dst_angles - src_angles
    
    # SO(2) matrix: [[cos, -sin], [sin, cos]]
    cos_t = torch.cos(delta_theta)
    sin_t = torch.sin(delta_theta)
    
    # Flattened representation: [cos, -sin, sin, cos]
    edge_attr = torch.stack([cos_t, -sin_t, sin_t, cos_t], dim=1)
    
    return edge_attr

_cached_edge_index = None
_cached_edge_attr = None

def create_sinogram_data(sinogram_tensor, geom, grid_res=20, object_radius=50.0):
    """
    Converts a standard 2D sinogram tensor [num_views, num_detectors]
    into a PyTorch Geometric Data object with SO(2) restriction maps.
    The graph topology is cached after the first computation.
    """
    global _cached_edge_index, _cached_edge_attr
    
    assert sinogram_tensor.shape == (geom.num_views, geom.det_col_count), "Sinogram shape mismatch"
    
    # Flatten sinogram to create node features [num_nodes, 1]
    x = sinogram_tensor.view(-1, 1).float()
    
    if _cached_edge_index is None or _cached_edge_attr is None:
        # Build graph topology
        _cached_edge_index = build_sinogram_graph_topology(geom, grid_res, object_radius)
        
        # Build restriction maps
        _cached_edge_attr = build_so2_restriction_maps(_cached_edge_index, geom)
        
    data = Data(x=x, edge_index=_cached_edge_index, edge_attr=_cached_edge_attr)
    return data
