import numpy as np
import torch
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.models.SinoSheavesNN.graph_data import create_sinogram_data

def test_create_sinogram_data():
    num_views = 10
    num_det = 50
    angles = np.linspace(-np.pi/6, np.pi/6, num_views)
    geom = DBTGeometry(angles, src_radius=500, det_radius=100, det_col_count=num_det, det_pixel_size=2.0)
    
    # Create a dummy sinogram
    sinogram = torch.randn(num_views, num_det)
    
    # Generate the graph data
    data = create_sinogram_data(sinogram, geom, grid_res=20, object_radius=50.0)
    
    # Assertions
    assert data.x.shape == (num_views * num_det, 1), f"Expected x shape {(num_views * num_det, 1)}, got {data.x.shape}"
    assert data.edge_index.shape[0] == 2, "edge_index should have shape [2, E]"
    assert data.edge_attr.shape[1] == 4, "edge_attr should have shape [E, 4] for SO(2) matrices"
    assert data.edge_attr.shape[0] == data.edge_index.shape[1], "Mismatch between number of edges and edge attributes"
    
    # Check if SO(2) constraints hold (det(R) == 1)
    # R = [[a, b], [c, d]] -> det = a*d - b*c
    # In edge_attr: [cos, -sin, sin, cos] -> a=attr[0], b=attr[1], c=attr[2], d=attr[3]
    dets = data.edge_attr[:, 0] * data.edge_attr[:, 3] - data.edge_attr[:, 1] * data.edge_attr[:, 2]
    assert torch.allclose(dets, torch.ones_like(dets), atol=1e-5), "SO(2) determinant is not 1"
    
    print("All tests passed for create_sinogram_data!")

if __name__ == "__main__":
    test_create_sinogram_data()
