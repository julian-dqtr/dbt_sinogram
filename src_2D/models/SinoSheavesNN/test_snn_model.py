import torch
import numpy as np
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.models.SinoSheavesNN.graph_data import create_sinogram_data
from src_2D.models.SinoSheavesNN.snn_model import SinoSheafNet

def test_snn_forward():
    # 1. Create a dummy graph
    num_views = 10
    num_det = 50
    angles = np.linspace(-np.pi/6, np.pi/6, num_views)
    geom = DBTGeometry(angles, src_radius=500, det_radius=100, det_col_count=num_det, det_pixel_size=2.0)
    
    sinogram = torch.randn(num_views, num_det)
    data = create_sinogram_data(sinogram, geom, grid_res=20, object_radius=50.0)
    
    # 2. Initialize Model
    num_stalks = 8
    model = SinoSheafNet(num_stalks=num_stalks, num_layers=3)
    
    # 3. Forward Pass
    try:
        out = model(data)
    except Exception as e:
        assert False, f"Forward pass failed with error: {e}"
        
    # 4. Check shapes
    num_nodes = num_views * num_det
    assert out.shape == (num_nodes, 1), f"Expected shape {(num_nodes, 1)}, got {out.shape}"
    
    # 5. Check gradients (ensure no disconnected graphs block backprop)
    out.sum().backward()
    
    # Check if parameters got gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive gradients."
        
    print("All tests passed for SinoSheafNet forward and backward passes!")

if __name__ == "__main__":
    test_snn_forward()
