import nbformat
import sys

def add_snn_cells(notebook_path):
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
            
        md_cell = nbformat.v4.new_markdown_cell(source="""## SinoSheavesNN Analysis

In this section, we evaluate the Physics-Informed **Sheaf Neural Network (SNN)** architecture. 
The SNN natively respects the rotational symmetries of the Radon transform by passing features through $SO(2)$ restriction maps, while the Helgason-Ludwig moments constrain the mass and center of mass of the predictions.

Below, we load the model, perform a forward pass, compute the physical consistency metrics, and visualize the predicted sinogram.""")

        code_cell_1 = nbformat.v4.new_code_cell(source="""import torch
import matplotlib.pyplot as plt
import numpy as np
from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.models.SinoSheavesNN.snn_model import SinoSheafNet
from src_2D.models.SinoSheavesNN.graph_data import create_sinogram_data
from src_2D.models.SinoSheavesNN.physics_loss import HelgasonLudwigLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = DBTGeometryConfig()
geom = DBTGeometry(
    angles=config.full_angles,
    src_radius=config.src_radius_mm,
    det_radius=config.det_radius_mm,
    det_col_count=config.det_col_count,
    det_pixel_size=config.det_pixel_size_mm,
)

# 1. Load Dataset
dataset = SinogramCompletionDataset(n_samples=5, geometry_config=config, device=str(device), is_validation_or_test=True)

# 2. Initialize Model
# NOTE: Update the hyperparameters based on your best Optuna trial!
model_path = project_root / "outputs" / "2d" / "best_snn_model.pth"
model = SinoSheafNet(num_stalks=32, num_layers=6).to(device)

if model_path.exists():
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded trained SinoSheavesNN model.")
else:
    print("Warning: Pre-trained weights not found. Using untrained model for demonstration.\\nTrain the model using run_training.py to see accurate results.")

model.eval()""")

        code_cell_2 = nbformat.v4.new_code_cell(source="""# 3. Qualitative Evaluation & Physics Verification
inc_sino, full_sino, phantom = dataset[0]

inc_sino_tensor = inc_sino[0].to(device)
target_sino_tensor = full_sino[0].to(device)

# Build the Graph Data dynamically
graph_data = create_sinogram_data(inc_sino_tensor, geom).to(device)

with torch.no_grad():
    out = model(graph_data)
    pred_sino = out.view(geom.num_views, geom.det_col_count)
    
# Physics Metrics
hl_loss = HelgasonLudwigLoss(geom).to(device)
m0_loss, m1_loss = hl_loss(pred_sino)

from skimage.metrics import structural_similarity as ssim
target_np = target_sino_tensor.cpu().numpy()
pred_np = pred_sino.cpu().numpy()
d_range = max(target_np.max() - target_np.min(), 1.0)
snn_ssim = ssim(target_np, pred_np, data_range=d_range)

print("--- SinoSheavesNN Performance ---")
print(f"SSIM: {snn_ssim:.4f}")
print(f"Helgason-Ludwig 0th Moment (Mass Variance): {m0_loss.item():.4f}")
print(f"Helgason-Ludwig 1st Moment (Center of Mass Residual): {m1_loss.item():.4f}")

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

im0 = axes[0].imshow(inc_sino_tensor.cpu().numpy(), cmap='gray', aspect='auto')
axes[0].set_title("Input (Incomplete Sinogram)")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(pred_np, cmap='gray', aspect='auto')
axes[1].set_title("SinoSheavesNN Prediction")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(target_np, cmap='gray', aspect='auto')
axes[2].set_title("Ground Truth (Full Sinogram)")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()""")

        nb.cells.extend([md_cell, code_cell_1, code_cell_2])
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
            
        print("Successfully added SinoSheavesNN cells to the notebook.")
    except Exception as e:
        print(f"Error updating notebook: {e}")

if __name__ == "__main__":
    add_snn_cells("/home/jdq/Master_Thesis/notebooks/model_analysis.ipynb")
