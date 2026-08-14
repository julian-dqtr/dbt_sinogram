import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
import argparse
from tqdm import tqdm

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.models.SinoSheavesNN.snn_model import SinoSheafNet
from src_2D.models.SinoSheavesNN.physics_loss import AnnealedLoss
from src_2D.models.SinoSheavesNN.graph_data import create_sinogram_data

def train_epoch(model, dataloader, optimizer, loss_fn, geom, epoch, device):
    model.train()
    total_loss = 0.0
    total_data_loss = 0.0
    total_m0_loss = 0.0
    total_m1_loss = 0.0
    
    for batch_idx, (incomplete_sino, full_sino, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        # Batch size is usually 1 for graph data unless we batch PyG Data objects.
        # For simplicity in this first implementation, we process the first item in the batch.
        # (Standard DataLoader gives shape [B, 1, Views, Det])
        inc_sino = incomplete_sino[0, 0].to(device)
        target_sino = full_sino[0, 0].to(device)
        
        # 1. Graph Construction
        graph_data = create_sinogram_data(inc_sino, geom).to(device)
        
        # 2. Forward Pass
        optimizer.zero_grad()
        out = model(graph_data)
        
        # Reshape output from [num_nodes, 1] back to [num_views, num_detectors]
        pred_sino = out.view(geom.num_views, geom.det_col_count)
        
        # 3. Physics-Informed Annealed Loss
        loss, data_loss, m0_loss, m1_loss = loss_fn(pred_sino, target_sino, epoch)
        
        # 4. Backward & Optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_data_loss += data_loss.item()
        total_m0_loss += m0_loss.item()
        total_m1_loss += m1_loss.item()
        
    num_batches = len(dataloader)
    return total_loss/num_batches, total_data_loss/num_batches, total_m0_loss/num_batches, total_m1_loss/num_batches

def main():
    parser = argparse.ArgumentParser(description="Train SinoSheavesNN")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1) # PyG batching requires custom collate, keep 1 for now
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default="SinoSheavesNN")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Initialize wandb
    # wandb.init(project=args.wandb_project, config=vars(args))

    # Geometry Setup
    config = DBTGeometryConfig()
    geom = DBTGeometry(
        angles=config.full_angles,
        src_radius=config.src_radius_mm,
        det_radius=config.det_radius_mm,
        det_col_count=config.det_col_count,
        det_pixel_size=config.det_pixel_size_mm,
    )
    
    # Dataset
    dataset = SinogramCompletionDataset(
        n_samples=args.n_samples,
        geometry_config=config,
        device=args.device
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Model & Loss
    model = SinoSheafNet(num_stalks=16, num_layers=4).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = AnnealedLoss(geom, lambda_m0=0.1, lambda_m1=0.1, anneal_epochs=50).to(args.device)
    
    print(f"Starting training on {args.device} for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        loss, d_loss, m0_loss, m1_loss = train_epoch(model, dataloader, optimizer, loss_fn, geom, epoch, args.device)
        
        print(f"Epoch {epoch}: Total={loss:.4f} | Data={d_loss:.4f} | M0={m0_loss:.4f} | M1={m1_loss:.4f}")
        
        # Uncomment when wandb is logged in
        # wandb.log({
        #     "Total Loss": loss,
        #     "Data Loss (MSE)": d_loss,
        #     "M0 Variance Loss": m0_loss,
        #     "M1 Residual Loss": m1_loss,
        #     "Epoch": epoch
        # })
        
    print("Training completed.")
    
if __name__ == "__main__":
    main()
