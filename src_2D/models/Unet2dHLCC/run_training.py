import os
import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm import tqdm

torch.backends.cudnn.enabled = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.models.Unet2dHLCC.unet_hlcc import Unet2dHLCC
from src_2D.models.SinoSheavesNN.physics_loss import AnnealedLoss

def train_epoch(model, dataloader, optimizer, loss_fn, geom, epoch, device, local_rank, is_distributed):
    model.train()
    total_loss = 0.0
    total_data_loss = 0.0
    total_m0_loss = 0.0
    total_m1_loss = 0.0
    
    batch_iter = dataloader
    if local_rank == 0:
        batch_iter = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        
    # Precompute acquired_mask for Data Consistency
    # We create it once on the correct device and reuse it for all batches.
    import numpy as np
    full_angles_deg = np.rad2deg(geom.angles)
    config = DBTGeometryConfig() # We can instantiate it locally or pass it
    in_window = (full_angles_deg >= config.angle_min_deg) & (full_angles_deg <= config.angle_max_deg)
    acquired_mask_1d = torch.zeros(geom.num_views, device=device)
    acquired_mask_1d[in_window] = 1.0
    # Reshape to broadcast correctly against [B, C, Views, Detectors]
    acquired_mask = acquired_mask_1d.view(1, 1, geom.num_views, 1)

    for batch_idx, (incomplete_sino, full_sino, _) in enumerate(batch_iter):
        # We assume batch_size=1 for the physics loss to work correctly
        # incomplete_sino is [B, C, Views, Detectors]
        inc_sino = incomplete_sino.to(device)
        
        # Target for MSE
        target_sino = full_sino[0, 0].to(device) # shape [Views, Detectors]
        
        # 1. Forward Pass with Data Consistency Layer
        optimizer.zero_grad()
        out = model(inc_sino, acquired_mask) # shape [B, C, Views, Detectors]
        
        # Reshape output to match target and physics loss
        pred_sino = out[0, 0] # shape [Views, Detectors]
        
        # 2. Physics-Informed Annealed Loss
        loss, data_loss, m0_loss, m1_loss = loss_fn(pred_sino, target_sino, epoch)
        
        # 3. Backward & Optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_data_loss += data_loss.item()
        total_m0_loss += m0_loss.item()
        total_m1_loss += m1_loss.item()
        
    num_batches = max(1, len(dataloader))
    avg_loss = total_loss / num_batches
    avg_data_loss = total_data_loss / num_batches
    avg_m0_loss = total_m0_loss / num_batches
    avg_m1_loss = total_m1_loss / num_batches
    
    # Sync metrics across GPUs
    if is_distributed:
        metrics = torch.tensor([avg_loss, avg_data_loss, avg_m0_loss, avg_m1_loss], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= dist.get_world_size()
        avg_loss, avg_data_loss, avg_m0_loss, avg_m1_loss = metrics.tolist()

    return avg_loss, avg_data_loss, avg_m0_loss, avg_m1_loss

def main():
    parser = argparse.ArgumentParser(description="Train Unet2dHLCC")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--filters", type=int, default=32, help="Number of filters for U-Net")
    parser.add_argument("--lambda_m0", type=float, default=1e-7)
    parser.add_argument("--lambda_m1", type=float, default=1e-7)
    parser.add_argument("--anneal_epochs", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default="Unet2dHLCC")
    parser.add_argument("--use-wandb", action="store_true", help="Log metrics to W&B")
    parser.add_argument("--checkpoint_dir", type=Path, default=PROJECT_ROOT / "outputs" / "2d" / "checkpoints_unet2dhlcc")
    args = parser.parse_args()

    # Note: we force batch_size to 1 because physics_loss currently supports batch_size=1
    if args.batch_size != 1:
        print("Warning: forcing batch_size to 1 because HelgasonLudwigLoss currently requires it.")
        args.batch_size = 1

    # Setup DDP
    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb (Main Process Only)
    if local_rank == 0 and args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    # Geometry Setup
    config = DBTGeometryConfig()
    geom = DBTGeometry(
        angles=config.full_angles,
        src_radius=config.src_radius_mm,
        det_radius=config.det_radius_mm,
        det_col_count=config.det_col_count,
        det_pixel_size=config.det_pixel_size_mm,
    )
    
    # Dataset & Dataloader
    dataset = SinogramCompletionDataset(
        n_samples=args.n_samples,
        geometry_config=config,
        device=str(device)
    )
    
    sampler = DistributedSampler(dataset) if is_distributed else None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler)
    
    # Model & Loss
    model = Unet2dHLCC(in_channels=1, out_channels=1, filters=args.filters).to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = AnnealedLoss(geom, lambda_m0=args.lambda_m0, lambda_m1=args.lambda_m1, anneal_epochs=args.anneal_epochs).to(device)
    
    if local_rank == 0:
        print(f"Starting training for {args.epochs} epochs. Distributed: {is_distributed}")
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_loss = float('inf')
    
    for epoch in range(args.epochs):
        if is_distributed:
            sampler.set_epoch(epoch)
            
        loss, d_loss, m0_loss, m1_loss = train_epoch(model, dataloader, optimizer, loss_fn, geom, epoch, device, local_rank, is_distributed)
        
        if local_rank == 0:
            print(f"Epoch {epoch}: Total={loss:.4f} | Data={d_loss:.4f} | M0={m0_loss:.4f} | M1={m1_loss:.4f}")
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "Total Loss": loss,
                    "Data Loss (MSE)": d_loss,
                    "M0 Variance Loss": m0_loss,
                    "M1 Residual Loss": m1_loss,
                    "Epoch": epoch
                })
            
            if loss < best_loss:
                best_loss = loss
                model_to_save = model.module if is_distributed else model
                torch.save(model_to_save.state_dict(), args.checkpoint_dir / "best_unet2dhlcc_model.pt")
                print(f"  --> Saved new best model with loss: {best_loss:.4f}")
        
    if local_rank == 0:
        print("Training completed. Generating visualizations...")
        
        # We need to collect training losses during the loop if we want to plot them.
        # But since we didn't store them in a list globally, we'll just generate the example figure.
        try:
            from src_2D.models.Unet2dHLCC.evaluate_2d import generate_example_figure
            args.figures_dir = PROJECT_ROOT / "outputs" / "2d" / "figures_unet2dhlcc"
            
            # Use the latest model weights for evaluation
            if is_distributed:
                model.module.eval()
                generate_example_figure(model.module, dataset, device, geom, args, wandb.run if wandb.run is not None else None)
            else:
                model.eval()
                generate_example_figure(model, dataset, device, geom, args, wandb.run if wandb.run is not None else None)
        except ImportError as e:
            print(f"Warning: could not import or generate visualisations: {e}")
        
    if is_distributed:
        dist.destroy_process_group()
    
if __name__ == "__main__":
    main()
