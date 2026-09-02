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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.models.SinoSheavesNN.snn_model import SinoSheafNet
from src_2D.models.SinoSheavesNN.physics_loss import AnnealedLoss
from src_2D.models.SinoSheavesNN.graph_data import create_sinogram_data

def train_epoch(model, dataloader, optimizer, loss_fn, geom, epoch, device, local_rank, is_distributed):
    model.train()
    total_loss = 0.0
    total_data_loss = 0.0
    total_m0_loss = 0.0
    total_m1_loss = 0.0
    
    batch_iter = dataloader
    if local_rank == 0:
        batch_iter = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        
    for batch_idx, (incomplete_sino, full_sino, _) in enumerate(batch_iter):
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
    parser = argparse.ArgumentParser(description="Train SinoSheavesNN")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.00010616520112201657)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--num_stalks", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--lambda_m0", type=float, default=0.009346058331947786)
    parser.add_argument("--lambda_m1", type=float, default=0.016022295869809182)
    parser.add_argument("--anneal_epochs", type=int, default=7)
    parser.add_argument("--wandb_project", type=str, default="SinoSheavesNN")
    parser.add_argument("--use-wandb", action="store_true", help="Log metrics to W&B")
    parser.add_argument("--checkpoint_dir", type=Path, default=PROJECT_ROOT / "outputs" / "2d" / "checkpoints")
    args = parser.parse_args()

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
    model = SinoSheafNet(num_stalks=args.num_stalks, num_layers=args.num_layers).to(device)
    
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
                torch.save(model_to_save.state_dict(), args.checkpoint_dir / "best_snn_model.pt")
                print(f"  --> Saved new best model with loss: {best_loss:.4f}")
        
    if local_rank == 0:
        print("Training completed.")
        
    if is_distributed:
        dist.destroy_process_group()
    
if __name__ == "__main__":
    main()
