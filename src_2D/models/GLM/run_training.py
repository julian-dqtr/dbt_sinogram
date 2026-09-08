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
from torch_geometric.data import Batch
import wandb
from tqdm import tqdm

torch.backends.cudnn.enabled = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.models.GLM.glm_model import GLMNet
from src_2D.models.SinoSheavesNN.physics_loss import AnnealedLoss
from src_2D.utils.evaluation import get_soft_acquired_mask
from src_2D.models.GLM.glm_graph_data import create_glm_sinogram_data
from torch.optim.lr_scheduler import CosineAnnealingLR
from src_2D.utils.evaluation import save_training_curve, save_random_dataset_preview, resolve_compute_device
from src_2D.models.SinoSheavesNN.evaluate_2d import generate_snn_example_figure

@torch.no_grad()
def calibrate_loss(loss_fn, dataloader, geom, device, n_batches=5, is_distributed=False, local_rank=0):
    ref_sinos = []
    for incomplete, target, _ in dataloader:
        if len(ref_sinos) >= n_batches:
            break
        ref_sinos.append(target[:, 0].to(device))
    if len(ref_sinos) > 0:
        stacked = torch.cat(ref_sinos, dim=0)
        loss_fn.calibrate(stacked)
        if is_distributed:
            dist.broadcast(loss_fn.physics_loss_fn.scale_m0, src=0)
            dist.broadcast(loss_fn.physics_loss_fn.scale_m1, src=0)
        if local_rank == 0:
            print(f'[Calibration] scale_m0={loss_fn.physics_loss_fn.scale_m0.item():.4e}  scale_m1={loss_fn.physics_loss_fn.scale_m1.item():.4e}')

def train_epoch(model, dataloader, optimizer, loss_fn, geom, epoch, device, local_rank, is_distributed):
    model.train()
    total_loss = 0.0
    
    batch_iter = dataloader
    if local_rank == 0:
        batch_iter = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
        
    soft_mask = get_soft_acquired_mask(geom, device).squeeze(1) # [1, num_views, 1]
    
    for incomplete_sino, full_sino, _ in batch_iter:
        batch_size = incomplete_sino.size(0)
        target_sinos = full_sino[:, 0].to(device)
        
        # 1. Graph Construction for Batch
        data_list = []
        for i in range(batch_size):
            inc_sino = incomplete_sino[i, 0] # [180, 128]
            data_list.append(create_glm_sinogram_data(inc_sino, geom))
        
        graph_batch = Batch.from_data_list(data_list).to(device)
        
        # 2. Forward Pass
        optimizer.zero_grad()
        out = model(graph_batch) # [N*180, 128]
        
        # Reshape output to [batch_size, num_views, num_detectors]
        pred_sinos = out.view(batch_size, geom.num_views, geom.det_col_count)
        
        # Apply Soft Data Consistency
        pred_sinos = soft_mask * incomplete_sino.squeeze(1).to(device) + (1 - soft_mask) * pred_sinos
        
        # 3. Physics-Informed Annealed Loss (assuming loss_fn handles batched inputs)
        loss, _, _, _ = loss_fn(pred_sinos, target_sinos, epoch)
        
        # 4. Backward & Optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / max(1, len(dataloader))
    
    if is_distributed:
        metrics = torch.tensor([avg_loss], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        avg_loss = (metrics / dist.get_world_size()).item()

    return avg_loss

def val_epoch(model, dataloader, loss_fn, geom, epoch, device, local_rank, is_distributed):
    model.eval()
    total_loss = 0.0
    
    batch_iter = dataloader
    if local_rank == 0:
        batch_iter = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
        
    soft_mask = get_soft_acquired_mask(geom, device).squeeze(1) # [1, num_views, 1]
    
    with torch.no_grad():
        for incomplete_sino, full_sino, _ in batch_iter:
            batch_size = incomplete_sino.size(0)
            target_sinos = full_sino[:, 0].to(device)
            
            data_list = []
            for i in range(batch_size):
                inc_sino = incomplete_sino[i, 0]
                data_list.append(create_glm_sinogram_data(inc_sino, geom))
            
            graph_batch = Batch.from_data_list(data_list).to(device)
            out = model(graph_batch)
            pred_sinos = out.view(batch_size, geom.num_views, geom.det_col_count)
            
            loss, _, _, _ = loss_fn(pred_sinos, target_sinos, epoch)
            total_loss += loss.item()
        
    avg_loss = total_loss / max(1, len(dataloader))
    
    if is_distributed:
        metrics = torch.tensor([avg_loss], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        avg_loss = (metrics / dist.get_world_size()).item()

    return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Train GLM Baseline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5) # Paper uses 5.10^-5
    parser.add_argument("--n_samples", type=int, default=2000)
    
    # GLM Hyperparameters
    parser.add_argument("--num_channels", type=int, default=24)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=7)
    
    parser.add_argument("--lambda_m0", type=float, default=0.0329)
    parser.add_argument("--lambda_m1", type=float, default=0.141)
    parser.add_argument("--anneal_epochs", type=int, default=12)
    parser.add_argument("--wandb_project", type=str, default="SinoSheavesNN")
    parser.add_argument("--wandb_name", type=str, default="GLM-Baseline")
    parser.add_argument("--use-wandb", action="store_true", help="Log metrics to W&B")
    parser.add_argument("--checkpoint_dir", type=Path, default=PROJECT_ROOT / "outputs" / "2d" / "checkpoints_glm")
    parser.add_argument("--figures_dir", type=Path, default=PROJECT_ROOT / "outputs" / "2d" / "figures_glm")
    args = parser.parse_args()

    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        device = resolve_compute_device()

    if local_rank == 0 and args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    config = DBTGeometryConfig()
    geom = DBTGeometry(
        angles=config.full_angles,
        src_radius=config.src_radius_mm,
        det_radius=config.det_radius_mm,
        det_col_count=config.det_col_count,
        det_pixel_size=config.det_pixel_size_mm,
    )
    
    # Dataset splits
    train_dataset = SinogramCompletionDataset(n_samples=args.n_samples, device="cpu")
    val_dataset = SinogramCompletionDataset(n_samples=max(1, args.n_samples // 5), device="cpu", is_validation_or_test=True)
    
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler)
    
    if local_rank == 0:
        save_random_dataset_preview(train_dataset, config, args)
    
    model = GLMNet(
        num_channels=args.num_channels, 
        num_layers=args.num_layers, 
        kernel_size=args.kernel_size
    ).to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    loss_fn = AnnealedLoss(geom, lambda_m0=args.lambda_m0, lambda_m1=args.lambda_m1, anneal_epochs=args.anneal_epochs).to(device)
    
    if local_rank == 0:
        print("Calibrating physics loss normalisation...")
    calibrate_loss(loss_fn, train_loader, geom, device, is_distributed=is_distributed, local_rank=local_rank)

    if local_rank == 0:
        print(f"Starting training for {args.epochs} epochs. Distributed: {is_distributed}")
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        
    best_val_loss = float('inf')
    train_losses = []
    
    for epoch in range(1, args.epochs + 1):
        if is_distributed:
            train_sampler.set_epoch(epoch)
            
        t_loss = train_epoch(model, train_loader, optimizer, loss_fn, geom, epoch, device, local_rank, is_distributed)
        v_loss = val_epoch(model, val_loader, loss_fn, geom, epoch, device, local_rank, is_distributed)
        
        scheduler.step()
        if local_rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:03d} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | LR: {current_lr:.6f}")
            train_losses.append(t_loss)
            
            if wandb.run is not None:
                wandb.log({"Train Loss": t_loss, "Val Loss": v_loss, "Epoch": epoch, "LR": current_lr})
            
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                model_to_save = model.module if is_distributed else model
                torch.save(model_to_save.state_dict(), args.checkpoint_dir / "best_glm_model.pt")
                print(f"  --> Saved new best model (Val Loss: {best_val_loss:.4f})")
                
    if local_rank == 0:
        save_training_curve(train_losses, args)
        
        # Load best model for evaluation
        model_to_eval = model.module if is_distributed else model
        model_to_eval.load_state_dict(torch.load(args.checkpoint_dir / "best_glm_model.pt", map_location=device))
        
        # Hack to temporarily override generate_snn_example_figure's hardcoded paths if needed,
        # but the evaluate script should respect the args.figures_dir.
        generate_snn_example_figure(model_to_eval, val_dataset, device, config, args, wandb.run if wandb.run is not None else None)
        print("Training completed.")
        
    if is_distributed:
        dist.destroy_process_group()
    
if __name__ == "__main__":
    main()
