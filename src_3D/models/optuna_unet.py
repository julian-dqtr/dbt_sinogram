import argparse
import sys
import os
from pathlib import Path

import optuna
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conf.geometry import DBTGeometryConfig
from data.dataset import SinogramCompletionDataset
from models.Unet import SinogramUNet
from models.pipeline import UNet25DWrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-epochs", type=int, default=15)
    parser.add_argument("--n-samples", type=int, default=16) # Moins d'échantillons pour Optuna 3D
    parser.add_argument("--n-jobs", type=int, default=8, help="Number of parallel trials (1 per GPU)")
    parser.add_argument("--use-wandb", action="store_true", help="Log trials to W&B")
    return parser.parse_args()

def objective(trial, args):
    # --- Assign GPU based on trial number ---
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        gpu_id = trial.number % num_gpus
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
        
    print(f"[Trial {trial.number}] Using device {device}")
    
    if torch.cuda.is_available():
        probe = torch.randn(1, 1, 8, 8, device=device)
        conv = torch.nn.Conv2d(1, 1, kernel_size=3).to(device)
        try:
            _ = conv(probe)
        except RuntimeError as exc:
            if "CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH" in str(exc):
                torch.backends.cudnn.enabled = False
    
    # --- Hyperparameters Search Space ---
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    filters_base = trial.suggest_categorical("filters", [16, 32])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])

    run = None
    if args.use_wandb:
        import wandb
        run = wandb.init(
            project="dbt-sinogram-optuna-3d",
            group=trial.study.study_name,
            name=f"trial_{trial.number}",
            config={"lr": lr, "filters": filters_base, "optimizer": optimizer_name},
            reinit=True
        )

    # --- Dataset ---
    train_dataset = SinogramCompletionDataset(n_samples=args.n_samples, phantom_type="ellipses", device=str(device))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataset = SinogramCompletionDataset(n_samples=max(1, args.n_samples // 5), phantom_type="shepp_logan", device=str(device))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # --- Model ---
    unet = SinogramUNet(in_channels=1, out_channels=1).to(device)
    unet.network.filters = [filters_base, filters_base*2, filters_base*4, filters_base*8]
    model = UNet25DWrapper(unet).to(device)
    
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    criterion = torch.nn.MSELoss()

    best_ssim = 0.0
    epoch_iter = tqdm(range(args.n_epochs), desc=f"Trial {trial.number}", leave=False)
    for epoch in epoch_iter:
        model.train()
        running_train_loss = 0.0
        for incomplete, target, _ in train_loader:
            incomplete = incomplete.to(device)
            target = target.to(device)
            
            # Slice sampling for 2.5D (like in train_model.py)
            Y = incomplete.shape[3]
            num_slices = 16
            if Y > num_slices:
                slice_indices = torch.randperm(Y, device=device)[:num_slices]
                incomplete_train = incomplete[:, :, :, slice_indices, :]
                target_train = target[:, :, :, slice_indices, :]
            else:
                incomplete_train = incomplete
                target_train = target
                
            output = model(incomplete_train)
            loss = criterion(output, target_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            
        train_loss = running_train_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_ssim = 0.0
        running_val_loss = 0.0
        total_slices = 0
        with torch.no_grad():
            for incomplete, target, _ in val_loader:
                incomplete = incomplete.to(device)
                target = target.to(device)
                output = model(incomplete)
                
                v_loss = criterion(output, target)
                running_val_loss += v_loss.item()
                
                target_np = target.cpu().numpy()
                output_np = output.cpu().numpy()
                batch_ssim = 0.0
                
                # Compute SSIM slice by slice for 3D
                for b in range(target_np.shape[0]):
                    for y in range(target_np.shape[3]):
                        t = target_np[b, 0, :, y, :]
                        o = output_np[b, 0, :, y, :]
                        batch_ssim += ssim(t, o, data_range=2.0)
                        total_slices += 1
                val_ssim += batch_ssim
                
        val_ssim /= max(1, total_slices)
        val_loss = running_val_loss / max(1, len(val_loader))
        
        if run is not None:
            import wandb
            wandb.log({
                "epoch": epoch,
                "val_ssim": val_ssim,
                "loss/train": train_loss,
                "loss/val": val_loss
            })
            
        trial.report(val_ssim, epoch)
        if trial.should_prune():
            if run is not None:
                run.finish()
            raise optuna.exceptions.TrialPruned()
            
        if val_ssim > best_ssim:
            best_ssim = val_ssim

    if run is not None:
        run.finish()
        
    return best_ssim

def main():
    args = parse_args()
    
    # Use SQLite backend to allow multiple processes
    storage_name = "sqlite:///optuna_3d.db"
    study_name = "unet_3d_optimization"
    
    study = optuna.create_study(
        direction="maximize", 
        study_name=study_name, 
        storage=storage_name, 
        load_if_exists=True
    )
    
    print(f"Starting Optuna search with {args.n_jobs} parallel jobs...")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, n_jobs=args.n_jobs)
    
    print("\n=== Best Trial ===")
    print(f"Value (SSIM): {study.best_trial.value:.4f}")
    print("Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
