import argparse
import json
import sys
from pathlib import Path

import optuna
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
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


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for Unet2dHLCC")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials for this worker")
    parser.add_argument("--n-epochs", type=int, default=15, help="Number of epochs per trial")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of training samples")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel trials per process")
    parser.add_argument("--gpu-id", type=int, default=None, help="Explicit GPU ID to use")
    parser.add_argument("--use-wandb", action="store_true", help="Log trials to W&B")
    parser.add_argument(
        "--storage",
        type=str,
        default=f"sqlite:///{PROJECT_ROOT / 'db' / 'optuna_unet2dhlcc.db'}",
        help="Optuna storage URL for distributed optimization",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="unet2dhlcc_optimization",
        help="Name of the study",
    )
    return parser.parse_args()


def objective(trial, args):
    if args.gpu_id is not None:
        device = torch.device(f"cuda:{args.gpu_id}")
    elif torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = trial.number % num_gpus
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
        
    print(f"[Trial {trial.number}] Using device {device}")
    
    # --- Hyperparameters Search Space ---
    lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    filters = trial.suggest_categorical("filters", [16, 32, 64])
    lambda_m0 = trial.suggest_float("lambda_m0", 1e-2, 2.0, log=True)
    lambda_m1 = trial.suggest_float("lambda_m1", 1e-3, 1.0, log=True)
    # anneal_epochs should be ~10-30% of total epochs so MSE loss dominates early
    anneal_epochs = trial.suggest_int("anneal_epochs", 20, max(21, args.n_epochs // 2))
    batch_size = 1

    run = None
    if args.use_wandb:
        import wandb
        run = wandb.init(
            project="dbt-sinogram-optuna-2d",
            group="Unet2dHLCC_" + args.study_name,
            name=f"trial_{trial.number}",
            config={
                "lr": lr, 
                "filters": filters, 
                "lambda_m0": lambda_m0,
                "lambda_m1": lambda_m1,
                "anneal_epochs": anneal_epochs
            },
            reinit=True
        )

    try:
        config = DBTGeometryConfig()
        geom = DBTGeometry(
            angles=config.full_angles,
            src_radius=config.src_radius_mm,
            det_radius=config.det_radius_mm,
            det_col_count=config.det_col_count,
            det_pixel_size=config.det_pixel_size_mm,
        )

        train_dataset = SinogramCompletionDataset(n_samples=args.n_samples, phantom_type="mixed", device=str(device))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = SinogramCompletionDataset(n_samples=max(1, args.n_samples // 5), phantom_type="mixed", device=str(device), is_validation_or_test=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = Unet2dHLCC(in_channels=1, out_channels=1, filters=filters).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=lr * 0.01)
        loss_fn = AnnealedLoss(geom, lambda_m0=lambda_m0, lambda_m1=lambda_m1, anneal_epochs=anneal_epochs).to(device)

        # Get soft acquired_mask for Data Consistency
        from src_2D.utils.evaluation import get_soft_acquired_mask
        acquired_mask = get_soft_acquired_mask(geom, device)

        # Calibrate physics loss normalisation before training
        with torch.no_grad():
            ref_sinos = []
            for i, (_, full, _) in enumerate(train_loader):
                if i >= 5:
                    break
                ref_sinos.append(full[0, 0].to(device))
            loss_fn.calibrate(torch.stack(ref_sinos))

        best_ssim = 0.0
        epoch_iter = tqdm(range(args.n_epochs), desc=f"Trial {trial.number}", leave=False)
        
        for epoch in epoch_iter:
            model.train()
            running_train_loss = 0.0
            
            for incomplete, target, _ in train_loader:
                inc_sino = incomplete.to(device)
                target_sino = target[0, 0].to(device)
                
                optimizer.zero_grad()
                out = model(inc_sino, acquired_mask)
                
                pred_sino = out[0, 0]
                loss, _, _, _ = loss_fn(pred_sino, target_sino, epoch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_train_loss += loss.item()
                
            train_loss = running_train_loss / max(1, len(train_loader))
            scheduler.step()

            # Validation
            model.eval()
            val_ssim = 0.0
            running_val_loss = 0.0
            with torch.no_grad():
                for incomplete, target, _ in val_loader:
                    inc_sino = incomplete.to(device)
                    target_sino = target[0, 0].to(device)
                    
                    out = model(inc_sino, acquired_mask)
                    pred_sino = out[0, 0]
                    
                    v_loss, _, _, _ = loss_fn(pred_sino, target_sino, epoch)
                    running_val_loss += v_loss.item()
                    
                    target_np = target_sino.cpu().numpy()
                    pred_np = pred_sino.cpu().numpy()
                    d_range = max(target_np.max() - target_np.min(), 1.0)
                    val_ssim += ssim(target_np, pred_np, data_range=d_range)
                    
            val_ssim /= max(1, len(val_loader))
            val_loss = running_val_loss / max(1, len(val_loader))
            
            if run is not None:
                run.log({
                    "epoch": epoch,
                    "val_ssim": val_ssim,
                    "loss/train": train_loss,
                    "loss/val": val_loss
                })
                
            trial.report(val_ssim, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                try:
                    global_best = trial.study.best_value
                except (ValueError, KeyError):
                    global_best = -float("inf")

                if val_ssim > global_best:
                    model_save_path = PROJECT_ROOT / "outputs" / "2d" / "best_model_unet2dhlcc.pt"
                    model_save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), model_save_path)

        return best_ssim
    finally:
        if run is not None:
            run.finish()


def main():
    args = parse_args()
    
    study = optuna.create_study(
        direction="maximize", 
        study_name=args.study_name, 
        storage=args.storage, 
        load_if_exists=True
    )
    
    print(f"Starting Optuna search with {args.n_jobs} parallel jobs on study '{args.study_name}'...")
    study.optimize(
        lambda trial: objective(trial, args), 
        n_trials=args.n_trials, 
        n_jobs=args.n_jobs, 
        show_progress_bar=(args.n_jobs == 1)
    )
    
    try:
        print("\n=== Best Trial ===")
        print(f"Value (SSIM): {study.best_trial.value:.4f}")
        print("Params:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
            
        best_params_path = PROJECT_ROOT / "outputs" / "2d" / "best_unet2dhlcc_optuna_params.json"
        best_params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(best_params_path, "w") as f:
            json.dump({
                "best_value_ssim": study.best_trial.value,
                "params": study.best_trial.params
            }, f, indent=4)
        print(f"\n[+] Saved best parameters to {best_params_path}")
    except ValueError:
        print("\nNo completed trials found in this study yet.")


if __name__ == "__main__":
    main()
