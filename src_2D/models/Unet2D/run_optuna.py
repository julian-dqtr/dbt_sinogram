import argparse
import json
import sys
from pathlib import Path

import optuna
import torch
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.models.Unet2D.unet_2d import SinogramUNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-epochs", type=int, default=15)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=8, help="Number of parallel trials")
    parser.add_argument("--use-wandb", action="store_true", help="Log trials to W&B")
    return parser.parse_args()

def objective(trial, args):
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
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD", "RMSprop"])
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])

    run = None
    if args.use_wandb:
        import wandb
        run = wandb.init(
            project="dbt-sinogram-optuna-2d",
            group=trial.study.study_name,
            name=f"trial_{trial.number}",
            config={"lr": lr, "filters": filters_base, "optimizer": optimizer_name, "batch_size": batch_size},
            reinit=True
        )

    # --- Dataset ---
    train_dataset = SinogramCompletionDataset(n_samples=args.n_samples, phantom_type="mixed", device=str(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = SinogramCompletionDataset(n_samples=max(1, args.n_samples // 5), phantom_type="mixed", device=str(device), is_validation_or_test=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Model ---
    model = SinogramUNet(in_channels=1, out_channels=1, filters=filters_base).to(device)
    
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        
    criterion = torch.nn.MSELoss()

    best_ssim = 0.0
    epoch_iter = tqdm(range(args.n_epochs), desc=f"Trial {trial.number}", leave=False)
    for epoch in epoch_iter:
        model.train()
        running_train_loss = 0.0
        for incomplete, target, _ in train_loader:
            incomplete = incomplete.to(device)
            target = target.to(device)
            output = model(incomplete)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            
        train_loss = running_train_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_ssim = 0.0
        running_val_loss = 0.0
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
                for i in range(target_np.shape[0]):
                    batch_ssim += ssim(target_np[i, 0], output_np[i, 0], data_range=float(target_np[i, 0].max() - target_np[i, 0].min()))
                val_ssim += batch_ssim / target_np.shape[0]
                
        val_ssim /= max(1, len(val_loader))
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
            try:
                global_best = trial.study.best_value
            except (ValueError, KeyError):
                global_best = -float("inf")

            if val_ssim > global_best:
                model_save_path = PROJECT_ROOT / "outputs" / "2d" / "best_model_unet2d.pt"
                model_save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_save_path)

    if run is not None:
        run.finish()
        
    return best_ssim

def main():
    args = parse_args()
    
    storage_name = f"sqlite:///{PROJECT_ROOT / 'db' / 'optuna_unet2d.db'}"
    study_name = "unet2d_optimization"
    
    study = optuna.create_study(
        direction="maximize", 
        study_name=study_name, 
        storage=storage_name, 
        load_if_exists=True
    )
    
    print(f"Starting Optuna search in this process...")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, show_progress_bar=False)
    
    print("\n=== Best Trial ===")
    print(f"Value (SSIM): {study.best_trial.value:.4f}")
    print("Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
        
    best_params_path = PROJECT_ROOT / "outputs" / "2d" / "best_optuna_params.json"
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_params_path, "w") as f:
        json.dump({
            "best_value_ssim": study.best_trial.value,
            "params": study.best_trial.params
        }, f, indent=4)
    print(f"\n[+] Saved best parameters to {best_params_path}")

if __name__ == "__main__":
    main()
