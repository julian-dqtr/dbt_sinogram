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

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.models.SinoSheavesNN.snn_model import SinoSheafNet
from src_2D.models.SinoSheavesNN.physics_loss import AnnealedLoss
from src_2D.models.SinoSheavesNN.graph_data import create_sinogram_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-epochs", type=int, default=15)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--use-wandb", action="store_true", help="Log trials to W&B")
    return parser.parse_args()

def objective(trial, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Hyperparameters Search Space ---
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    num_stalks = trial.suggest_categorical("num_stalks", [8, 16, 32])
    num_layers = trial.suggest_int("num_layers", 2, 6, step=2)
    lambda_m0 = trial.suggest_float("lambda_m0", 1e-3, 1.0, log=True)
    lambda_m1 = trial.suggest_float("lambda_m1", 1e-3, 1.0, log=True)
    anneal_epochs = trial.suggest_int("anneal_epochs", 5, 20)
    batch_size = 1 # Keep batch_size=1 due to PyG dynamic graph construction for now

    run = None
    if args.use_wandb:
        import wandb
        run = wandb.init(
            project="dbt-sinogram-optuna-2d",
            group="SinoSheavesNN_" + trial.study.study_name,
            name=f"trial_{trial.number}",
            config={
                "lr": lr, 
                "num_stalks": num_stalks, 
                "num_layers": num_layers, 
                "lambda_m0": lambda_m0,
                "lambda_m1": lambda_m1,
                "anneal_epochs": anneal_epochs
            },
            reinit=True
        )

    # --- Geometry Setup ---
    config = DBTGeometryConfig()
    geom = DBTGeometry(
        angles=config.full_angles,
        src_radius=config.src_radius_mm,
        det_radius=config.det_radius_mm,
        det_col_count=config.det_col_count,
        det_pixel_size=config.det_pixel_size_mm,
    )

    # --- Dataset ---
    train_dataset = SinogramCompletionDataset(n_samples=args.n_samples, phantom_type="mixed", device=str(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = SinogramCompletionDataset(n_samples=max(1, args.n_samples // 5), phantom_type="mixed", device=str(device), is_validation_or_test=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Model & Loss ---
    model = SinoSheafNet(num_stalks=num_stalks, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = AnnealedLoss(geom, lambda_m0=lambda_m0, lambda_m1=lambda_m1, anneal_epochs=anneal_epochs).to(device)

    best_ssim = 0.0
    epoch_iter = tqdm(range(args.n_epochs), desc=f"Trial {trial.number}", leave=False)
    
    for epoch in epoch_iter:
        model.train()
        running_train_loss = 0.0
        
        for incomplete, target, _ in train_loader:
            inc_sino = incomplete[0, 0].to(device)
            target_sino = target[0, 0].to(device)
            
            # Graph Construction
            graph_data = create_sinogram_data(inc_sino, geom).to(device)
            
            optimizer.zero_grad()
            out = model(graph_data)
            
            pred_sino = out.view(geom.num_views, geom.det_col_count)
            loss, _, _, _ = loss_fn(pred_sino, target_sino, epoch)
            
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
                inc_sino = incomplete[0, 0].to(device)
                target_sino = target[0, 0].to(device)
                
                graph_data = create_sinogram_data(inc_sino, geom).to(device)
                out = model(graph_data)
                pred_sino = out.view(geom.num_views, geom.det_col_count)
                
                v_loss, _, _, _ = loss_fn(pred_sino, target_sino, epoch)
                running_val_loss += v_loss.item()
                
                # Compute SSIM on the sinogram
                target_np = target_sino.cpu().numpy()
                pred_np = pred_sino.cpu().numpy()
                
                # Use data_range derived from target to prevent warnings
                d_range = max(target_np.max() - target_np.min(), 1.0)
                val_ssim += ssim(target_np, pred_np, data_range=d_range)
                
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

    if run is not None:
        run.finish()
        
    return best_ssim

def main():
    args = parse_args()
    study = optuna.create_study(direction="maximize", study_name="snn_2d_optimization")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, show_progress_bar=True)
    
    print("\n=== Best Trial ===")
    print(f"Value (SSIM): {study.best_trial.value:.4f}")
    print("Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
        
    best_params_path = PROJECT_ROOT / "outputs" / "2d" / "best_snn_optuna_params.json"
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_params_path, "w") as f:
        json.dump({
            "best_value_ssim": study.best_trial.value,
            "params": study.best_trial.params
        }, f, indent=4)
    print(f"\n[+] Saved best parameters to {best_params_path}")

if __name__ == "__main__":
    main()
