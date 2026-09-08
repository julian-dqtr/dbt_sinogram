import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import optuna
import torch
from skimage.metrics import structural_similarity as ssim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

torch.backends.cudnn.enabled = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.models.GLM.glm_graph_data import create_glm_sinogram_data
from src_2D.models.SinoSheavesNN.physics_loss import AnnealedLoss
from src_2D.models.GLM.glm_model import GLMNet
from src_2D.utils.evaluation import get_soft_acquired_mask


@torch.no_grad()
def calibrate_loss(loss_fn, dataloader, geom, device, n_batches=5):
    ref_sinos = []
    for incomplete, target, _ in dataloader:
        if len(ref_sinos) >= n_batches:
            break
        ref_sinos.append(target[:, 0].to(device))
    if len(ref_sinos) > 0:
        stacked = torch.cat(ref_sinos, dim=0)  # [n_batches * b_size, V, D]
        loss_fn.calibrate(stacked)


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for GLM Baseline")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials for this worker")
    parser.add_argument("--n-epochs", type=int, default=15, help="Number of epochs per trial")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of training samples")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel trials per process")
    parser.add_argument("--gpu-id", type=int, default=None, help="Explicit GPU ID to use")
    parser.add_argument("--use-wandb", action="store_true", help="Log trials to W&B")
    parser.add_argument(
        "--storage",
        type=str,
        default=f"sqlite:///{PROJECT_ROOT / 'db' / 'optuna_glm_2d.db'}",
        help="Optuna storage URL for distributed optimization",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="glm_2d_optimization",
        help="Name of the study",
    )
    return parser.parse_args()


def objective(trial, args):
    # --- Assign GPU ---
    if args.gpu_id is not None:
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(args.gpu_id)
    elif torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = trial.number % num_gpus
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device("cpu")

    print(f"[Trial {trial.number}] Using device {device}")

    # --- Hyperparameters Search Space ---
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    num_channels = trial.suggest_categorical("num_channels", [16, 24, 32, 64])
    num_layers = trial.suggest_categorical("num_layers", [2, 3, 4, 6])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7, 9])
    
    lambda_m0 = trial.suggest_float("lambda_m0", 1e-3, 0.5, log=True)
    lambda_m1 = trial.suggest_float("lambda_m1", 1e-3, 0.5, log=True)
    anneal_epochs = trial.suggest_int("anneal_epochs", 3, max(4, args.n_epochs // 2))
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = args.batch_size

    run = None
    if args.use_wandb:
        import wandb

        run = wandb.init(
            project="dbt-sinogram-optuna-2d",
            group="GLM_" + args.study_name,
            name=f"trial_{trial.number}_gpu{device.index if device.type == 'cuda' else 'cpu'}",
            config={
                "lr": lr,
                "num_channels": num_channels,
                "num_layers": num_layers,
                "kernel_size": kernel_size,
                "lambda_m0": lambda_m0,
                "lambda_m1": lambda_m1,
                "anneal_epochs": anneal_epochs,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "n_samples": args.n_samples,
                "n_epochs": args.n_epochs,
            },
            reinit=True,
        )

    try:
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
        train_dataset = SinogramCompletionDataset(n_samples=args.n_samples, phantom_type="mixed", device="cpu")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = SinogramCompletionDataset(
            n_samples=max(1, args.n_samples // 5), phantom_type="mixed", device="cpu", is_validation_or_test=True
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # --- Model & Loss ---
        model = GLMNet(num_channels=num_channels, num_layers=num_layers, kernel_size=kernel_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=lr * 0.01)
        loss_fn = AnnealedLoss(geom, lambda_m0=lambda_m0, lambda_m1=lambda_m1, anneal_epochs=anneal_epochs).to(device)

        soft_mask = get_soft_acquired_mask(geom, device).squeeze(1)  # [1, num_views, 1]

        # Calibrate physics loss
        calibrate_loss(loss_fn, train_loader, geom, device)

        best_ssim = 0.0
        epoch_iter = tqdm(range(args.n_epochs), desc=f"Trial {trial.number}", leave=False)

        for epoch in epoch_iter:
            model.train()
            running_train_loss = 0.0

            for incomplete, target, _ in train_loader:
                b_size = incomplete.size(0)
                target_sinos = target[:, 0].to(device)

                data_list = []
                for i in range(b_size):
                    inc_sino = incomplete[i, 0]
                    data_list.append(create_glm_sinogram_data(inc_sino, geom))

                graph_batch = Batch.from_data_list(data_list).to(device)

                optimizer.zero_grad()
                out = model(graph_batch)
                
                # Reshape output to [batch_size, num_views, num_detectors]
                pred_sinos = out.view(b_size, geom.num_views, geom.det_col_count)

                # Apply Soft Data Consistency
                pred_sinos = soft_mask * incomplete.squeeze(1).to(device) + (1 - soft_mask) * pred_sinos

                loss, _, _, _ = loss_fn(pred_sinos, target_sinos, epoch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_train_loss += loss.item()

            scheduler.step()
            train_loss = running_train_loss / max(1, len(train_loader))

            # Validation
            model.eval()
            val_ssim = 0.0
            running_val_loss = 0.0
            with torch.no_grad():
                for incomplete, target, _ in val_loader:
                    b_size = incomplete.size(0)
                    target_sinos = target[:, 0].to(device)

                    data_list = []
                    for i in range(b_size):
                        inc_sino = incomplete[i, 0]
                        data_list.append(create_glm_sinogram_data(inc_sino, geom))

                    graph_batch = Batch.from_data_list(data_list).to(device)
                    out = model(graph_batch)
                    pred_sinos = out.view(b_size, geom.num_views, geom.det_col_count)

                    # Apply Soft Data Consistency
                    pred_sinos = soft_mask * incomplete.squeeze(1).to(device) + (1 - soft_mask) * pred_sinos

                    v_loss, _, _, _ = loss_fn(pred_sinos, target_sinos, epoch)
                    running_val_loss += v_loss.item()

                    # Compute SSIM on the sinogram (batch-wise)
                    target_np = target_sinos.cpu().numpy()
                    pred_np = pred_sinos.cpu().numpy()

                    batch_ssim = 0.0
                    for i in range(b_size):
                        d_range = max(float(target_np[i].max() - target_np[i].min()), 1.0)
                        batch_ssim += ssim(target_np[i], pred_np[i], data_range=d_range)
                    val_ssim += batch_ssim / b_size

            val_ssim /= max(1, len(val_loader))
            val_loss = running_val_loss / max(1, len(val_loader))

            if run is not None:
                run.log(
                    {
                        "epoch": epoch,
                        "val_ssim": val_ssim,
                        "loss/train": train_loss,
                        "loss/val": val_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

            trial.report(val_ssim, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_ssim > best_ssim:
                best_ssim = val_ssim

                # Save best model checkpoint globally
                try:
                    global_best = trial.study.best_value
                except (ValueError, KeyError):
                    global_best = -float("inf")

                if val_ssim > global_best:
                    model_save_path = PROJECT_ROOT / "outputs" / "2d" / "best_model_glm.pt"
                    model_save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), model_save_path)

        return best_ssim
    finally:
        if run is not None:
            run.finish()


def ensure_sqlite_wal(storage_url: str):
    """Enable SQLite WAL mode to avoid database locking during distributed multi-worker runs."""
    if storage_url.startswith("sqlite:///"):
        db_path = Path(storage_url.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(str(db_path), timeout=60)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=60000;")
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Warning: Could not enable WAL mode on SQLite: {e}")


def main():
    args = parse_args()

    ensure_sqlite_wal(args.storage)

    storage = optuna.storages.RDBStorage(
        url=args.storage,
        engine_kwargs={"connect_args": {"timeout": 60}},
    )

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=storage,
        pruner=pruner,
        load_if_exists=True,
    )

    print(f"Starting Optuna search with {args.n_jobs} parallel jobs on study '{args.study_name}' (GPU: {args.gpu_id})...")
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=(args.n_jobs == 1),
    )

    try:
        print("\n=== Best Trial ===")
        print(f"Value (SSIM): {study.best_trial.value:.4f}")
        print("Params:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

        best_params_path = PROJECT_ROOT / "outputs" / "2d" / "best_glm_optuna_params.json"
        best_params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(best_params_path, "w") as f:
            json.dump(
                {
                    "best_value_ssim": study.best_trial.value,
                    "params": study.best_trial.params,
                },
                f,
                indent=4,
            )
        print(f"\n[+] Saved best parameters to {best_params_path}")
    except (ValueError, KeyError):
        print("\nNo completed trials found in this study yet.")


if __name__ == "__main__":
    main()
