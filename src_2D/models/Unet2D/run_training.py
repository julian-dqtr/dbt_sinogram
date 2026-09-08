from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.utils.evaluation import (generate_example_figure,
                                              resolve_compute_device,
                                              save_random_dataset_preview,
                                              save_training_curve)
from src_2D.models.Unet2D.unet_2d import SinogramUNet
from src_2D.conf.geometry_conf_2d import DBTGeometryConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 2D sinogram completion model.")
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0042275432102115195)
    parser.add_argument("--filters", type=int, default=32)
    parser.add_argument("--optimizer", type=str, choices=("Adam", "AdamW", "RMSprop"), default="AdamW")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "outputs/2d/checkpoints")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "outputs/2d/visualisation")
    parser.add_argument("--reconstruct-iters", type=int, default=15)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="dbt-sinogram-completion-2D")
    parser.add_argument("--wandb-name", type=str, default=None, help="Name of the wandb run")
    parser.add_argument("--wandb-mode", type=str, choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_compute_device()
    geometry = DBTGeometryConfig()

    train_dataset = SinogramCompletionDataset(n_samples=args.n_samples, phantom_type="mixed", device=str(device))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = SinogramCompletionDataset(n_samples=max(1, args.n_samples // 5), phantom_type="mixed", device=str(device), is_validation_or_test=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    save_random_dataset_preview(train_dataset, geometry, args)

    model = SinogramUNet(in_channels=1, out_channels=1, filters=args.filters).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    run = None
    if args.use_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args), mode=args.wandb_mode)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float("inf")
    epochs_without_improvement = 0
    start_time = time.time()
    train_losses = []

    model.train()
    
    epoch_iter = tqdm(range(args.max_epochs), desc="training", unit="epoch")
    try:
        for epoch in epoch_iter:
            running_loss = 0.0
            
            batch_iter = tqdm(train_loader, desc=f"epoch {epoch + 1}", leave=False, unit="batch")
            for incomplete, target, _ in batch_iter:
                incomplete = incomplete.to(device)
                target = target.to(device)

                refined_output = model(incomplete)
                loss = criterion(refined_output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batch_iter.set_postfix(unet=f"{loss.item():.4f}")

            train_loss = running_loss / max(1, len(train_loader))
            
            # Validation loop
            model.eval()
            val_loss = 0.0
            val_psnr = 0.0
            val_ssim = 0.0
            with torch.no_grad():
                for incomplete, target, _ in val_loader:
                    incomplete = incomplete.to(device)
                    target = target.to(device)
                    refined_output = model(incomplete)
                    
                    val_loss += criterion(refined_output, target).item()
                    
                    # Calculate metrics on CPU
                    target_np = target.cpu().numpy()
                    output_np = refined_output.cpu().numpy()
                    
                    batch_psnr = 0.0
                    batch_ssim = 0.0
                    for i in range(target_np.shape[0]):
                        t = target_np[i, 0]
                        o = output_np[i, 0]
                        batch_psnr += psnr(t, o, data_range=float(t.max() - t.min()))
                        batch_ssim += ssim(t, o, data_range=float(t.max() - t.min()))
                    
                    val_psnr += batch_psnr / target_np.shape[0]
                    val_ssim += batch_ssim / target_np.shape[0]

            val_loss /= max(1, len(val_loader))
            val_psnr /= max(1, len(val_loader))
            val_ssim /= max(1, len(val_loader))
            model.train()

            train_losses.append(train_loss)
            epoch_iter.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", ssim=f"{val_ssim:.4f}")

            if run is not None:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "loss/train_unet": train_loss,
                    "loss/val_unet": val_loss,
                    "metrics/val_psnr": val_psnr,
                    "metrics/val_ssim": val_ssim,
                    "loss/best_val": min(best_loss, val_loss),
                })

            if val_loss < best_loss - args.min_delta:
                best_loss = val_loss
                epochs_without_improvement = 0
                state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save(state_dict, args.checkpoint_dir / "best_model.pt")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"\nNo improvement for {args.patience} epochs, stopping early at epoch {epoch + 1}.")
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user, saving current model state.")
        state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        torch.save(state_dict, args.checkpoint_dir / "interrupted_model.pt")

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed / 60:.1f} min. Best loss (U-Net): {best_loss:.4f}")
    
    # Reload best model for saving curve/examples (optional, but good for validation example)
    if (args.checkpoint_dir / "best_model.pt").exists():
        state_dict = torch.load(args.checkpoint_dir / "best_model.pt", map_location=device, weights_only=True)
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

    save_training_curve(train_losses, args)
    # Generate an example from validation set to see training result
    generate_example_figure(model, val_dataset, device, geometry, args, run)
        
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
