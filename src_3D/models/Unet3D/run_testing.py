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

from src_3D.conf.geometry_conf_3d import DBTGeometryConfig
from src_3D.data.dataset_3d import SinogramCompletionDataset
from src_3D.models.pipeline_3d import UNet25DWrapper
from src_3D.models.Unet3D.evaluate_3d import (generate_example_figure,
                                              resolve_compute_device)
from src_3D.models.Unet3D.unet_3d import SinogramUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the 3D sinogram completion model.")
    parser.add_argument("--n-samples", type=int, default=30, help="Number of test samples")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "outputs/3d/checkpoints")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "outputs/3d/visualisation")
    parser.add_argument("--reconstruct-iters", type=int, default=15)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="dbt-sinogram-completion-3D")
    parser.add_argument("--wandb-name", type=str, default=None, help="Name of the wandb run for testing")
    parser.add_argument("--wandb-mode", type=str, choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_compute_device()
    geometry = DBTGeometryConfig()

    test_dataset = SinogramCompletionDataset(n_samples=args.n_samples, phantom_type="shepp_logan", device=str(device), is_validation_or_test=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    unet = SinogramUNet(in_channels=1, out_channels=1).to(device)
    model = UNet25DWrapper(unet).to(device)
    criterion = torch.nn.MSELoss()

    run = None
    if args.use_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args), mode=args.wandb_mode, job_type="test")

    model_path = args.checkpoint_dir / "best_model.pt"
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    # Map location handles if it was saved via DDP
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # Remove 'module.' prefix if the model was saved with DDP
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
            
    model.load_state_dict(clean_state_dict)

    print("\nEvaluating model on test set...")
    model.eval()
    test_loss = 0.0
    test_psnr = 0.0
    test_ssim = 0.0
    start_time = time.time()
    
    with torch.no_grad():
        for incomplete, target, _ in tqdm(test_loader, desc="Testing"):
            incomplete = incomplete.to(device)
            target = target.to(device)
            refined_output = model(incomplete)

            test_loss += criterion(refined_output, target).item()

            target_np = target.cpu().numpy()
            output_np = refined_output.cpu().numpy()

            batch_psnr = 0.0
            batch_ssim = 0.0
            total_slices = 0
            for b in range(target_np.shape[0]):
                for y in range(target_np.shape[3]):
                    t = target_np[b, 0, :, y, :]
                    o = output_np[b, 0, :, y, :]
                    batch_psnr += psnr(t, o, data_range=1.0) # We clamped densities to [0, 1]
                    batch_ssim += ssim(t, o, data_range=1.0)
                    total_slices += 1

            test_psnr += batch_psnr / total_slices
            test_ssim += batch_ssim / total_slices

    test_loss /= max(1, len(test_loader))
    test_psnr /= max(1, len(test_loader))
    test_ssim /= max(1, len(test_loader))
    elapsed = time.time() - start_time
    
    print(f"Testing finished in {elapsed:.1f} s.")
    print(f"Test Loss: {test_loss:.4f} | Test PSNR: {test_psnr:.4f} | Test SSIM: {test_ssim:.4f}")

    if run is not None:
        import wandb
        wandb.log({
            "metrics/test_loss": test_loss,
            "metrics/test_psnr": test_psnr,
            "metrics/test_ssim": test_ssim,
        })

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    generate_example_figure(model, test_dataset, device, geometry, args, run)
        
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
