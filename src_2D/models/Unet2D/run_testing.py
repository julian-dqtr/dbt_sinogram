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

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.models.Unet2D.evaluate_2d import (generate_example_figure,
                                              resolve_compute_device)
from src_2D.models.Unet2D.unet_2d import SinogramUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the 2D sinogram completion model.")
    parser.add_argument("--n-samples", type=int, default=60, help="Number of test samples")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--filters", type=int, default=32)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "outputs/2d/checkpoints")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "outputs/2d/visualisation")
    parser.add_argument("--reconstruct-iters", type=int, default=20)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="dbt-sinogram-completion-2D")
    parser.add_argument("--wandb-name", type=str, default=None, help="Name of the wandb run for testing")
    parser.add_argument("--wandb-mode", type=str, choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_compute_device()
    geometry = DBTGeometryConfig()

    test_dataset = SinogramCompletionDataset(n_samples=args.n_samples, phantom_type="shepp_logan", device=str(device), is_validation_or_test=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = SinogramUNet(in_channels=1, out_channels=1, filters=args.filters).to(device)
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
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

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
            for i in range(target_np.shape[0]):
                t = target_np[i, 0]
                o = output_np[i, 0]
                batch_psnr += psnr(t, o, data_range=float(t.max() - t.min())) # Updated data_range to 1.0 since we clamped to [0, 1]
                batch_ssim += ssim(t, o, data_range=float(t.max() - t.min())) # Updated data_range to 1.0 since we clamped to [0, 1]

            test_psnr += batch_psnr / target_np.shape[0]
            test_ssim += batch_ssim / target_np.shape[0]

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
