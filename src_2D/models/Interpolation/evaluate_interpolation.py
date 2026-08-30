import argparse
import random
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.models.Interpolation.interpolator_2d import (
    LinearViewInterpolator, SinusoidalViewInterpolator)
from src_2D.models.Unet2D.evaluate_2d import (build_full_astra_geometries,
                                              plot_qualitative_example,
                                              reconstruct_volume_fbp,
                                              resolve_compute_device)


def evaluate_interpolator(interpolator, dataset, device, geometry, args, name="interpolation"):
    interpolator.eval()
    interpolator.to(device)
    
    idx = random.randrange(len(dataset))
    incomplete, full, phantom = dataset[idx]
    
    with torch.no_grad():
        incomplete_batch = incomplete.unsqueeze(0).to(device)
        refined_out = interpolator(incomplete_batch)
    
    # Calculate metrics on the sinogram
    full_np = full.squeeze(0).cpu().numpy()
    refined_np = refined_out.squeeze(0).squeeze(0).cpu().numpy()
    
    val_mse = mse(full_np, refined_np)
    val_psnr = psnr(full_np, refined_np, data_range=full_np.max() - full_np.min())
    val_ssim = ssim(full_np, refined_np, data_range=full_np.max() - full_np.min())
    
    print(f"Metrics for {name} on sample {idx}:")
    print(f"MSE:  {val_mse:.6f}")
    print(f"PSNR: {val_psnr:.2f} dB")
    print(f"SSIM: {val_ssim:.4f}")
    
    # Reconstruct FBP
    reconstructed_image = None
    try:
        proj_geom, vol_geom = build_full_astra_geometries(geometry, tuple(phantom.shape[1:]))
        reconstructed_image = reconstruct_volume_fbp(
            refined_np * 100.0, proj_geom, vol_geom
        )
    except Exception as exc:
        print(f"Skipping volume reconstruction panel ({exc}).")
        
    save_path = args.figures_dir / f"{name}_example.png"
    
    # Use a dummy baseline for the 4th panel if evaluating interpolation directly
    # Or, we can use Sinusoidal as baseline and Linear as refined, etc.
    # We will pass the refined_out as both baseline and refined to satisfy the plot function
    fig = plot_qualitative_example(
        phantom.squeeze(0), 
        full.squeeze(0), 
        incomplete.squeeze(0), 
        refined_out.squeeze(0).squeeze(0).cpu(), # baseline (we show the same here)
        refined_out.squeeze(0).squeeze(0).cpu(), # refined
        reconstructed_image, 
        geometry, 
        save_path
    )
    print(f"Saved qualitative example to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate 2D Interpolation models")
    parser.add_argument("--data_dir", type=str, default="data_generation_2D", help="Directory with the dataset")
    parser.add_argument("--figures_dir", type=str, default="outputs/figures", help="Directory to save output figures")
    parser.add_argument("--reconstruct_iters", type=int, default=30, help="SIRT iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    args.figures_dir = Path(args.figures_dir)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_compute_device()
    geometry = DBTGeometryConfig()
    
    try:
        dataset = SinogramCompletionDataset(n_samples=10, is_validation_or_test=True)
    except Exception as e:
        print(f"Could not load dataset from {args.data_dir}: {e}")
        return

    print("Evaluating LinearViewInterpolator...")
    linear_interp = LinearViewInterpolator()
    evaluate_interpolator(linear_interp, dataset, device, geometry, args, name="linear_interp")
    
    print("\nEvaluating SinusoidalViewInterpolator...")
    sinusoidal_interp = SinusoidalViewInterpolator()
    evaluate_interpolator(sinusoidal_interp, dataset, device, geometry, args, name="sinusoidal_interp")


if __name__ == "__main__":
    main()
