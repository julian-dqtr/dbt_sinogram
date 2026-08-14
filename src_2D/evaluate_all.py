import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.models.Interpolation.interpolator_2d import (
    LinearViewInterpolator, SinusoidalViewInterpolator)
from src_2D.models.Unet2D.evaluate_2d import resolve_compute_device

# Mock imports for other models, to be updated with actual paths
# from src_2D.models.Unet2D.unet_2d import UNet2D
# from src_2D.models.GLM.glm_model import GLM
# from src_2D.models.LPD.lpd_model import LPD
# from src_2D.models.SNN.snn_model import SNN


def load_model(model_name: str, device: torch.device):
    """Factory function to load models. Update with actual checkpoint paths."""
    if model_name == "LinearInterpolation":
        return LinearViewInterpolator().to(device)
    elif model_name == "SinusoidalInterpolation":
        return SinusoidalViewInterpolator().to(device)
    elif model_name == "UNet2D":
        from src_2D.models.Unet2D.unet_2d import SinogramUNet
        model = SinogramUNet(in_channels=1, out_channels=1, filters=32).to(device)
        model.load_state_dict(torch.load("outputs/2d/checkpoints/best_model.pt", map_location=device))
        return model
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def evaluate_models(models: dict, dataset, device, num_samples: int = 10):
    """Evaluates a dictionary of models on the first `num_samples` of the dataset."""
    results = []
    
    # We evaluate on a fixed subset to ensure fair comparison
    num_samples = min(num_samples, len(dataset))
    
    for i in range(num_samples):
        incomplete, full, phantom = dataset[i]
        incomplete_batch = incomplete.unsqueeze(0).to(device)
        full_np = full.squeeze(0).cpu().numpy()
        
        for model_name, model in models.items():
            model.eval()
            with torch.no_grad():
                refined_out = model(incomplete_batch)
            
            refined_np = refined_out.squeeze(0).squeeze(0).cpu().numpy()
            
            val_mse = mse(full_np, refined_np)
            val_psnr = psnr(full_np, refined_np, data_range=full_np.max() - full_np.min())
            val_ssim = ssim(full_np, refined_np, data_range=full_np.max() - full_np.min())
            
            results.append({
                "Sample": i,
                "Model": model_name,
                "MSE": val_mse,
                "PSNR": val_psnr,
                "SSIM": val_ssim
            })
            
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate all 2D models")
    parser.add_argument("--data_dir", type=str, default="data_generation_2D", help="Directory with the dataset")
    parser.add_argument("--out_dir", type=str, default="outputs/evaluation", help="Directory to save output metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic evaluation")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    args = parser.parse_args()

    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_compute_device()
    
    try:
        dataset = SinogramCompletionDataset(n_samples=args.num_samples, is_validation_or_test=True)
    except Exception as e:
        print(f"Could not load dataset from {args.data_dir}: {e}")
        return

    model_names = [
        "LinearInterpolation",
        "SinusoidalInterpolation",
        "UNet2D",
        # "GLM",
        # "LPD",
        # "SNN"
    ]
    
    models = {}
    for name in model_names:
        try:
            print(f"Loading {name}...")
            models[name] = load_model(name, device)
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")
            
    if not models:
        print("No models loaded. Exiting.")
        return

    print(f"Evaluating models on {args.num_samples} samples...")
    df_results = evaluate_models(models, dataset, device, num_samples=args.num_samples)
    
    csv_path = args.out_dir / "model_comparison_metrics.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")
    
    # Print summary
    print("\nMean Metrics per Model:")
    summary = df_results.groupby("Model")[["MSE", "PSNR", "SSIM"]].mean()
    print(summary)


if __name__ == "__main__":
    main()
