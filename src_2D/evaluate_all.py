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
from src_2D.utils.evaluation import resolve_compute_device


def load_model(model_name: str, device: torch.device):
    """Factory function to load models. Returns (model_or_callable, is_nn_model)."""
    if model_name == "UNet2D":
        from src_2D.models.Unet2D.unet_2d import SinogramUNet
        model = SinogramUNet(in_channels=1, out_channels=1, filters=32).to(device)
        model.load_state_dict(torch.load("outputs/2d/checkpoints/best_model.pt", map_location=device, weights_only=True))
        return model, True
    elif model_name == "UNet2dRNO":
        from src_2D.models.Unet2dRNO.unet_rno import RadonInformedUNet
        from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
        geometry = DBTGeometryConfig()
        model = RadonInformedUNet(geometry_config=geometry, in_channels=1, out_channels=1, filters=32).to(device)
        model.load_state_dict(torch.load("outputs/2d/checkpoints_rno/best_model.pt", map_location=device, weights_only=True))
        return model, True
    elif model_name == "UNet2dHLCC":
        from src_2D.models.Unet2dHLCC.unet_hlcc import Unet2dHLCC
        model = Unet2dHLCC(in_channels=1, out_channels=1, filters=32).to(device)
        model.load_state_dict(torch.load("outputs/2d/checkpoints_unet2dhlcc/best_unet2dhlcc_model.pt", map_location=device, weights_only=True))
        return model, True
    elif model_name == "SNN":
        from src_2D.models.SinoSheavesNN.snn_model import SinoSheafNet
        model = SinoSheafNet(num_stalks=128, num_layers=6).to(device)
        model.load_state_dict(torch.load("outputs/2d/checkpoints/best_snn_model.pt", map_location=device, weights_only=True))
        return model, True
    elif model_name == "GLM":
        from src_2D.models.GLM.glm_model import GLMNet
        model = GLMNet(num_channels=24, num_layers=3, kernel_size=7).to(device)
        model.load_state_dict(torch.load("outputs/2d/checkpoints_glm/best_glm_model.pt", map_location=device, weights_only=True))
        return model, True
    elif model_name == "ZeroFilling":
        from src_2D.models.baselines import zero_filling
        return zero_filling, False
    elif model_name == "LinearInterp":
        from src_2D.models.baselines import linear_interpolation
        return linear_interpolation, False
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def evaluate_models(models: dict, dataset, device, num_samples: int = 10):
    """Evaluates a dictionary of models on the first `num_samples` of the dataset."""
    results = []

    # We evaluate on a fixed subset to ensure fair comparison
    num_samples = min(num_samples, len(dataset))

    from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
    config = DBTGeometryConfig()
    from src_2D.utils.evaluation import get_soft_acquired_mask
    from src_2D.geometry.dbt_geometry_2d import DBTGeometry
    geom = DBTGeometry(
        angles=config.full_angles,
        src_radius=config.src_radius_mm,
        det_radius=config.det_radius_mm,
        det_col_count=config.det_col_count,
        det_pixel_size=config.det_pixel_size_mm,
    )
    acquired_mask = get_soft_acquired_mask(geom, device)

    for i in range(num_samples):
        incomplete, full, phantom = dataset[i]
        incomplete_batch = incomplete.unsqueeze(0).to(device)
        full_np = full.squeeze(0).cpu().numpy()

        for model_name, (model_or_fn, is_nn) in models.items():
            if is_nn:
                model_or_fn.eval()

            with torch.no_grad():
                if model_name == "SNN":
                    # SNN requires PyG Data conversion
                    from src_2D.models.SinoSheavesNN.graph_data import create_sinogram_data
                    from torch_geometric.data import Batch
                    inc_sino = incomplete[0].to(device)  # [180, 128]
                    graph_data = create_sinogram_data(inc_sino, geom).to(device)
                    graph_batch = Batch.from_data_list([graph_data])
                    refined_out = model_or_fn(graph_batch)  # [180, 128]
                    refined_out = refined_out.unsqueeze(0).unsqueeze(0)  # [1, 1, 180, 128]
                elif model_name == "GLM":
                    # GLM requires PyG Data conversion
                    from src_2D.models.GLM.glm_graph_data import create_glm_sinogram_data
                    from torch_geometric.data import Batch
                    inc_sino = incomplete[0].to(device)  # [180, 128]
                    graph_data = create_glm_sinogram_data(inc_sino, geom).to(device)
                    graph_batch = Batch.from_data_list([graph_data])
                    refined_out = model_or_fn(graph_batch)  # [180, 128]
                    refined_out = refined_out.unsqueeze(0).unsqueeze(0)  # [1, 1, 180, 128]
                elif model_name == "UNet2dHLCC":
                    refined_out = model_or_fn(incomplete_batch, acquired_mask)
                elif is_nn:
                    refined_out = model_or_fn(incomplete_batch)
                else:
                    # Non-learned baseline (callable)
                    refined_out = model_or_fn(incomplete_batch)

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
        "ZeroFilling",
        "LinearInterp",
        "UNet2D",
        "UNet2dRNO",
        "UNet2dHLCC",
        "SNN",
        "GLM",
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
