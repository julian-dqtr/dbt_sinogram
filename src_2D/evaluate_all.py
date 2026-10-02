"""Evaluate every model of the protocol on the immutable TEST split.

    python src_2D/evaluate_all.py
    python src_2D/evaluate_all.py --models ZeroFilling LinearInterp UNet2D GCN_L6 SNN_L6 --num_samples 200

Outputs (in --out_dir):
    model_comparison_metrics.csv   one row per (sample, model): sinogram + image metrics
    per_view_mse.csv               MSE of every view, per model: the error profile as a
                                   function of the angular distance to the acquired window,
                                   to be compared with the angular reach of each graph model.

A model whose checkpoint is missing, legacy, or trained on another geometry is reported as
NOT EVALUATED and left out of the table: a randomly initialised network is never scored.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Allow script to be run directly by adding project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.models.factory import get_model
from src_2D.utils.evaluation import build_full_astra_geometries, reconstruct_volume_fbp
from src_2D.utils.metrics import sinogram_metrics, ssim

torch.backends.cudnn.enabled = False

DEFAULT_MODELS = [
    "ZeroFilling", "LinearInterp", "UNet2D", "UNet2dHLCC",
    "GCN_L6", "GCN_L12", "GCN_L18", "SNN_L6", "SNN_L12", "SNN_L18",
]


def image_metrics(sinogram: np.ndarray, phantom: np.ndarray, config, proj_geom, vol_geom) -> dict:
    """FBP of a (normalised) sinogram compared with the phantom (values in [0, 1], data_range = 1)."""
    reco = reconstruct_volume_fbp(sinogram * config.sino_norm, proj_geom, vol_geom)
    if reco is None:
        raise RuntimeError("ASTRA is unavailable: the image-domain metrics cannot be computed.")
    mse = float(np.mean((reco - phantom) ** 2))
    return {
        "img_psnr": float(10.0 * np.log10(1.0 / max(mse, 1e-12))),
        "img_ssim": ssim(phantom, reco, data_range=1.0),
    }


@torch.no_grad()
def evaluate_models(models: dict, dataset, device, config: DBTGeometryConfig):
    geom = DBTGeometry.from_config(config)
    missing_views = ~geom.acquired_view_mask
    proj_geom, vol_geom = build_full_astra_geometries(config, config.image_shape)

    rows = []
    per_view_sq_err = {name: np.zeros(geom.num_views) for name in models}

    for i in tqdm(range(len(dataset)), desc="Evaluating samples"):
        incomplete, full, phantom = dataset[i]
        full_np = full[0].numpy()
        phantom_np = phantom[0].numpy()

        if i == 0:
            rows.append({"Sample": -1, "Model": "GroundTruthSinogram (FBP reference)",
                         **image_metrics(full_np, phantom_np, config, proj_geom, vol_geom)})

        for name, model in models.items():
            # Every model, learned or not, shares the same interface.
            pred = model(incomplete.unsqueeze(0).to(device))[0, 0].float().cpu().numpy()
            per_view_sq_err[name] += ((pred - full_np) ** 2).mean(axis=1)
            rows.append({
                "Sample": i, "Model": name,
                **sinogram_metrics(pred, full_np, missing_views),
                **image_metrics(pred, phantom_np, config, proj_geom, vol_geom),
            })

    per_view = pd.DataFrame({name: err / len(dataset) for name, err in per_view_sq_err.items()})
    per_view.insert(0, "angle_deg", np.rad2deg(geom.angles))
    per_view.insert(1, "acquired", geom.acquired_view_mask)
    return pd.DataFrame(rows), per_view


def main():
    parser = argparse.ArgumentParser(description="Evaluate all 2D models on the test split")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--out_dir", type=Path, default=PROJECT_ROOT / "outputs/evaluation")
    parser.add_argument("--num_samples", type=int, default=200, help="Size of the test split")
    parser.add_argument("--noise_level", type=float, default=1e5)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DBTGeometryConfig()
    dataset = SinogramCompletionDataset(args.num_samples, split="test", noise_level=args.noise_level, geometry_config=config)

    models, info, skipped = {}, {}, {}
    for name in args.models:
        try:
            model, is_nn = get_model(name, device, geometry_config=config)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            skipped[name] = str(exc).splitlines()[0]
            continue
        models[name] = model
        info[name] = sum(p.numel() for p in model.parameters()) if isinstance(model, torch.nn.Module) else 0

    for name, reason in skipped.items():
        print(f"[NOT EVALUATED] {name}: {reason}")
    if not models:
        print("No model could be loaded. Train them first with src_2D/train.py.")
        return

    results, per_view = evaluate_models(models, dataset, device, config)
    results.to_csv(args.out_dir / "model_comparison_metrics.csv", index=False)
    per_view.to_csv(args.out_dir / "per_view_mse.csv", index=False)

    metric_cols = ["mse_wedge", "psnr_wedge", "ssim_wedge", "ssim", "img_psnr", "img_ssim"]
    per_model = results[results["Sample"] >= 0].groupby("Model", sort=False)[metric_cols]
    mean, std = per_model.mean(), per_model.std()
    summary = pd.DataFrame({"# Params": [f"{info[m]:,}" for m in mean.index]}, index=mean.index)
    for col in metric_cols:
        summary[col] = [f"{mean.loc[m, col]:.4f} ± {std.loc[m, col]:.4f}" for m in mean.index]

    print(f"\nTest split: {len(dataset)} samples, sinogram data_range = 1.0, image metrics after FBP (ram-lak)")
    print(results[results["Sample"] < 0][["Model", "img_psnr", "img_ssim"]].to_string(index=False))
    print(summary.to_string())
    print(f"\nSaved to {args.out_dir}/model_comparison_metrics.csv and per_view_mse.csv")


if __name__ == "__main__":
    main()
