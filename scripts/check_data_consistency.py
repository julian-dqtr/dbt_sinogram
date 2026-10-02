#!/usr/bin/env python3
"""Are the acquired views of a completed sinogram exactly the measured ones?

    .venv/bin/python scripts/check_data_consistency.py                          # UNet2D, UNet2dHLCC
    .venv/bin/python scripts/check_data_consistency.py --models UNet2D_N200 UNet2D_N200_P4 --num_samples 200

For every model, the acquired views (+/-25 degrees, 51 views) of the output are subtracted from the
same views of the limited-angle input, on the test split. With the hard data consistency (the default
since 2026-09-30) the difference is exactly 0. The last columns show the same outputs with the LEGACY
soft mask (a taper over the 4 border views on each side of the window), which is what the runs trained
before that date produced.

Note that the input is NOISY while the ground truth is clean: the acquired views of the output are
equal to the input, not to the ground truth.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.models.factory import get_model
from src_2D.utils.evaluation import get_data_consistency_mask

torch.backends.cudnn.enabled = False  # Tesla K80, as in evaluate_all.py
LEGACY_BLEND_WIDTH_DEG = 5.0


def acquired_difference(output: torch.Tensor, incomplete: torch.Tensor, acquired: np.ndarray) -> dict:
    """Difference output - input restricted to the acquired views: [N, 1, V, D] tensors."""
    difference = (output - incomplete)[:, 0, acquired].abs()          # [N, V_acquired, D]
    per_view = difference.amax(dim=(0, 2))
    return {"max |output - input|": difference.max().item(), "mean |output - input|": difference.mean().item(),
            "views that differ": f"{int((per_view > 0).sum())} / {int(acquired.sum())}"}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Difference between the measured views and the same views of the model output")
    parser.add_argument("--models", nargs="+", default=["UNet2D", "UNet2dHLCC"])
    parser.add_argument("--num_samples", type=int, default=100, help="Size of the test split")
    parser.add_argument("--noise_level", type=float, default=1e5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DBTGeometryConfig()
    geom = DBTGeometry.from_config(config)
    acquired = np.asarray(geom.acquired_view_mask, dtype=bool)
    dataset = SinogramCompletionDataset(args.num_samples, split="test", noise_level=args.noise_level, geometry_config=config)
    samples = [dataset[i] for i in range(len(dataset))]
    incomplete = torch.stack([s[0] for s in samples])  # [N, 1, V, D], noisy measurements, zero in the missing views
    full = torch.stack([s[1] for s in samples])        # [N, 1, V, D], clean target

    rows = []
    for name in args.models:
        try:
            model, _ = get_model(name, device, geometry_config=config)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"[NOT EVALUATED] {name}: {str(exc).splitlines()[0]}")
            continue

        def predict() -> torch.Tensor:
            return torch.cat([model(incomplete[i:i + 25].to(device)).float().cpu() for i in range(0, len(incomplete), 25)])

        row = {"model": name, **acquired_difference(predict(), incomplete, acquired)}
        # The same network with the mask of the runs trained before 2026-09-30.
        network = getattr(model, "model", model)  # "_P<k>" models wrap the network
        if hasattr(network, "dc_mask"):
            network.dc_mask = get_data_consistency_mask(geom, device, LEGACY_BLEND_WIDTH_DEG).to(network.dc_mask.dtype)
            legacy = acquired_difference(predict(), incomplete, acquired)
            row.update({f"legacy soft mask: {key}": value for key, value in legacy.items() if key != "mean |output - input|"})
        rows.append(row)

    signal = incomplete[:, 0, acquired]
    noise = (incomplete - full)[:, 0, acquired].abs()
    print(f"\n{len(dataset)} test samples, {int(acquired.sum())} acquired views out of {geom.num_views}. "
          f"Measured values: mean {signal.mean():.3f}, max {signal.max():.3f}.")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.3e}"))
    print(f"\nFor scale, the noise of the input itself (|input - clean ground truth| on the acquired views): "
          f"mean {noise.mean():.3e}, max {noise.max():.3e}.")


if __name__ == "__main__":
    main()
