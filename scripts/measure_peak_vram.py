#!/usr/bin/env python3
"""Peak GPU memory of a training step for the runs trained before train.py recorded it.

    python scripts/measure_peak_vram.py                 # every run of outputs/2d/checkpoints/ without peak_memory_gib
    python scripts/measure_peak_vram.py --runs GCN_L18  # selected runs

For each run, the architecture is read from its checkpoint and two training steps (forward, loss,
backward, gradient clipping, AdamW step: the optimiser state exists from the first step) are run on
one GPU with the per-GPU batch size of the protocol. The peak memory allocated by PyTorch is written
to results/tables/peak_vram.csv, read by the training cost table of results/model_analysis.ipynb.
Without DDP: the gradient buckets of DDP add about one copy of the gradients (a few MB here).
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.models.factory import CHECKPOINT_ROOT, build_model
from src_2D.models.SinoSheavesNN.physics_loss import AnnealedLoss
from src_2D.utils.checkpoint import load_checkpoint

torch.backends.cudnn.enabled = False  # as in train.py (K80)
OUT_CSV = PROJECT_ROOT / "results/tables/peak_vram.csv"


def peak_training_memory(model_name: str, model_config: dict, physics: bool, batch_size: int, device) -> float:
    """Peak memory (GiB) allocated by PyTorch over two training steps of a freshly built model."""
    config = DBTGeometryConfig()
    model, _ = build_model(model_name, config, **model_config)
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = AnnealedLoss(DBTGeometry.from_config(config)).to(device) if physics else None
    shape = (batch_size, 1, len(config.full_angles), config.det_col_count)
    incomplete, target = torch.rand(shape, device=device), torch.rand(shape, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(2):
        pred = model(incomplete)
        loss = loss_fn(pred[:, 0], target[:, 0], 0)[0] if physics else torch.nn.functional.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    del model, optimizer, pred, loss
    return peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=None, help="Checkpoint folders (default: every run without a recorded peak)")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        sys.exit("A GPU is needed: the point is to measure GPU memory.")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)

    runs = args.runs or sorted(
        p.parent.name for p in CHECKPOINT_ROOT.glob("*/training_stats.json")
        if "peak_memory_gib" not in json.loads(p.read_text())
    )
    rows = []
    for run in runs:
        payload = load_checkpoint(CHECKPOINT_ROOT / run / "best_model.pt", map_location="cpu")
        meta = payload.get("meta", {})
        batch_size = int(meta.get("args", {}).get("batch_size", 4))
        peak = peak_training_memory(payload["model_name"], payload["model_config"], bool(meta.get("physics", False)),
                                    batch_size, device)
        rows.append({"run": run, "model": payload["model_name"], "batch_size_per_gpu": batch_size,
                     "peak_memory_gib": round(peak, 3), "gpu": props.name,
                     "gpu_memory_gib": round(props.total_memory / 2**30, 2), "measured_on": date.today().isoformat()})
        print(f"{run:18s} batch {batch_size}: peak {peak:.2f} GiB on {props.name}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    if OUT_CSV.exists():  # keep the measurements of the other runs
        previous = pd.read_csv(OUT_CSV)
        table = pd.concat([previous[~previous["run"].isin(table["run"])], table]).sort_values("run")
    table.to_csv(OUT_CSV, index=False)
    print(f"Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
