"""Self-describing checkpoints.

A checkpoint stores everything needed to rebuild the network it contains:

    {
        "format_version": 1,
        "model_name":     "SNN" | "GCN" | "UNet2D" | "UNet2dHLCC",
        "model_config":   constructor keyword arguments (JSON-serialisable),
        "geometry":       DBTGeometryConfig.to_dict(),
        "model_state":    state_dict,
        "meta":           free-form (epoch, metrics, CLI args, git commit, loss scales...),
    }

so that ``src_2D.models.factory.get_model`` never has to guess an architecture.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

FORMAT_VERSION = 1


def current_git_commit() -> Optional[str]:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except Exception:
        return None


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model_name: str,
    model_config: Dict[str, Any],
    geometry_config,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    model = getattr(model, "module", model)  # unwrap DDP / DataParallel
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "model_name": model_name,
        "model_config": dict(model_config),
        "geometry": geometry_config.to_dict(),
        "model_state": model.state_dict(),
        "meta": {"git_commit": current_git_commit(), **(meta or {})},
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)  # atomic: a killed job never leaves a truncated checkpoint


def load_checkpoint(path: Path, map_location: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    """Load a self-describing checkpoint. Raises on legacy (bare state_dict) files."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not (isinstance(payload, dict) and "model_state" in payload and "model_config" in payload):
        raise ValueError(
            f"{path} is a legacy checkpoint (bare state_dict, fan-beam era) without its "
            "architecture / geometry description. Retrain the model with src_2D/train.py."
        )
    return payload
