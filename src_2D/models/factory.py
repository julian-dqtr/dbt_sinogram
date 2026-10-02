"""Single entry point to build, save-path and reload every model of the protocol.

Learned models (all share the ``completed = model(incomplete)`` interface):

    "UNet2D"      U-Net, MSE loss
    "UNet2dHLCC"  same U-Net, MSE + HLCC loss
    "GCN"         view-graph GCN baseline (no rotation)
    "SNN"         SinoSheavesNN (hard-coded SO(2) restriction maps)

Graph models accept a depth suffix selecting the run: "GCN_L6", "SNN_L12", "SNN_L18"...

Non-learned baselines: "ZeroFilling", "LinearInterp".
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.utils.checkpoint import load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs/2d/checkpoints"

GRAPH_MODELS = ("GCN", "SNN")
UNET_MODELS = ("UNet2D", "UNet2dHLCC")
LEARNED_MODELS = UNET_MODELS + GRAPH_MODELS
BASELINES = ("ZeroFilling", "LinearInterp")
LEGACY_MODELS = ("GLM", "UNet2dRNO")

DEFAULT_MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "UNet2D": {"filters": 32},
    "UNet2dHLCC": {"filters": 32},
    "GCN": {"num_stalks": 32, "num_layers": 6, "k": 12, "sigma_deg": 5.0},
    "SNN": {"num_stalks": 32, "num_layers": 6, "k": 12, "sigma_deg": 5.0},
}
_ALIASES = {"SinoSheavesNN": "SNN", "GNN": "GCN"}


def parse_model_name(name: str) -> Tuple[str, Dict[str, Any]]:
    """Split "SNN_L12" into ("SNN", {"num_layers": 12}). Plain names return no override."""
    match = re.fullmatch(r"(?P<base>[A-Za-z0-9]+?)(?:_L(?P<layers>\d+))?", name)
    if match is None:
        raise ValueError(f"Cannot parse model name: {name}")
    base = _ALIASES.get(match["base"], match["base"])
    overrides: Dict[str, Any] = {}
    if match["layers"] is not None:
        if base not in GRAPH_MODELS:
            raise ValueError(f"The depth suffix _L<n> only applies to {GRAPH_MODELS}, got {name}")
        overrides["num_layers"] = int(match["layers"])
    return base, overrides


def default_run_name(model_name: str, model_config: Dict[str, Any]) -> str:
    """Name of the checkpoint folder of a run (one folder per depth for the graph models)."""
    if model_name in GRAPH_MODELS:
        return f"{model_name}_L{model_config['num_layers']}"
    return model_name


def build_model(model_name: str, geometry_config: Optional[DBTGeometryConfig] = None, **model_config):
    """Instantiate a fresh (untrained) learned model. Returns (model, full_model_config)."""
    base, overrides = parse_model_name(model_name)
    if base in LEGACY_MODELS:
        raise ValueError(f"{base} is a legacy fan-beam model and is not part of the parallel-beam protocol.")
    if base not in LEARNED_MODELS:
        raise ValueError(f"Unknown learned model: {model_name}. Available: {LEARNED_MODELS}")

    for key, value in overrides.items():
        if key in model_config and model_config[key] != value:
            raise ValueError(f"'{model_name}' implies {key}={value} but {key}={model_config[key]} was requested.")
    config = {**DEFAULT_MODEL_CONFIG[base], **model_config, **overrides}
    geometry_config = geometry_config or DBTGeometryConfig()

    if base == "UNet2D":
        from src_2D.models.Unet2D.unet_2d import SinogramUNet as cls
    elif base == "UNet2dHLCC":
        from src_2D.models.Unet2dHLCC.unet_hlcc import Unet2dHLCC as cls
    elif base == "GCN":
        from src_2D.models.SinoSheavesNN.snn_model import SinoGCN as cls
    else:
        from src_2D.models.SinoSheavesNN.snn_model import SinoSheafNet as cls

    return cls(geometry_config=geometry_config, **config), config


def get_model(
    model_name: str,
    device: torch.device,
    load_weights: bool = True,
    checkpoint_path: Optional[Path] = None,
    geometry_config: Optional[DBTGeometryConfig] = None,
    **kwargs,
):
    """
    Instantiate a model and (by default) load its trained weights.

    The architecture and the geometry are read from the checkpoint itself, the load is
    strict, and any failure RAISES: an evaluation must never silently score a randomly
    initialised network. A checkpoint trained on another geometry than ``geometry_config``
    (default: the current ``DBTGeometryConfig()``) is rejected as well.

    Returns:
        (model_or_callable, is_nn)
    """
    base, overrides = parse_model_name(model_name)

    if base == "ZeroFilling":
        from src_2D.models.baselines import zero_filling
        return zero_filling, False
    if base == "LinearInterp":
        from src_2D.models.baselines import linear_interpolation
        return linear_interpolation, False

    geometry_config = geometry_config or DBTGeometryConfig()

    if not load_weights:
        model, _ = build_model(model_name, geometry_config, **kwargs)
        return model.to(device).eval(), True

    if checkpoint_path is None:
        config = {**DEFAULT_MODEL_CONFIG.get(base, {}), **kwargs, **overrides}
        checkpoint_path = CHECKPOINT_ROOT / default_run_name(base, config) / "best_model.pt"
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint for {model_name} at {checkpoint_path}")

    payload = load_checkpoint(checkpoint_path, map_location=device)
    if payload["model_name"] != base:
        raise ValueError(f"{checkpoint_path} contains a {payload['model_name']} model, expected {base}")

    for key, value in overrides.items():
        if payload["model_config"].get(key) != value:
            raise ValueError(f"'{model_name}' implies {key}={value} but {checkpoint_path} has {key}={payload['model_config'].get(key)}")

    checkpoint_geometry = DBTGeometryConfig.from_dict(payload["geometry"])
    if checkpoint_geometry != geometry_config:
        raise ValueError(
            f"{checkpoint_path} was trained on a different geometry:\n  checkpoint: {checkpoint_geometry}\n"
            f"  requested:  {geometry_config}"
        )

    model, _ = build_model(base, checkpoint_geometry, **payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    print(f"[+] Loaded {model_name} from {checkpoint_path} (config: {payload['model_config']})")
    return model.to(device).eval(), True
