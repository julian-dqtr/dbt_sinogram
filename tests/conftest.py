"""Shared fixtures of the smoke / unit tests. Run with:  .venv/bin/python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The Tesla K80 is not supported by the cuDNN shipped with recent PyTorch builds.
torch.backends.cudnn.enabled = False

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.geometry.dbt_geometry_2d import DBTGeometry


@pytest.fixture(scope="session")
def config():
    return DBTGeometryConfig()


@pytest.fixture(scope="session")
def geom(config):
    return DBTGeometry.from_config(config)


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def clean_val_batch(config):
    """8 noiseless validation samples: (incomplete [8,1,V,D], full [8,1,V,D], phantom [8,1,H,W])."""
    dataset = SinogramCompletionDataset(8, split="val", noise_level=0.0, geometry_config=config)
    items = [dataset[i] for i in range(len(dataset))]
    return tuple(torch.stack([item[j] for item in items]) for j in range(3))


def analytic_disk_sinogram(geom, radius=20.0, center=(15.0, -10.0), density=0.5):
    """Exact parallel-beam sinogram of a uniform disk, averaged over each detector pixel.

    p(theta, s) = 2 * density * sqrt(r^2 - (s - c . u(theta))^2), u(theta) = (cos theta, -sin theta),
    integrated in closed form over every pixel (no quadrature error from the square-root edges).
    Its moments are known exactly: M0 = density * pi * r^2 and
    M1(theta) = M0 * (cx * cos(theta) - cy * sin(theta)).
    """
    n_det, ds = geom.det_col_count, geom.det_pixel_size
    edges = (np.arange(n_det + 1) - n_det / 2.0) * ds
    shift = center[0] * np.cos(geom.angles) - center[1] * np.sin(geom.angles)
    t = np.clip(edges[None, :] - shift[:, None], -radius, radius)
    # Antiderivative of 2 * sqrt(r^2 - t^2)
    primitive = t * np.sqrt(radius**2 - t**2) + radius**2 * np.arcsin(t / radius)
    return torch.from_numpy(density * np.diff(primitive, axis=1) / ds).float()
