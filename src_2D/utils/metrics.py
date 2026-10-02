"""Single source of truth for the sinogram metrics used by every training / evaluation script.

Conventions (to be stated in the thesis):

- Sinograms are compared in normalised units (raw line integrals in mm divided by
  ``DBTGeometryConfig.sino_norm``).
- ``DATA_RANGE`` is FIXED to 1.0 for PSNR and SSIM. A per-sample range would make the
  metrics depend on the phantom and is what made earlier numbers incomparable.
- "wedge" metrics are restricted to the missing views, the only place where a model
  actually predicts something: acquired views are copied by the data-consistency step and
  the zero background otherwise dominates whole-sinogram scores.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from skimage.metrics import structural_similarity

DATA_RANGE = 1.0


def ssim(target: np.ndarray, pred: np.ndarray, data_range: float = DATA_RANGE) -> float:
    """Scalar SSIM. Wraps ``structural_similarity``, whose return type depends on its flags."""
    return float(structural_similarity(target, pred, data_range=data_range))  # type: ignore[arg-type]


def ssim_map(target: np.ndarray, pred: np.ndarray, data_range: float = DATA_RANGE) -> np.ndarray:
    """Per-pixel SSIM map (same wrapper, ``full=True`` branch)."""
    return np.asarray(structural_similarity(target, pred, data_range=data_range, full=True)[1])


def _psnr(mse: float) -> float:
    return float(10.0 * np.log10(DATA_RANGE**2 / max(mse, 1e-12)))


def sinogram_metrics(pred: np.ndarray, target: np.ndarray, missing_views: np.ndarray) -> Dict[str, float]:
    """Metrics of one predicted sinogram.

    Args:
        pred, target: [V, D] arrays in normalised units.
        missing_views: boolean [V], True for the views that had to be completed.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    missing_views = np.asarray(missing_views, dtype=bool)

    sq_err = (pred - target) ** 2
    mse_full = float(sq_err.mean())
    mse_wedge = float(sq_err[missing_views].mean())

    per_pixel_ssim = ssim_map(target, pred)

    return {
        "mse": mse_full,
        "psnr": _psnr(mse_full),
        "ssim": float(per_pixel_ssim.mean()),
        "mse_wedge": mse_wedge,
        "psnr_wedge": _psnr(mse_wedge),
        "ssim_wedge": float(per_pixel_ssim[missing_views].mean()),
    }


METRIC_KEYS = ("mse", "psnr", "ssim", "mse_wedge", "psnr_wedge", "ssim_wedge")
