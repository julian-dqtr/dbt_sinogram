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

from typing import Dict, Optional, Tuple

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


def _wedge_dirichlet_energies(sinogram: np.ndarray, missing_views: np.ndarray) -> Tuple[float, float]:
    """Dirichlet energies of the wedge of a [V, D] sinogram, along the angle and along the detector.

    Along the angle, only pairs of consecutive MISSING views count: the pairs that straddle the
    border of the acquired window (copied by data consistency) are left out.
    """
    consecutive_missing = missing_views[1:] & missing_views[:-1]
    d_angle = np.diff(sinogram, axis=0)[consecutive_missing]
    d_detector = np.diff(sinogram[missing_views], axis=1)
    return float(np.sum(d_angle**2)), float(np.sum(d_detector**2))


def smoothness_metrics(pred: np.ndarray, target: np.ndarray, missing_views: np.ndarray) -> Dict[str, float]:
    """Oversmoothing indicators of one completed sinogram, on the missing wedge.

    Ratios of Dirichlet energies prediction / ground truth, along the angle and along the detector:
    1 means as much variation as the ground truth, < 1 smoother (a washed-out, "grey" wedge),
    > 1 rougher (noise, artefacts).
    """
    missing_views = np.asarray(missing_views, dtype=bool)
    pred_angle, pred_detector = _wedge_dirichlet_energies(np.asarray(pred, dtype=np.float64), missing_views)
    true_angle, true_detector = _wedge_dirichlet_energies(np.asarray(target, dtype=np.float64), missing_views)
    return {
        "dirichlet_angle_ratio": pred_angle / max(true_angle, 1e-12),
        "dirichlet_detector_ratio": pred_detector / max(true_detector, 1e-12),
    }


class PerViewMoments:
    """Streaming inter-sample variance of [V, D] sinograms, reduced per view.

    ``per_view_variance()`` is the population variance over the samples (1/n), averaged over the
    detector. For the ground truth, it is exactly the lowest per-view MSE that a prediction
    independent of the measurements (the same for every sample) can reach on these samples: the
    floor of a graph model beyond its angular reach.

    The sums are shifted by the first sample, so that a quantity identical for every sample gets a
    variance of exactly 0 (not a rounding residual).
    """

    def __init__(self) -> None:
        self.count = 0
        self._shift: Optional[np.ndarray] = None
        self._sum: Optional[np.ndarray] = None
        self._sum_sq: Optional[np.ndarray] = None

    def update(self, sinogram: np.ndarray) -> None:
        x = np.asarray(sinogram, dtype=np.float64)
        if self._shift is None:
            self._shift = x.copy()
            self._sum = np.zeros_like(x)
            self._sum_sq = np.zeros_like(x)
        delta = x - self._shift
        self._sum += delta
        self._sum_sq += delta**2
        self.count += 1

    def per_view_variance(self) -> np.ndarray:
        if self._sum is None or self._sum_sq is None:
            raise ValueError("No sample was accumulated.")
        mean = self._sum / self.count
        variance = np.maximum(self._sum_sq / self.count - mean**2, 0.0)
        return variance.mean(axis=1)
