"""Helgason-Ludwig consistency conditions (HLCC) in the Chebyshev form of Huang et al. (2017).

Y. Huang et al., "Restoration of missing data in limited angle tomography based on
Helgason-Ludwig consistency conditions", Biomed. Phys. Eng. Express, 2017.

For a parallel-beam sinogram p(theta, s) whose detector covers |s| <= S, the MOMENT CURVE of order n is

    a_n(theta) = int p(theta, s) U_n(s / S) ds / S        (U_n: Chebyshev polynomial of the second kind).

The HLCC state that a_n is a trigonometric polynomial of degree n with the parity of n:

    a_n in H_n = span{cos(m theta), sin(m theta) : 0 <= m <= n, m = n mod 2},      dim H_n = n + 1.

(a_0 * S is the total mass, constant over the views; a_1 * S^2 / 2 is the first moment, a pure
sinusoid; orders 2 and 3 contain the angular frequencies {0, 2} and {1, 3}.)

The repository uses this fact in two ways:

- CONSISTENCY PENALTY (``models/SinoSheavesNN/physics_loss.py``): the distance of a_n to H_n over all
  the views, "is this sinogram the projection of SOME object?".
- MOMENT REGRESSION (``HLCCMomentProjection`` below): the n + 1 coefficients of a_n are fitted on the
  ACQUIRED views and the curve is evaluated on the missing ones, "what are the moments of THIS
  object in the views that were not measured?". The completed sinogram is then corrected so that its
  missing views have exactly these moments.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn


def detector_coordinate(geom) -> torch.Tensor:
    """Normalised detector coordinate x = s / S in (-1, 1), [D] float64 (S: half width of the detector)."""
    num_det = geom.det_col_count
    return (torch.arange(num_det, dtype=torch.float64) - num_det / 2.0 + 0.5) * (2.0 / num_det)


def chebyshev_u(x: torch.Tensor, max_order: int) -> torch.Tensor:
    """Chebyshev polynomials of the second kind U_0..U_max_order evaluated at x: [len(x), max_order + 1]."""
    polys = [torch.ones_like(x), 2.0 * x]
    for _ in range(2, max_order + 1):
        polys.append(2.0 * x * polys[-1] - polys[-2])  # U_{n+1} = 2 x U_n - U_{n-1}
    return torch.stack(polys[: max_order + 1], dim=1)


def moment_operator(geom, max_order: int) -> torch.Tensor:
    """[D, max_order + 1] float64 matrix such that ``sinogram @ moment_operator`` is a_n(theta)."""
    x = detector_coordinate(geom)
    return chebyshev_u(x, max_order) * (2.0 / geom.det_col_count)  # dx = ds / S = 2 / D


def harmonic_basis(angles, order: int) -> torch.Tensor:
    """Basis of H_order sampled at ``angles`` (radians): [V, order + 1] float64."""
    theta = torch.as_tensor(np.asarray(angles), dtype=torch.float64)
    columns = []
    for m in range(order % 2, order + 1, 2):
        columns.append(torch.cos(m * theta))
        if m > 0:
            columns.append(torch.sin(m * theta))
    return torch.stack(columns, dim=1)


def residual_projectors(angles, max_order: int) -> torch.Tensor:
    """[max_order + 1, V, V] float64: I - P_n, with P_n the orthogonal projector onto H_n."""
    num_views = len(angles)
    eye = torch.eye(num_views, dtype=torch.float64)
    projectors = []
    for n in range(max_order + 1):
        basis = harmonic_basis(angles, n)
        projectors.append(eye - basis @ torch.linalg.pinv(basis))
    return torch.stack(projectors)


class HLCCMomentProjection(nn.Module):
    """Moment regression of Huang et al. (2017), used as a post-processing of a completed sinogram.

    For every order n <= max_order:

    1. a_n is computed on the ACQUIRED views of the measured (noisy) sinogram;
    2. its n + 1 harmonic coefficients are fitted by least squares on these views and the curve is
       evaluated on every view (eq. 13-14 of the paper);
    3. each MISSING view of the completed sinogram receives the correction
       sum_n c_n(theta) W(s) U_n(s),  W = sqrt(1 - (s / S)^2)   (inverse Chebyshev transform, eq. 7),
       whose coefficients c_n are chosen so that the moments of orders 0..max_order of the corrected
       view are exactly the regressed ones;
    4. the correction is kept only INSIDE THE SHADOW of the object, i.e. where the completed view
       exceeds ``support_threshold`` times its maximum: a sinogram is zero outside that shadow and
       must stay so. Without this step the correction leaks into the zero background, which costs
       0.03 of wedge SSIM on the U-Net for the same MSE (docs/hlcc_projection.md). The moments are
       then matched only approximately; ``support_threshold=None`` skips the step (exact moments).

    The acquired views are never modified and the module has no trainable parameter. The regression
    from a +/-25 degree window is only stable up to order 4 (scripts/hlcc_order_study.py), hence the
    default.

    Shapes: [..., V, D] for both arguments (e.g. [V, D] or [B, 1, V, D]); computed in float64.
    """

    moment_op: torch.Tensor
    synthesis: torch.Tensor
    extrapolation: torch.Tensor
    acquired: torch.Tensor

    def __init__(self, geom, max_order: int = 4, support_threshold: Optional[float] = 0.03):
        super().__init__()
        if max_order < 0:
            raise ValueError(f"max_order must be >= 0, got {max_order}")
        self.max_order = max_order
        self.support_threshold = support_threshold

        x = detector_coordinate(geom)
        moment_op = moment_operator(geom, max_order)                                # [D, K+1]
        atoms = torch.sqrt(1.0 - x**2).unsqueeze(1) * chebyshev_u(x, max_order)     # W U_n, [D, K+1]
        # Moments of the atoms: on a discrete detector W U_n is only approximately orthogonal to
        # U_m, so the coefficients are obtained by solving this small system (exact moment matching).
        gram = moment_op.T @ atoms                                                  # [K+1, K+1]
        synthesis = atoms @ torch.linalg.inv(gram)                                  # [D, K+1]

        acquired = torch.from_numpy(np.asarray(geom.acquired_view_mask, dtype=bool))
        extrapolation = []
        for n in range(max_order + 1):
            basis = harmonic_basis(geom.angles, n)                                  # [V, n+1]
            extrapolation.append(basis @ torch.linalg.pinv(basis[acquired]))        # [V, V_acquired]

        self.register_buffer("moment_op", moment_op, persistent=False)
        self.register_buffer("synthesis", synthesis, persistent=False)
        self.register_buffer("extrapolation", torch.stack(extrapolation), persistent=False)
        self.register_buffer("acquired", acquired, persistent=False)

    def moments(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Moment curves a_0..a_K of a sinogram: [..., V, D] -> [..., V, K+1] (float64)."""
        return sinogram.double() @ self.moment_op

    def regressed_moments(self, measured: torch.Tensor) -> torch.Tensor:
        """Moment curves on ALL the views, regressed from the acquired views of ``measured``."""
        acquired_moments = self.moments(measured[..., self.acquired, :])             # [..., V_acq, K+1]
        return torch.einsum("nva,...an->...vn", self.extrapolation, acquired_moments)

    def forward(self, completed: torch.Tensor, measured: torch.Tensor) -> torch.Tensor:
        delta = self.regressed_moments(measured) - self.moments(completed)           # [..., V, K+1]
        correction = delta @ self.synthesis.T                                        # [..., V, D]
        if self.support_threshold is not None:
            signal = completed.double().clamp_min(0.0)
            correction = correction * (signal > self.support_threshold * signal.amax(dim=-1, keepdim=True))
        missing = (~self.acquired).to(correction.dtype).unsqueeze(1)                 # [V, 1]
        return (completed.double() + correction * missing).to(completed.dtype)


class HLCCProjectedModel(nn.Module):
    """A completion model followed by ``HLCCMomentProjection``: same interface, no extra parameter."""

    def __init__(self, model: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]], geom, max_order: int = 4):
        super().__init__()
        self.model = model
        self.projection = HLCCMomentProjection(geom, max_order)

    def forward(self, incomplete: torch.Tensor) -> torch.Tensor:
        return self.projection(self.model(incomplete), incomplete)
