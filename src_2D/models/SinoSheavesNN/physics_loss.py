import math

import torch
import torch.nn as nn

from src_2D.utils.hlcc import moment_operator, residual_projectors


class HelgasonLudwigLoss(nn.Module):
    """
    Helgason-Ludwig consistency penalty of orders 0..``max_order`` for a PARALLEL-BEAM sinogram.

    The moment curve of order n (Chebyshev form, see ``src_2D/utils/hlcc.py``) is

        a_n(θ) = Σ_s p(θ, s) · U_n(s / S) · ds / S

    and the HLCC state that a_n lies in H_n = span{cos(mθ), sin(mθ) : m ≤ n, m ≡ n mod 2}. The raw
    loss of order n is the mean squared part of a_n OUTSIDE H_n, L_n = ‖(I − P_n) a_n‖² / V:

    - order 0: a_0 · S is the total mass, which must be *constant* across the views, so L_0 is the
      (population) variance of a_0 over the views;
    - order 1: a_1 · S² / 2 is the first moment (mass times the projection of the centre of mass on
      the detector axis), which must vary as A·cos(θ) + B·sin(θ);
    - orders 2 and 3: second and third moments, angular frequencies {0, 2} and {1, 3}.

    ``max_order = 1`` is the loss used by the first UNet2dHLCC runs (up to constant factors that the
    calibration removes). These conditions are only valid for parallel-beam data:
    tests/test_physics_loss.py checks that the ground truth of the configured geometry satisfies them.

    Input sinogram shape: [num_views, num_detectors] or [B, num_views, num_detectors].
    """

    # Buffers registered in __init__, declared here so static checkers know their type.
    moment_op: torch.Tensor
    residual_projectors: torch.Tensor
    scales: torch.Tensor

    def __init__(self, geom, max_order: int = 1):
        super().__init__()
        if max_order < 0:
            raise ValueError(f"max_order must be >= 0, got {max_order}")
        self.max_order = max_order
        self.num_views = geom.num_views
        self.half_width = geom.det_col_count * geom.det_pixel_size / 2.0  # S

        self.register_buffer("moment_op", moment_operator(geom, max_order).float())                  # [D, K+1]
        self.register_buffer("residual_projectors", residual_projectors(geom.angles, max_order).float())  # [K+1, V, V]
        # Scale constants of the normalisation, one per order (updated by calibrate())
        self.register_buffer("scales", torch.ones(max_order + 1))

    # ------------------------------------------------------------------
    # Calibration: call this on the first few batches to set the scales
    # ------------------------------------------------------------------
    @torch.no_grad()
    def calibrate(self, sinogram: torch.Tensor) -> None:
        """
        Compute the raw physics losses on a reference sinogram and store their
        magnitudes as normalisation constants.

        Calibrate on the ZERO-FILLED (incomplete) sinograms: the normalised loss then reads
        as "fraction of the inconsistency of the naive input" (1.0 for zero-filling, ~0 for
        the ground truth). Never calibrate on the ground truth, whose loss is ~0.
        """
        self.scales.copy_(torch.clamp(self.raw_losses(sinogram).detach(), min=1e-12))

    # ------------------------------------------------------------------
    def moments(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Moment curves a_0..a_K of a sinogram: [B, V, K+1] (a [V, D] input counts as B = 1)."""
        if sinogram.dim() == 2:
            sinogram = sinogram.unsqueeze(0)  # [1, V, D]
        return sinogram @ self.moment_op

    def _residuals(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Part of every moment curve outside its harmonic space: [B, K+1, V]."""
        return torch.einsum("nuv,bvn->bnu", self.residual_projectors, self.moments(sinogram))

    def raw_losses(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Un-normalised loss of every order, averaged over the batch: [K+1]."""
        return (self._residuals(sinogram) ** 2).mean(dim=(0, 2))

    @torch.no_grad()
    def relative_residuals(self, sinogram: torch.Tensor) -> torch.Tensor:
        """‖(I − P_n) a_n‖² / ‖a_n‖² per sample and order, [B, K+1]: a scale-free consistency metric
        (0 for a consistent sinogram). Computed in float64: the ground truth is ~1e-6."""
        if sinogram.dim() == 2:
            sinogram = sinogram.unsqueeze(0)
        moments = sinogram.double() @ self.moment_op.double()                       # [B, V, K+1]
        residuals = torch.einsum("nuv,bvn->bnu", self.residual_projectors.double(), moments)
        return (residuals**2).sum(dim=2) / (moments**2).sum(dim=1).clamp_min(1e-30)

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        # Normalised so each order is ~O(1) relative to the calibrated reference
        return self.raw_losses(sinogram) / self.scales


class AnnealedLoss(nn.Module):
    """
    Combines MSE data-fidelity loss with normalised HLCC physics losses.

    Schedule:
        total = MSE  +  alpha(epoch) * Σ_n λ_n * L_n_norm          (n = 0..max_order)

    where alpha ramps from 0 → 1 over `anneal_epochs` using a cosine warm-up,
    giving the network time to learn the basic reconstruction before physics
    constraints are introduced. λ_0 = ``lambda_m0``, λ_1 = ``lambda_m1`` and every order n ≥ 2
    shares ``lambda_high``.

    The λ are NOT relative weights with respect to the MSE. Only the physics
    losses are normalised, by their value on the zero-filled input (see
    ``HelgasonLudwigLoss.calibrate``): each L_norm is ~1 for zero-filling and ~0 for the
    ground truth. The MSE stays in raw normalised-sinogram units, where zero-filling
    already scores only ~2e-2. At equal relative progress, the physics term therefore
    weighs about λ / 2e-2 = 50 λ times the data term. Measured on the U-Net (orders 0-1): with
    λ = 0.1 the norm of the physics gradient is 26 to 52 times the one of the MSE gradient and the
    run under-fits (train MSE 11 times higher after 200 epochs); the ratio is proportional to λ,
    i.e. ~0.3 for the default λ = 1e-3. optuna_search.py searches λ in [1e-5, 1e-2].

    With a clean, fully supervised target, the penalty does not move the optimum: the
    ground truth already satisfies the HLCC (normalised residual ~1e-5, see
    tests/test_physics_loss.py). It only reshapes the optimisation path, i.e. it can act
    as a regulariser at best.
    """

    lambdas: torch.Tensor

    def __init__(self, geom, lambda_m0: float = 1e-3, lambda_m1: float = 1e-3, anneal_epochs: int = 20,
                 lambda_high: float = 1e-3, max_order: int = 1):
        super().__init__()
        self.data_loss_fn = nn.MSELoss()
        self.physics_loss_fn = HelgasonLudwigLoss(geom, max_order)
        lambdas = ([lambda_m0, lambda_m1] + [lambda_high] * max(0, max_order - 1))[: max_order + 1]
        self.register_buffer("lambdas", torch.tensor(lambdas, dtype=torch.float32))
        self.anneal_epochs = anneal_epochs
        self._calibrated = False

    # ------------------------------------------------------------------
    def calibrate(self, sinogram: torch.Tensor) -> None:
        """Delegate to HelgasonLudwigLoss.calibrate(); call before training."""
        self.physics_loss_fn.calibrate(sinogram)
        self._calibrated = True

    # ------------------------------------------------------------------
    def alpha(self, completed_epochs: int) -> float:
        """Cosine warm-up of the physics weight: 0 before the first epoch, 1 after ``anneal_epochs``."""
        if self.anneal_epochs <= 0:
            return 1.0
        t = min(1.0, max(0.0, completed_epochs / self.anneal_epochs))
        return 0.5 * (1.0 - math.cos(math.pi * t))

    def forward(self, pred_sino: torch.Tensor, true_sino: torch.Tensor, current_epoch: int):
        """``current_epoch`` is the number of COMPLETED epochs (0 during the first one).

        Returns (total loss, MSE, normalised physics losses [max_order + 1]).
        """
        loss_data = self.data_loss_fn(pred_sino, true_sino)
        physics_losses = self.physics_loss_fn(pred_sino)

        physics_term = self.alpha(current_epoch) * (self.lambdas * physics_losses).sum()
        total_loss = loss_data + physics_term

        return total_loss, loss_data, physics_losses
