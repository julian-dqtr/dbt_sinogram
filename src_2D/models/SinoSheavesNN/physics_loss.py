import torch
import torch.nn as nn


class HelgasonLudwigLoss(nn.Module):
    """
    Computes the 0th and 1st Helgason-Ludwig moment consistency losses for a sinogram.

    The HLCC state that:
    - 0th moment M(θ) = Σ_s p(s,θ)·ds should be *constant* across all views.
    - 1st moment C(θ) = Σ_s s·p(s,θ)·ds should vary sinusoidally as A·cos(θ) + B·sin(θ).

    Input sinogram shape: [num_views, num_detectors] or [B, num_views, num_detectors].
    """

    def __init__(self, geom):
        super().__init__()
        self.num_views = geom.num_views
        self.num_det = geom.det_col_count
        self.ds = geom.det_pixel_size

        # Detector positions s_d = (d - num_det/2 + 0.5) * ds
        d_idx = torch.arange(self.num_det, dtype=torch.float32)
        s_positions = (d_idx - self.num_det / 2.0 + 0.5) * self.ds
        self.register_buffer("s_positions", s_positions)

        # Pre-compute (I - P) for the 1st moment, where P is the projection onto
        # the span of [cos(θ), sin(θ)]. Residual (I-P)·C(θ) = 0 iff C(θ) is sinusoidal.
        angles = torch.tensor(geom.angles, dtype=torch.float32)
        cos_t = torch.cos(angles).unsqueeze(1)
        sin_t = torch.sin(angles).unsqueeze(1)
        Phi = torch.cat([cos_t, sin_t], dim=1)  # [V, 2]
        Phi_T_Phi_inv = torch.linalg.inv(Phi.t() @ Phi)
        P = Phi @ Phi_T_Phi_inv @ Phi.t()       # [V, V]
        self.register_buffer("I_minus_P", torch.eye(self.num_views) - P)

        # Scale constants for normalisation (updated by calibrate())
        self.register_buffer("scale_m0", torch.ones(1))
        self.register_buffer("scale_m1", torch.ones(1))

    # ------------------------------------------------------------------
    # Calibration: call this on the first few batches to set the scales
    # ------------------------------------------------------------------
    @torch.no_grad()
    def calibrate(self, sinogram: torch.Tensor) -> None:
        """
        Compute the raw physics losses on a reference sinogram and store their
        magnitudes as normalisation constants. Call once before training begins
        so that loss_m0 and loss_m1 are in the same ballpark as the MSE.
        """
        m0, m1 = self._raw_losses(sinogram)
        self.scale_m0 = torch.clamp(m0.detach(), min=1e-8)
        self.scale_m1 = torch.clamp(m1.detach(), min=1e-8)

    # ------------------------------------------------------------------
    def _raw_losses(self, sinogram: torch.Tensor):
        if sinogram.dim() == 2:
            sinogram = sinogram.unsqueeze(0)  # [1, V, D]

        # 0th moment: should be constant over views
        M_theta = sinogram.sum(dim=2) * self.ds  # [B, V]
        loss_m0 = torch.var(M_theta, dim=1).mean()

        # 1st moment: should lie in span{cos,sin}
        C_theta = (sinogram @ self.s_positions) * self.ds  # [B, V]
        residual = C_theta @ self.I_minus_P.t()            # [B, V]
        loss_m1 = (residual ** 2).sum(dim=1).mean() / self.num_views

        return loss_m0, loss_m1

    def forward(self, sinogram: torch.Tensor):
        loss_m0, loss_m1 = self._raw_losses(sinogram)
        # Normalised so each component is ~O(1) relative to the calibrated reference
        return loss_m0 / self.scale_m0, loss_m1 / self.scale_m1


class AnnealedLoss(nn.Module):
    """
    Combines MSE data-fidelity loss with normalised HLCC physics losses.

    Schedule:
        total = MSE  +  alpha(epoch) * (λ_m0 * L_m0_norm + λ_m1 * L_m1_norm)

    where alpha ramps from 0 → 1 over `anneal_epochs` using a cosine warm-up,
    giving the network time to learn the basic reconstruction before physics
    constraints are introduced.

    The physics losses are *normalised* at calibration time so they naturally
    sit at the same scale as the MSE, making λ_m0 and λ_m1 true relative weights
    rather than order-of-magnitude tuning knobs.
    """

    def __init__(self, geom, lambda_m0: float = 1.0, lambda_m1: float = 1.0, anneal_epochs: int = 50):
        super().__init__()
        self.data_loss_fn = nn.MSELoss()
        self.physics_loss_fn = HelgasonLudwigLoss(geom)
        self.lambda_m0 = lambda_m0
        self.lambda_m1 = lambda_m1
        self.anneal_epochs = anneal_epochs
        self._calibrated = False

    # ------------------------------------------------------------------
    def calibrate(self, sinogram: torch.Tensor) -> None:
        """Delegate to HelgasonLudwigLoss.calibrate(); call before training."""
        self.physics_loss_fn.calibrate(sinogram)
        self._calibrated = True

    # ------------------------------------------------------------------
    def forward(self, pred_sino: torch.Tensor, true_sino: torch.Tensor, current_epoch: int):
        loss_data = self.data_loss_fn(pred_sino, true_sino)

        loss_m0_norm, loss_m1_norm = self.physics_loss_fn(pred_sino)

        # Cosine warm-up: smoother than a linear ramp, avoids abrupt gradient changes
        if self.anneal_epochs > 0:
            t = min(1.0, current_epoch / self.anneal_epochs)
            alpha = 0.5 * (1.0 - torch.cos(torch.tensor(t * 3.14159265)).item())
        else:
            alpha = 1.0

        physics_term = alpha * (self.lambda_m0 * loss_m0_norm + self.lambda_m1 * loss_m1_norm)
        total_loss = loss_data + physics_term

        return total_loss, loss_data, loss_m0_norm, loss_m1_norm
