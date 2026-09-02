import torch
import torch.nn as nn
import numpy as np

class HelgasonLudwigLoss(nn.Module):
    """
    Computes the 0th and 1st Helgason-Ludwig moment consistency losses for a sinogram.
    Assumes the sinogram is of shape [batch_size, num_views, num_detectors]
    or [num_views, num_detectors].
    """
    def __init__(self, geom):
        super(HelgasonLudwigLoss, self).__init__()
        self.geom = geom
        self.num_views = geom.num_views
        self.num_det = geom.det_col_count
        self.ds = geom.det_pixel_size
        
        # Array of detector positions 's'
        # s = (d - num_det/2 + 0.5) * ds
        d_idx = torch.arange(self.num_det, dtype=torch.float32)
        s_positions = (d_idx - self.num_det / 2.0 + 0.5) * self.ds
        # Register as buffer so it moves to correct device
        self.register_buffer('s_positions', s_positions)
        
        # Matrix for 1st moment projection
        angles = torch.tensor(geom.angles, dtype=torch.float32)
        cos_theta = torch.cos(angles).view(-1, 1)
        sin_theta = torch.sin(angles).view(-1, 1)
        Phi = torch.cat([cos_theta, sin_theta], dim=1) # Shape: [num_views, 2]
        
        # P = Phi * (Phi^T * Phi)^-1 * Phi^T
        # We precompute the projection matrix (I - P)
        Phi_T_Phi_inv = torch.linalg.inv(torch.matmul(Phi.t(), Phi))
        P = torch.matmul(Phi, torch.matmul(Phi_T_Phi_inv, Phi.t()))
        I_minus_P = torch.eye(self.num_views) - P
        self.register_buffer('I_minus_P', I_minus_P)

    def forward(self, sinogram):
        """
        sinogram shape: [num_views, num_detectors]
        We flatten the batch dimension if it exists, or just handle 2D.
        """
        if sinogram.dim() == 3:
            # Handle batch dimension by taking mean over batch, or loop.
            # Assuming shape [1, V, D] for now
            sinogram = sinogram.squeeze(0)
            
        assert sinogram.shape == (self.num_views, self.num_det), f"Shape mismatch: {sinogram.shape}"
        
        # 0th Moment: Mass conservation
        # M(theta) = sum_s p(s, theta) * ds
        M_theta = torch.sum(sinogram, dim=1) * self.ds # Shape: [num_views]
        
        # Loss is the variance of M(theta) (it should be constant)
        loss_m0 = torch.var(M_theta)
        
        # 1st Moment: Center of mass moves sinusoidally
        # C(theta) = sum_s s * p(s, theta) * ds
        C_theta = torch.matmul(sinogram, self.s_positions) * self.ds # Shape: [num_views]
        
        # It should perfectly fit A*cos(theta) + B*sin(theta)
        # So (I - P) * C_theta should be 0
        residual = torch.matmul(self.I_minus_P, C_theta)
        loss_m1 = torch.sum(residual ** 2) / self.num_views
        
        return loss_m0, loss_m1

class AnnealedLoss(nn.Module):
    """
    Combines MSE Data loss with Physics-Informed Loss, 
    using a scheduling factor to anneal the physics weights.
    """
    def __init__(self, geom, lambda_m0=1.0, lambda_m1=1.0, anneal_epochs=50):
        super(AnnealedLoss, self).__init__()
        self.data_loss_fn = nn.MSELoss()
        self.physics_loss_fn = HelgasonLudwigLoss(geom)
        self.lambda_m0 = lambda_m0
        self.lambda_m1 = lambda_m1
        self.anneal_epochs = anneal_epochs

    def forward(self, pred_sino, true_sino, current_epoch):
        # Data loss
        loss_data = self.data_loss_fn(pred_sino, true_sino)
        
        # Physics loss
        loss_m0, loss_m1 = self.physics_loss_fn(pred_sino)
        
        # Annealing factor: 0 at epoch 0, 1 at epoch anneal_epochs
        alpha = min(1.0, current_epoch / self.anneal_epochs)
        
        total_loss = loss_data + alpha * (self.lambda_m0 * loss_m0 + self.lambda_m1 * loss_m1)
        
        return total_loss, loss_data, loss_m0, loss_m1
