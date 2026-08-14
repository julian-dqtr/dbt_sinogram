import torch
import numpy as np
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.models.SinoSheavesNN.physics_loss import HelgasonLudwigLoss, AnnealedLoss

def test_physics_loss():
    num_views = 20
    num_det = 100
    angles = np.linspace(-np.pi/4, np.pi/4, num_views)
    geom = DBTGeometry(angles, src_radius=500, det_radius=100, det_col_count=num_det, det_pixel_size=2.0)
    
    # Create dummy sinograms
    pred_sino = torch.randn(num_views, num_det, requires_grad=True)
    true_sino = torch.randn(num_views, num_det)
    
    # 1. Test HelgasonLudwigLoss
    hl_loss = HelgasonLudwigLoss(geom)
    m0_loss, m1_loss = hl_loss(pred_sino)
    
    assert m0_loss.item() >= 0, "m0 variance should be non-negative"
    assert m1_loss.item() >= 0, "m1 residual should be non-negative"
    
    # 2. Test AnnealedLoss
    annealed_loss = AnnealedLoss(geom, lambda_m0=0.1, lambda_m1=0.1, anneal_epochs=10)
    
    # Test epoch 0 (alpha = 0)
    total_loss_epoch0, loss_data, loss_m0, loss_m1 = annealed_loss(pred_sino, true_sino, current_epoch=0)
    assert torch.allclose(total_loss_epoch0, loss_data), "At epoch 0, total loss should equal data loss"
    
    # Test epoch 5 (alpha = 0.5)
    total_loss_epoch5, _, _, _ = annealed_loss(pred_sino, true_sino, current_epoch=5)
    expected_loss5 = loss_data + 0.5 * (0.1 * loss_m0 + 0.1 * loss_m1)
    assert torch.allclose(total_loss_epoch5, expected_loss5), "Annealing interpolation failed at epoch 5"
    
    # Test epoch 20 (alpha = 1.0, capped)
    total_loss_epoch20, _, _, _ = annealed_loss(pred_sino, true_sino, current_epoch=20)
    expected_loss20 = loss_data + 1.0 * (0.1 * loss_m0 + 0.1 * loss_m1)
    assert torch.allclose(total_loss_epoch20, expected_loss20), "Annealing cap failed at epoch 20"
    
    # 3. Test gradients
    total_loss_epoch20.backward()
    assert pred_sino.grad is not None, "Sinogram did not receive gradients from the physics loss"
    assert not torch.isnan(pred_sino.grad).any(), "Gradients contain NaN"
    assert not torch.isinf(pred_sino.grad).any(), "Gradients contain Inf"
    
    print("All tests passed for Physics-Informed Loss!")

if __name__ == "__main__":
    test_physics_loss()
