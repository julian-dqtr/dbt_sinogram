"""HLCC loss: exact values on analytic data, ~0 on the ground truth of the simulated geometry."""
import numpy as np
import pytest
import torch

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.models.SinoSheavesNN.physics_loss import AnnealedLoss, HelgasonLudwigLoss
from tests.conftest import analytic_disk_sinogram


def test_moments_match_closed_form(geom):
    radius, center, density = 20.0, (15.0, -10.0), 0.5
    sino = analytic_disk_sinogram(geom, radius, center, density)
    m0, m1 = HelgasonLudwigLoss(geom).moments(sino)

    mass = density * np.pi * radius**2
    expected_m1 = mass * (center[0] * np.cos(geom.angles) - center[1] * np.sin(geom.angles))
    np.testing.assert_allclose(m0[0].numpy(), mass, rtol=2e-3)
    np.testing.assert_allclose(m1[0].numpy(), expected_m1, atol=2e-3 * mass * np.hypot(*center))


def test_loss_is_zero_on_consistent_data_and_exact_on_a_known_violation(geom):
    loss = HelgasonLudwigLoss(geom)
    sino = analytic_disk_sinogram(geom)
    l0, l1 = loss._raw_losses(sino)
    m0 = loss.moments(sino)[0].mean().item()
    assert l0.item() < 1e-5 * m0**2  # relative std of M0 below 0.3 %

    # Known order-0 violation: scale ONE view by (1 + eps) -> M0 has a single outlier of
    # size eps * mass, whose unbiased variance over V views is (eps * mass)^2 / V.
    eps, V = 0.1, geom.num_views
    corrupted = sino.clone()
    corrupted[40] *= 1.0 + eps
    l0_corrupted, _ = loss._raw_losses(corrupted)
    assert l0_corrupted.item() == pytest.approx((eps * m0) ** 2 / V, rel=0.02)

    # Known order-1 violation: add a constant c to M1 (not in span{cos, sin} on a half circle).
    # Expected residual energy: ||(I - P) c 1||^2 / V, computed independently with numpy.
    Phi = np.stack([np.cos(geom.angles), np.sin(geom.angles)], axis=1)
    ones = np.ones(V)
    residual = ones - Phi @ np.linalg.lstsq(Phi, ones, rcond=None)[0]
    # A sinogram whose only first-moment content is a constant: two pixels (-a, +a) per view.
    s = loss.s_positions.numpy()
    two_point = torch.zeros(V, geom.det_col_count)
    two_point[:, 10], two_point[:, -11] = 1.0, 2.0  # symmetric pixels, M1 = (2 - 1) * s[-11] * ds
    c = s[-11] * geom.det_pixel_size
    _, l1_const = loss._raw_losses(two_point)
    assert l1_const.item() == pytest.approx(c**2 * (residual**2).sum() / V, rel=1e-4)


def test_ground_truth_of_the_simulated_geometry_satisfies_hlcc(geom, clean_val_batch):
    """THE test of the parallel-beam pivot: loss(GT) must be < 1 % of loss(zero-filled input)."""
    incomplete, full, _ = clean_val_batch
    loss = HelgasonLudwigLoss(geom)
    l0_gt, l1_gt = loss._raw_losses(full[:, 0])
    l0_zf, l1_zf = loss._raw_losses(incomplete[:, 0])
    assert l0_gt / l0_zf < 1e-3, f"order 0: {l0_gt / l0_zf:.2e}"
    assert l1_gt / l1_zf < 1e-2, f"order 1: {l1_gt / l1_zf:.2e}"

    m0, m1 = loss.moments(full[:, 0])
    rel_std_m0 = (m0.std(dim=1) / m0.mean(dim=1)).max().item()
    assert rel_std_m0 < 5e-3, f"M0 varies by {rel_std_m0:.2%} across views"


def test_legacy_fan_beam_ground_truth_violates_hlcc():
    """Documents WHY the stationary-detector fan-beam was abandoned (figure-ready numbers)."""
    fan_config = DBTGeometryConfig(beam="fanflat_vec", det_pixel_size_mm=2.5)
    fan_geom = DBTGeometry.from_config(fan_config)
    dataset = SinogramCompletionDataset(4, split="val", noise_level=0.0, geometry_config=fan_config)
    full = torch.stack([dataset[i][1][0] for i in range(4)])
    m0, _ = HelgasonLudwigLoss(fan_geom).moments(full)
    assert (m0.std(dim=1) / m0.mean(dim=1)).mean().item() > 0.15


def test_annealed_loss_schedule_and_calibration(geom, clean_val_batch):
    incomplete, full, _ = clean_val_batch
    loss_fn = AnnealedLoss(geom, lambda_m0=0.5, lambda_m1=0.5, anneal_epochs=10)
    assert loss_fn.alpha(0) == 0.0 and loss_fn.alpha(5) == pytest.approx(0.5) and loss_fn.alpha(10) == 1.0

    loss_fn.calibrate(incomplete[:, 0])
    # After calibration the zero-filled input scores exactly 1 on both terms, the GT ~0.
    _, _, zf_m0, zf_m1 = loss_fn(incomplete[:, 0], full[:, 0], 10)
    total_gt, mse_gt, gt_m0, gt_m1 = loss_fn(full[:, 0], full[:, 0], 10)
    assert zf_m0.item() == pytest.approx(1.0, rel=1e-4) and zf_m1.item() == pytest.approx(1.0, rel=1e-4)
    assert mse_gt.item() == 0.0 and gt_m0.item() < 1e-3 and gt_m1.item() < 1e-2
    # The total objective must rank the ground truth far below the naive input.
    total_zf = loss_fn(incomplete[:, 0], full[:, 0], 10)[0]
    assert total_gt.item() < 0.01 * total_zf.item()
