"""HLCC loss: exact values on analytic data, ~0 on the ground truth of the simulated geometry."""
import numpy as np
import pytest
import torch

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.models.SinoSheavesNN.physics_loss import AnnealedLoss, HelgasonLudwigLoss
from src_2D.utils.hlcc import detector_coordinate, harmonic_basis, residual_projectors
from tests.conftest import analytic_disk_sinogram


def test_moments_match_closed_form(geom):
    radius, center, density = 20.0, (15.0, -10.0), 0.5
    sino = analytic_disk_sinogram(geom, radius, center, density)
    loss = HelgasonLudwigLoss(geom)
    moments = loss.moments(sino)[0].numpy()  # [V, 2]: Chebyshev moments a_0 = M0 / S and a_1 = 2 M1 / S^2
    S = loss.half_width

    mass = density * np.pi * radius**2
    expected_m1 = mass * (center[0] * np.cos(geom.angles) - center[1] * np.sin(geom.angles))
    np.testing.assert_allclose(moments[:, 0] * S, mass, rtol=2e-3)
    np.testing.assert_allclose(moments[:, 1] * S**2 / 2, expected_m1, atol=2e-3 * mass * np.hypot(*center))


def test_harmonic_spaces_have_the_helgason_ludwig_structure(geom):
    """H_n has dimension n + 1, contains the frequencies m <= n of the parity of n and nothing else."""
    theta = torch.as_tensor(geom.angles)
    projectors = residual_projectors(geom.angles, 4)
    for n in range(5):
        basis = harmonic_basis(geom.angles, n)
        assert basis.shape == (geom.num_views, n + 1)
        assert torch.linalg.matrix_rank(basis) == n + 1
        assert (projectors[n] @ basis).abs().max() < 1e-10           # allowed harmonics: no residual
        for m in (n + 1, n + 2):                                      # wrong parity, too high a frequency
            forbidden = torch.cos(m * theta)
            assert (projectors[n] @ forbidden).norm() > 0.1 * forbidden.norm()


def test_loss_is_zero_on_consistent_data_and_exact_on_a_known_violation(geom):
    loss = HelgasonLudwigLoss(geom)
    sino = analytic_disk_sinogram(geom)
    l0, l1 = loss.raw_losses(sino)
    a0 = loss.moments(sino)[0, :, 0].mean().item()
    assert l0.item() < 1e-5 * a0**2  # relative std of the mass below 0.3 %

    # Known order-0 violation: scale ONE view by (1 + eps) -> a_0 has a single outlier of
    # size eps * a0, whose population variance over V views is (eps * a0)^2 (1 - 1/V) / V.
    eps, V = 0.1, geom.num_views
    corrupted = sino.clone()
    corrupted[40] *= 1.0 + eps
    l0_corrupted = loss.raw_losses(corrupted)[0]
    assert l0_corrupted.item() == pytest.approx((eps * a0) ** 2 * (1 - 1 / V) / V, rel=0.02)

    # Known order-1 violation: add a constant c to a_1 (not in span{cos, sin} on a half circle).
    # Expected residual energy: ||(I - P) c 1||^2 / V, computed independently with numpy.
    Phi = np.stack([np.cos(geom.angles), np.sin(geom.angles)], axis=1)
    ones = np.ones(V)
    residual = ones - Phi @ np.linalg.lstsq(Phi, ones, rcond=None)[0]
    # A sinogram whose only first-moment content is a constant: two pixels (-x, +x) per view.
    x = detector_coordinate(geom).numpy()
    two_point = torch.zeros(V, geom.det_col_count)
    two_point[:, 10], two_point[:, -11] = 1.0, 2.0  # symmetric pixels, a_1 = (2 - 1) * U_1(x[-11]) * dx
    c = 2.0 * x[-11] * (2.0 / geom.det_col_count)
    l0_const, l1_const = loss.raw_losses(two_point)
    assert l1_const.item() == pytest.approx(c**2 * (residual**2).sum() / V, rel=1e-4)
    assert l0_const.item() < 1e-12  # the mass of this sinogram is the same in every view


def test_orders_0_and_1_are_the_historical_loss(geom, clean_val_batch):
    """max_order = 1 must reproduce the loss of the first UNet2dHLCC runs once calibrated: the variance
    of M0 = sum_s p ds over the views, and the part of M1 = sum_s s p ds outside span{cos, sin}."""
    incomplete, full, _ = clean_val_batch
    noise = 0.05 * torch.randn(full[:, 0].shape, generator=torch.Generator().manual_seed(0))
    prediction = full[:, 0] + noise

    def historical(sinogram: torch.Tensor):
        ds = geom.det_pixel_size
        s = (np.arange(geom.det_col_count) - geom.det_col_count / 2.0 + 0.5) * ds
        m0 = sinogram.double().numpy().sum(axis=2) * ds
        m1 = (sinogram.double().numpy() @ s) * ds
        Phi = np.stack([np.cos(geom.angles), np.sin(geom.angles)], axis=1)
        residual = m1 - m1 @ (Phi @ np.linalg.pinv(Phi)).T
        return m0.var(axis=1, ddof=1).mean(), (residual**2).sum(axis=1).mean() / geom.num_views

    loss = HelgasonLudwigLoss(geom, max_order=1)
    loss.calibrate(incomplete[:, 0])
    normalised = loss(prediction).numpy()
    expected = np.array(historical(prediction)) / np.array(historical(incomplete[:, 0]))
    np.testing.assert_allclose(normalised, expected, rtol=1e-3)


def test_ground_truth_of_the_simulated_geometry_satisfies_hlcc(geom, clean_val_batch):
    """THE test of the parallel-beam pivot: loss(GT) must be < 1 % of loss(zero-filled input), up to order 3."""
    incomplete, full, _ = clean_val_batch
    loss = HelgasonLudwigLoss(geom, max_order=3)
    ratio = (loss.raw_losses(full[:, 0]) / loss.raw_losses(incomplete[:, 0])).tolist()
    assert ratio[0] < 1e-3, f"order 0: {ratio[0]:.2e}"
    assert all(r < 1e-2 for r in ratio[1:]), f"orders 1..3: {ratio[1:]}"
    assert loss.relative_residuals(full[:, 0]).max().item() < 1e-3

    m0 = loss.moments(full[:, 0])[..., 0]
    rel_std_m0 = (m0.std(dim=1) / m0.mean(dim=1)).max().item()
    assert rel_std_m0 < 5e-3, f"M0 varies by {rel_std_m0:.2%} across views"


def test_legacy_fan_beam_ground_truth_violates_hlcc():
    """Documents WHY the stationary-detector fan-beam was abandoned (figure-ready numbers)."""
    fan_config = DBTGeometryConfig(beam="fanflat_vec", det_pixel_size_mm=2.5)
    fan_geom = DBTGeometry.from_config(fan_config)
    dataset = SinogramCompletionDataset(4, split="val", noise_level=0.0, geometry_config=fan_config)
    full = torch.stack([dataset[i][1][0] for i in range(4)])
    m0 = HelgasonLudwigLoss(fan_geom).moments(full)[..., 0]
    assert (m0.std(dim=1) / m0.mean(dim=1)).mean().item() > 0.15


@pytest.mark.parametrize("max_order", [1, 3])
def test_annealed_loss_schedule_and_calibration(max_order, geom, clean_val_batch):
    incomplete, full, _ = clean_val_batch
    loss_fn = AnnealedLoss(geom, lambda_m0=0.5, lambda_m1=0.5, anneal_epochs=10, lambda_high=0.5, max_order=max_order)
    assert loss_fn.alpha(0) == 0.0 and loss_fn.alpha(5) == pytest.approx(0.5) and loss_fn.alpha(10) == 1.0
    assert loss_fn.lambdas.tolist() == [0.5] * (max_order + 1)

    loss_fn.calibrate(incomplete[:, 0])
    # After calibration the zero-filled input scores exactly 1 on every order, the GT ~0.
    _, _, zero_filled = loss_fn(incomplete[:, 0], full[:, 0], 10)
    total_gt, mse_gt, ground_truth = loss_fn(full[:, 0], full[:, 0], 10)
    assert zero_filled.shape == ground_truth.shape == (max_order + 1,)
    assert zero_filled.tolist() == pytest.approx([1.0] * (max_order + 1), rel=1e-4)
    assert mse_gt.item() == 0.0 and ground_truth[0].item() < 1e-3 and ground_truth.max().item() < 1e-2
    # The total objective must rank the ground truth far below the naive input.
    total_zf = loss_fn(incomplete[:, 0], full[:, 0], 10)[0]
    assert total_gt.item() < 0.01 * total_zf.item()


def test_lambda_high_weights_the_orders_above_one(geom):
    loss_fn = AnnealedLoss(geom, lambda_m0=1e-3, lambda_m1=2e-3, lambda_high=3e-3, max_order=3)
    assert loss_fn.lambdas.tolist() == pytest.approx([1e-3, 2e-3, 3e-3, 3e-3])
    assert AnnealedLoss(geom, lambda_m0=1e-3, lambda_m1=2e-3, max_order=0).lambdas.tolist() == pytest.approx([1e-3])
