"""Data-consistency mask: hard by default (the measurements come out bit for bit). The soft variant
(taper INSIDE the acquired window, never blending predictions with zeros) is kept for the first runs."""
import numpy as np
import pytest
import torch

from src_2D.utils.evaluation import apply_data_consistency, get_data_consistency_mask


def test_default_mask_is_hard(geom):
    mask = get_data_consistency_mask(geom, torch.device("cpu")).flatten().numpy()
    np.testing.assert_array_equal(mask, geom.acquired_view_mask.astype(np.float32))
    explicit = get_data_consistency_mask(geom, torch.device("cpu"), blend_width_deg=0.0).flatten().numpy()
    np.testing.assert_array_equal(mask, explicit)


def test_soft_mask_structure(geom):
    mask = get_data_consistency_mask(geom, torch.device("cpu"), blend_width_deg=5.0).flatten().numpy()
    acquired = geom.acquired_view_mask
    assert np.all(mask[~acquired] == 0.0), "a missing view has a non-zero measured weight"
    assert np.all(mask[acquired] > 0.0), "an acquired view is ignored"
    assert np.isclose(mask[acquired], 1.0).sum() == 51 - 2 * 4  # 4 tapered views per side (1 degree steps)
    # Symmetric about 0 degrees: view i (angle -90 + i) mirrors view 180 - i; -90 has no mirror on [-90, 90).
    np.testing.assert_allclose(mask[1:], mask[1:][::-1], atol=1e-6)
    inside = mask[acquired]
    assert np.all(np.diff(inside[:26]) >= 0) and np.all(np.diff(inside[25:]) <= 0)  # monotone taper


@pytest.mark.parametrize("blend_width_deg", [0.0, 5.0])
def test_dc_is_the_identity_on_the_ground_truth(blend_width_deg, geom, clean_val_batch):
    """An oracle prediction must go through the DC step unchanged (the old mask capped SSIM at 0.975)."""
    incomplete, full, _ = clean_val_batch
    mask = get_data_consistency_mask(geom, torch.device("cpu"), blend_width_deg)
    torch.testing.assert_close(apply_data_consistency(incomplete, full, mask), full, atol=1e-7, rtol=0)


@pytest.mark.parametrize("blend_width_deg", [0.0, 5.0])
def test_dc_never_attenuates_predictions_on_missing_views(blend_width_deg, geom, clean_val_batch):
    incomplete, _, _ = clean_val_batch
    mask = get_data_consistency_mask(geom, torch.device("cpu"), blend_width_deg)
    prediction = torch.full_like(incomplete, 3.0)
    out = apply_data_consistency(incomplete, prediction, mask)
    missing = torch.from_numpy(~geom.acquired_view_mask)
    assert torch.all(out[:, :, missing] == 3.0)


def test_hard_dc_returns_every_measured_view_bit_for_bit(geom, clean_val_batch):
    incomplete, _, _ = clean_val_batch
    mask = get_data_consistency_mask(geom, torch.device("cpu"))
    prediction = torch.randn(incomplete.shape, generator=torch.Generator().manual_seed(0))
    out = apply_data_consistency(incomplete, prediction, mask)
    acquired = torch.from_numpy(geom.acquired_view_mask)
    assert torch.equal(out[:, :, acquired], incomplete[:, :, acquired])
    assert torch.equal(out[:, :, ~acquired], prediction[:, :, ~acquired])
