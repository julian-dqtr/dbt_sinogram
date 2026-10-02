"""HLCC moment regression (Huang et al. 2017) used as a post-processing, and the "_P<k>" model names."""
import pytest
import torch

from src_2D.models.factory import build_model, get_model, parse_full_model_name, parse_model_name, run_name_of
from src_2D.utils.checkpoint import save_checkpoint
from src_2D.utils.hlcc import HLCCMomentProjection, chebyshev_u, detector_coordinate


def _missing(geom) -> torch.Tensor:
    return torch.from_numpy(~geom.acquired_view_mask)


def test_projected_views_get_exactly_the_regressed_moments(geom, clean_val_batch):
    incomplete, full, _ = clean_val_batch
    projection = HLCCMomentProjection(geom, max_order=4, support_threshold=None)
    prediction = full + 0.05 * torch.randn(full.shape, generator=torch.Generator().manual_seed(0))
    out = projection(prediction, incomplete)
    missing = _missing(geom)

    assert out.shape == prediction.shape and out.dtype == prediction.dtype
    assert torch.equal(out[:, :, ~missing], prediction[:, :, ~missing]), "an acquired view was modified"
    target = projection.regressed_moments(incomplete)[:, :, missing]
    torch.testing.assert_close(projection.moments(out.double())[:, :, missing], target, atol=1e-6, rtol=0)
    # Projecting twice changes nothing: the moments are already the regressed ones.
    torch.testing.assert_close(projection(out, incomplete), out, atol=1e-6, rtol=0)


def test_the_correction_stays_inside_the_shadow_of_the_object(geom, clean_val_batch):
    """Default behaviour: where the completed view is (almost) zero, nothing is added."""
    incomplete, full, _ = clean_val_batch
    missing = _missing(geom)
    prediction = (0.9 * full).clone()  # wrong mass, right support
    background = prediction <= 0.03 * prediction.amax(dim=-1, keepdim=True)

    leaking = HLCCMomentProjection(geom, max_order=4, support_threshold=None)(prediction, incomplete)
    inside = HLCCMomentProjection(geom, max_order=4)(prediction, incomplete)
    assert not torch.equal(leaking[background], prediction[background])
    assert torch.equal(inside[background], prediction[background])
    assert torch.equal(inside[:, :, ~missing], prediction[:, :, ~missing])

    def wedge_mse(sinogram):
        return ((sinogram - full)[:, :, missing] ** 2).mean().item()

    assert wedge_mse(inside) < 0.2 * wedge_mse(prediction)  # the 10 % mass error is mostly repaired


def test_an_error_made_of_low_order_atoms_is_removed(geom, clean_val_batch):
    """Add c(theta) W(s) U_2(s) to the ground truth of the missing views: the regression of order >= 2
    knows the true second moment from the acquired views and takes the error away."""
    incomplete, full, _ = clean_val_batch
    x = detector_coordinate(geom)
    atom = (torch.sqrt(1 - x**2) * chebyshev_u(x, 2)[:, 2]).float()
    missing = _missing(geom)
    corrupted = full.clone()
    corrupted[:, :, missing] += 0.2 * torch.cos(torch.as_tensor(geom.angles[missing.numpy()], dtype=torch.float32)).view(1, 1, -1, 1) * atom

    def wedge_mse(sinogram):
        return ((sinogram - full)[:, :, missing] ** 2).mean().item()

    before = wedge_mse(corrupted)
    assert wedge_mse(HLCCMomentProjection(geom, 1, support_threshold=None)(corrupted, incomplete)) > 0.9 * before  # order too low
    assert wedge_mse(HLCCMomentProjection(geom, 2, support_threshold=None)(corrupted, incomplete)) < 1e-2 * before


def test_the_clean_ground_truth_is_almost_a_fixed_point(geom, clean_val_batch):
    """With noise-free measurements, the regressed moments of orders <= 3 are the true ones up to the
    discretisation error, so the projection must barely move the ground truth."""
    incomplete, full, _ = clean_val_batch
    missing = _missing(geom)
    projected = HLCCMomentProjection(geom, max_order=3, support_threshold=None)(full, incomplete)
    change = ((projected - full)[:, :, missing] ** 2).mean().item()
    assert change < 1e-3 * (full[:, :, missing] ** 2).mean().item()


def test_model_names_with_run_tags_and_projection():
    assert parse_full_model_name("UNet2dHLCC_N200_P3") == ("UNet2dHLCC", {}, ("N200",), 3)
    assert parse_full_model_name("SNN_L12") == ("SNN", {"num_layers": 12}, (), None)
    assert parse_model_name("GCN_L18_N200") == ("GCN", {"num_layers": 18})
    assert run_name_of("UNet2D") == "UNet2D" and run_name_of("UNet2D_P4") == "UNet2D"
    assert run_name_of("UNet2dHLCC_N200_P4") == "UNet2dHLCC_N200"
    assert run_name_of("SNN") == "SNN_L6" and run_name_of("GCN_L12_N200") == "GCN_L12_N200"
    assert run_name_of("LinearInterp_P4") == "LinearInterp"
    for bad in ("UNet2D_L6", "UNet2D-P4", "UNet2D_", "UNet2D_4"):
        with pytest.raises(ValueError):
            parse_full_model_name(bad)


def test_projected_models_from_the_factory(config, geom, clean_val_batch, device, tmp_path):
    incomplete = clean_val_batch[0][:2].to(device)
    acquired = ~_missing(geom)

    baseline, is_nn = get_model("LinearInterp_P4", device, geometry_config=config)
    assert not is_nn and sum(p.numel() for p in baseline.parameters()) == 0
    plain, _ = get_model("LinearInterp", device, geometry_config=config)
    out = baseline(incomplete)
    assert out.shape == incomplete.shape and not torch.equal(out, plain(incomplete))
    assert torch.equal(out[:, :, acquired], incomplete[:, :, acquired])

    model, model_config = build_model("UNet2D", config, filters=8)
    path = tmp_path / "best_model.pt"
    save_checkpoint(path, model, "UNet2D", model_config, config)
    projected, is_nn = get_model("UNet2D_N200_P2", device, checkpoint_path=path)
    reference, _ = get_model("UNet2D_N200", device, checkpoint_path=path)
    assert is_nn and not projected.training
    assert sum(p.numel() for p in projected.parameters()) == sum(p.numel() for p in reference.parameters())
    with torch.no_grad():
        out, raw = projected(incomplete), reference(incomplete)
    assert torch.equal(out[:, :, acquired], incomplete[:, :, acquired])  # hard DC, untouched by the projection
    assert not torch.equal(out, raw)
