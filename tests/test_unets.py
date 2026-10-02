"""Smoke tests of UNet2D / UNet2dHLCC: shapes, gradients, a few optimisation steps, checkpoint round trip."""
import pytest
import torch

from src_2D.models.factory import build_model, get_model
from src_2D.models.SinoSheavesNN.physics_loss import AnnealedLoss
from src_2D.utils.checkpoint import save_checkpoint


@pytest.mark.parametrize("name", ["UNet2D", "UNet2dHLCC"])
def test_forward_shape_and_data_consistency(name, config, geom, clean_val_batch, device):
    incomplete = clean_val_batch[0][:2].to(device)
    model, _ = build_model(name, config, filters=8)
    model = model.to(device).eval()
    with torch.no_grad():
        out = model(incomplete)
    assert out.shape == incomplete.shape and torch.isfinite(out).all()

    # The core of the acquired window is copied from the measurements, bit for bit.
    core = (model.soft_mask.flatten() == 1.0)
    assert core.sum() == 43
    assert torch.equal(out[:, :, core], incomplete[:, :, core])
    # Without DC the network output differs there: the option really is wired.
    with torch.no_grad():
        assert not torch.equal(model(incomplete, apply_dc=False)[:, :, core], incomplete[:, :, core])


def test_both_unets_are_the_same_network(config):
    a, _ = build_model("UNet2D", config, filters=8)
    b, _ = build_model("UNet2dHLCC", config, filters=8)
    assert [(k, v.shape) for k, v in a.state_dict().items()] == [(k, v.shape) for k, v in b.state_dict().items()]


@pytest.mark.parametrize("name,physics", [("UNet2D", False), ("UNet2dHLCC", True)])
def test_a_few_training_steps_reduce_the_loss(name, physics, config, geom, clean_val_batch, device):
    torch.manual_seed(0)
    incomplete, full = clean_val_batch[0][:4].to(device), clean_val_batch[1][:4].to(device)
    model, _ = build_model(name, config, filters=8)
    model = model.to(device).train()
    loss_fn = AnnealedLoss(geom, lambda_m0=0.1, lambda_m1=0.1, anneal_epochs=0).to(device)
    loss_fn.calibrate(incomplete[:, 0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(8):
        pred = model(incomplete)
        if physics:
            loss, mse, m0, m1 = loss_fn(pred[:, 0], full[:, 0], current_epoch=1)
            assert all(torch.isfinite(t) for t in (mse, m0, m1))
        else:
            loss = torch.nn.functional.mse_loss(pred, full)
        optimizer.zero_grad()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


@pytest.mark.parametrize("name", ["UNet2D", "UNet2dHLCC"])
def test_checkpoint_round_trip_through_the_factory(name, config, clean_val_batch, device, tmp_path):
    incomplete = clean_val_batch[0][:1].to(device)
    model, model_config = build_model(name, config, filters=8)
    model = model.to(device).eval()
    path = tmp_path / "best_model.pt"
    save_checkpoint(path, model, name, model_config, config, meta={"epoch": 1})

    reloaded, is_nn = get_model(name, device, checkpoint_path=path)  # no filters given: read from the file
    assert is_nn and isinstance(reloaded, torch.nn.Module) and not reloaded.training
    with torch.no_grad():
        torch.testing.assert_close(reloaded(incomplete), model(incomplete))


def test_factory_raises_instead_of_scoring_a_random_network(config, device, tmp_path):
    # Legacy bare state_dict
    model, model_config = build_model("UNet2D", config, filters=8)
    legacy = tmp_path / "legacy.pt"
    torch.save(model.state_dict(), legacy)
    with pytest.raises(ValueError, match="legacy"):
        get_model("UNet2D", device, checkpoint_path=legacy)

    # Wrong model type, wrong geometry, missing file
    path = tmp_path / "best_model.pt"
    save_checkpoint(path, model, "UNet2D", model_config, config)
    with pytest.raises(ValueError, match="expected UNet2dHLCC"):
        get_model("UNet2dHLCC", device, checkpoint_path=path)
    from dataclasses import replace
    with pytest.raises(ValueError, match="different geometry"):
        get_model("UNet2D", device, checkpoint_path=path, geometry_config=replace(config, angle_max_deg=30.0))
    with pytest.raises(FileNotFoundError):
        get_model("UNet2D", device, checkpoint_path=tmp_path / "missing.pt")
