"""GCN baseline vs SinoSheavesNN: same topology, same parameters, only the restriction maps differ.

The SNN translates every stalk along the detector by t_f (theta_i - theta_j) (transport "shift"); the rotation
transport "so2" of the first SNN is still tested at the level of its operators."""
import numpy as np
import pytest
import torch

from src_2D.models.factory import build_model, get_model
from src_2D.models.SinoSheavesNN.snn_layers import ViewTransport
from src_2D.models.SinoSheavesNN.snn_model import ViewGraphNet
from src_2D.models.SinoSheavesNN.graph_data import build_adjacency, build_transport_operators, normalize_adjacency
from src_2D.utils.checkpoint import save_checkpoint

SMALL = dict(num_stalks=4)


def test_knn_topology(geom):
    W = build_adjacency(geom.angles, k=12, sigma_deg=5.0)
    V = geom.num_views
    assert torch.equal(W, W.T), "the view graph must be symmetric"
    neighbours = (W > 0).sum(dim=1) - 1  # without the self-loop
    assert neighbours[V // 2] == 12 and neighbours[0] == 6 and neighbours.max() == 12
    hops = (torch.arange(V)[:, None] - torch.arange(V)[None, :]).abs()
    assert torch.all((W > 0) == (hops <= 6)), "k = 12 means exactly 6 views on each side"
    # sigma is a true standard deviation: weight exp(-1/2) at a distance of sigma (5 views of 1 degree)
    assert W[90, 95].item() == pytest.approx(np.exp(-0.5), rel=1e-6)
    a_hat = normalize_adjacency(W, "sym")
    assert torch.linalg.eigvalsh(a_hat).abs().max().item() <= 1.0 + 1e-9  # non-expansive diffusion


def test_restriction_maps_are_rotations_fixed_by_the_angles(geom):
    a_hat, none = build_transport_operators(geom.angles, "identity", k=12)
    a_cos, a_sin = build_transport_operators(geom.angles, "so2", k=12)
    assert none is None and a_sin is not None
    edges = a_hat > 0
    # R_ij = [[c, -s], [s, c]] with (c, s) = (a_cos, a_sin) / a_hat is in SO(2): c^2 + s^2 = 1
    c, s = a_cos[edges] / a_hat[edges], a_sin[edges] / a_hat[edges]
    torch.testing.assert_close(c**2 + s**2, torch.ones_like(c))
    # ... equals R(theta_i - theta_j) ...
    theta = torch.as_tensor(geom.angles, dtype=torch.float32)
    delta = (theta[:, None] - theta[None, :])[edges]
    torch.testing.assert_close(c, torch.cos(delta), atol=1e-6, rtol=0)
    torch.testing.assert_close(s, torch.sin(delta), atol=1e-6, rtol=0)
    # ... and R_ji = R_ij^T (cos symmetric, sin antisymmetric): a genuine O(2)-bundle connection.
    torch.testing.assert_close(a_cos, a_cos.T)
    torch.testing.assert_close(a_sin, -a_sin.T)


@pytest.mark.parametrize("num_layers", [6, 12, 18])
def test_both_models_instantiate_at_every_depth_with_identical_parameters(num_layers, config):
    gcn, gcn_cfg = build_model(f"GCN_L{num_layers}", config, **SMALL)
    snn, snn_cfg = build_model(f"SNN_L{num_layers}", config, **SMALL)
    assert isinstance(gcn, ViewGraphNet) and isinstance(snn, ViewGraphNet)
    assert gcn_cfg["num_layers"] == snn_cfg["num_layers"] == num_layers and gcn_cfg["k"] == 12
    assert gcn.transport_op.transport == "identity" and snn.transport_op.transport == "shift"
    assert [(k, v.shape) for k, v in gcn.state_dict().items()] == [(k, v.shape) for k, v in snn.state_dict().items()]
    assert torch.equal(gcn.transport_op.a_hat, snn.transport_op.a_hat), "same topology and weights"
    assert snn.angular_reach_deg == num_layers * 6.0


def test_snn_reduces_to_gcn_when_the_slopes_are_zero(config, clean_val_batch):
    """With every translation set to 0 the SNN code path (FFT and back) gives the GCN: the ablation isolates the transport."""
    incomplete = clean_val_batch[0][:2]
    gcn, _ = build_model("GCN", config, num_layers=2, **SMALL)
    snn, _ = build_model("SNN", config, num_layers=2, **SMALL)
    assert isinstance(gcn, ViewGraphNet) and isinstance(snn, ViewGraphNet)
    snn.load_state_dict(gcn.state_dict())
    gcn.eval()
    snn.eval()
    with torch.no_grad():
        assert not torch.allclose(snn(incomplete), gcn(incomplete), atol=1e-5), "the translation has no effect"
        snn.transport_op.phase[..., 0], snn.transport_op.phase[..., 1] = 1.0, 0.0
        torch.testing.assert_close(snn(incomplete), gcn(incomplete), atol=1e-5, rtol=1e-4)


def test_so2_aggregation_matches_an_explicit_edge_loop(geom):
    torch.manual_seed(0)
    a_hat, _ = build_transport_operators(geom.angles, "identity", k=4)
    theta = torch.as_tensor(geom.angles, dtype=torch.float32)
    x = torch.randn(1, 4, geom.num_views, 3)  # 2 stalks, 3 detector pixels
    dense = ViewTransport(geom.angles, "so2", num_stalks=2, k=4)(x)

    i = 50
    expected = torch.zeros(4, 3)
    for j in torch.nonzero(a_hat[i]).flatten().tolist():
        d = theta[i] - theta[j]
        R = torch.tensor([[torch.cos(d), -torch.sin(d)], [torch.sin(d), torch.cos(d)]])
        for f in range(2):  # stalk f = channels (2f, 2f + 1)
            expected[2 * f:2 * f + 2] += a_hat[i, j] * (R @ x[0, 2 * f:2 * f + 2, j])
    torch.testing.assert_close(dense[0, :, i], expected, atol=1e-5, rtol=1e-4)


def shift_transport(geom, config, num_stalks=4, k=4):
    return ViewTransport(geom.angles, "shift", num_stalks, k=k, num_pixels=config.det_col_count,
                         pixel_size_mm=config.det_pixel_size_mm, max_depth_mm=99.0)


def test_shift_aggregation_matches_an_explicit_edge_loop(geom, config):
    """The gauge factorisation S(t theta_i) A_hat S(-t theta_j) equals the per-edge translation S(t (theta_i - theta_j))."""
    torch.manual_seed(0)
    transport = shift_transport(geom, config)
    L, Lp = config.det_col_count, transport.padded_length
    x = torch.randn(1, 8, geom.num_views, L, dtype=torch.float64)  # 4 stalks
    dense = transport.double()(x)

    i = 90
    theta = torch.as_tensor(geom.angles, dtype=torch.float64)
    a_hat = transport.a_hat
    omega = 2 * np.pi * torch.arange(Lp // 2 + 1, dtype=torch.float64) / Lp
    expected = torch.zeros(8, L, dtype=torch.float64)
    for j in torch.nonzero(a_hat[i]).flatten().tolist():
        for c in range(8):
            line = x[0, c, j]
            padded = torch.cat([line, line[-1] + (line[0] - line[-1]) * transport.ramp])
            shift_px = transport.slopes_mm_per_rad[c // 2] * (theta[i] - theta[j]) / config.det_pixel_size_mm
            moved = torch.fft.irfft(torch.fft.rfft(padded) * torch.exp(-1j * omega * shift_px), n=Lp)[:L]
            expected[c] += a_hat[i, j] * moved
    torch.testing.assert_close(dense[0, :, i], expected, atol=1e-6, rtol=1e-5)  # float32 phases


def test_shift_transport_aligns_the_traces_of_its_depth(geom, config):
    """Units and sign: a line whose content moves along the detector at t_f mm per radian (the trace of a point
    at depth t_f) is aligned by stalk f: its aggregated message is the line itself, unblurred. The GCN, and the
    stalk of opposite slope, average misaligned copies."""
    transport = shift_transport(geom, config, num_stalks=8, k=12).double()
    gcn = ViewTransport(geom.angles, "identity", num_stalks=8, k=12).double()
    f = 6
    slope = transport.slopes_mm_per_rad[f]                        # about 57 mm / rad
    s = torch.arange(config.det_col_count, dtype=torch.float64)
    theta = torch.as_tensor(geom.angles, dtype=torch.float64)
    centre = 64.0 + slope * theta / config.det_pixel_size_mm       # pixels, moves ~0.6 px per view
    trace = torch.exp(-0.5 * ((s[None, :] - centre[:, None]) / 2.0) ** 2)  # [V, L], blob of 2 px
    x = torch.zeros(1, 16, geom.num_views, config.det_col_count, dtype=torch.float64)
    x[0, 2 * f] = trace                                            # stalk f
    x[0, 2 * (7 - f)] = trace                                      # stalk of opposite slope
    i = 90                                                         # theta = 0: the blob is well inside the detector
    weight = transport.a_hat[i].sum()

    def error(messages, channel):
        return ((messages[0, channel, i] / weight - trace[i]).norm() / trace[i].norm()).item()

    shifted, averaged = transport(x), gcn(x)
    assert transport.slopes_mm_per_rad[7 - f] == pytest.approx(-slope)
    assert error(shifted, 2 * f) < 1e-6, "stalk f does not align the trace of slope t_f"
    assert error(shifted, 2 * (7 - f)) > 0.3 and error(averaged, 2 * f) > 0.1, "misaligned copies should blur"


@pytest.mark.parametrize("name", ["GCN", "SNN"])
def test_angular_receptive_field_is_exactly_layers_times_half_k(name, config, geom):
    """The locality limit studied in the thesis: measurements cannot influence a view further than
    num_layers * k/2 views away from the acquired window, whatever the weights."""
    torch.manual_seed(0)
    num_layers, k = 3, 4  # reach = 6 views
    model, _ = build_model(name, config, num_layers=num_layers, k=k, **SMALL)
    model.eval()
    acquired = np.flatnonzero(geom.acquired_view_mask)
    x1 = torch.zeros(1, 1, geom.num_views, config.det_col_count)
    x2 = x1.clone()
    x1[:, :, acquired] = torch.rand(1, 1, len(acquired), config.det_col_count)
    x2[:, :, acquired] = torch.rand(1, 1, len(acquired), config.det_col_count)
    with torch.no_grad():
        diff = (model(x1) - model(x2)).abs().amax(dim=(0, 1, 3))

    reach = num_layers * k // 2
    beyond = np.r_[0:acquired[0] - reach, acquired[-1] + reach + 1:geom.num_views]
    within = np.r_[acquired[0] - reach:acquired[0], acquired[-1] + 1:acquired[-1] + reach + 1]
    assert torch.all(diff[beyond] == 0), "information travelled further than the theoretical reach"
    assert torch.all(diff[within] > 0), "information did not reach a view inside the theoretical reach"


@pytest.mark.parametrize("name", ["GCN", "SNN"])
def test_training_step_and_checkpoint_round_trip(name, config, clean_val_batch, device, tmp_path):
    torch.manual_seed(0)
    incomplete, full = clean_val_batch[0][:2].to(device), clean_val_batch[1][:2].to(device)
    model, model_config = build_model(name, config, num_layers=2, **SMALL)
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(8):
        loss = torch.nn.functional.mse_loss(model(incomplete), full)
        optimizer.zero_grad()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]

    path = tmp_path / "best_model.pt"
    save_checkpoint(path, model, name, model_config, config)
    reloaded, _ = get_model(name, device, checkpoint_path=path)
    assert isinstance(reloaded, ViewGraphNet)
    assert reloaded.num_layers == 2  # architecture read from the checkpoint, not from a default
    with pytest.raises(ValueError, match="implies num_layers=12"):
        get_model(f"{name}_L12", device, checkpoint_path=path)
    with pytest.raises(ValueError, match="implies num_layers=6"):
        build_model(f"{name}_L6", config, num_layers=2)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(incomplete), model(incomplete))


def test_batch_elements_are_independent(config, clean_val_batch):
    incomplete = clean_val_batch[0][:3]
    model, _ = build_model("SNN", config, num_layers=2, **SMALL)
    model.eval()
    with torch.no_grad():
        batched = model(incomplete)
        single = torch.cat([model(incomplete[i:i + 1]) for i in range(3)])
    torch.testing.assert_close(batched, single, atol=1e-6, rtol=1e-5)
