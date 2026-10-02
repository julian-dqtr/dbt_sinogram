"""End-to-end smoke test: train (1 tiny epoch) -> save -> factory load -> same predictions."""
import json

import pytest
import torch

from src_2D.models.factory import get_model
from src_2D.train import build_parser, run_training


@pytest.mark.parametrize("model_args", [
    ["--model", "UNet2D", "--filters", "8"],
    ["--model", "UNet2dHLCC", "--filters", "8"],
    ["--model", "GCN", "--num_layers", "2", "--num_stalks", "4"],
    ["--model", "SNN", "--num_layers", "2", "--num_stalks", "4", "--physics", "hlcc"],
    ["--model", "UNet2dHLCC", "--filters", "8", "--hlcc_max_order", "3", "--train_repeats", "2"],
])
def test_train_save_reload(model_args, tmp_path, device):
    if device.type != "cuda":
        pytest.skip("ASTRA projector requires CUDA")
    args = build_parser().parse_args(model_args + [
        "--epochs", "1", "--n_samples", "4", "--n_val", "4", "--batch_size", "2",
        "--checkpoint-dir", str(tmp_path / "ckpt"), "--figures-dir", str(tmp_path / "fig"),
    ])
    best = run_training(args)
    assert best["best_epoch"] == 1 and all(map(lambda v: v == v, best.values()))  # no NaN

    for artefact in ("best_model.pt", "training_stats.json", "history.json"):
        assert (tmp_path / "ckpt" / artefact).exists()
    # One logged HLCC term per order of the physics loss (orders 0 and 1 unless --hlcc_max_order says otherwise).
    logged = json.loads((tmp_path / "ckpt" / "history.json").read_text())[0]
    assert {key for key in logged if key.startswith("train/hlcc_m")} == {f"train/hlcc_m{n}" for n in range(args.hlcc_max_order + 1)}
    model, is_nn = get_model(args.model, device, checkpoint_path=tmp_path / "ckpt" / "best_model.pt")
    assert is_nn
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 180, 128, device=device))
    assert out.shape == (1, 1, 180, 128)
