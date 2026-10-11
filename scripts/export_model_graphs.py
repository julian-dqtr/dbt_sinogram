#!/usr/bin/env python3
"""Computational graphs of the learned models, to check an architecture (the thesis figures are made by
scripts/plot_architectures.py).

    PATH=~/miniconda3/envs/tex/bin:$PATH python scripts/export_model_graphs.py     # torchview + ONNX
    python scripts/export_model_graphs.py --skip-torchview --onnx GCN_L18 SNN_L18

- torchview (graphviz `dot` must be on the PATH): results/architectures/torchview/<model>.{pdf,png} for
  UNet2D, UNet2dHLCC, UNet2dHLCC_P4, GCN and SNN at their default configuration (untrained: the graph
  does not depend on the weights). For GCN and SNN the operations inside the modules are shown too,
  otherwise the two graphs are identical (only the inside of ViewTransport differs).
- Netron files of the trained graph models (--onnx, checkpoints of outputs/2d/checkpoints/), in
  results/architectures/netron/, to open with Netron (https://netron.app or `pip install netron`):
  <run>.onnx when the ONNX exporter supports the model (GCN), else TorchScript <run>.pt (SNN: the complex
  FFT of the shift transport has no ONNX translation in torch 2.7). Each file is checked against PyTorch.
  ONNX and not HDF5: .h5 is the Keras / TensorFlow format, PyTorch models are exported to ONNX.
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.models.factory import get_model

OUT_DIR = PROJECT_ROOT / "results/architectures"
TORCHVIEW_MODELS = ("UNet2D", "UNet2dHLCC", "UNet2dHLCC_P4", "GCN", "SNN")
# Module nesting shown by torchview: down to the conv / norm / activation layers (one more level for the P4
# wrapper HLCCProjectedModel, two more for the transport of the graph models).
TORCHVIEW_DEPTH = {"UNet2D": 3, "UNet2dHLCC": 3, "UNet2dHLCC_P4": 4, "GCN": 4, "SNN": 4}
GRAPH_MODELS = ("GCN", "SNN")


def export_torchview(name: str, input_shape) -> None:
    from torchview import draw_graph

    model, _ = get_model(name, torch.device("cpu"), load_weights=False)
    graph = draw_graph(model, input_size=input_shape, depth=TORCHVIEW_DEPTH[name], expand_nested=True,
                       hide_inner_tensors=True, hide_module_functions=name not in GRAPH_MODELS, roll=False,
                       graph_name=name, device="cpu")
    out_dir = OUT_DIR / "torchview"
    for fmt in ("pdf", "png"):
        path = graph.visual_graph.render(filename=name, directory=out_dir, format=fmt, cleanup=True)
        print(f"wrote {Path(path).relative_to(PROJECT_ROOT)}")


def export_netron(run: str, input_shape) -> None:
    """ONNX if the exporter supports the model, else TorchScript; the output is checked against PyTorch."""
    model, _ = get_model(run, torch.device("cpu"))
    example = torch.zeros(input_shape)
    probe = torch.rand(input_shape, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        reference = model(probe)
    out_dir = OUT_DIR / "netron"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Single-file output: these models are small (~0.2 M parameters).
        program = torch.onnx.export(model, (example,), dynamo=True, input_names=["incomplete"],
                                    output_names=["completed"], verbose=False)
    except torch.onnx.OnnxExporterError as exc:
        path = out_dir / f"{run}.pt"
        print(f"[{run}] no ONNX translation ({str(exc).splitlines()[0][:80]}...): TorchScript instead")
        torch.jit.trace(model, example).save(str(path))
        with torch.no_grad():
            output = torch.jit.load(str(path))(probe)
    else:
        import onnx
        from onnx.reference import ReferenceEvaluator

        path = out_dir / f"{run}.onnx"
        program.optimize()
        program.save(str(path), external_data=False)
        onnx.checker.check_model(str(path))
        output = torch.from_numpy(ReferenceEvaluator(str(path)).run(None, {"incomplete": probe.numpy()})[0])
    error = (output - reference).abs().max().item()
    print(f"wrote {path.relative_to(PROJECT_ROOT)} (max |file - PyTorch| = {error:.1e}, outputs up to "
          f"{reference.abs().max().item():.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--torchview", nargs="*", default=list(TORCHVIEW_MODELS), choices=TORCHVIEW_MODELS)
    parser.add_argument("--skip-torchview", action="store_true")
    parser.add_argument("--onnx", nargs="*", default=["GCN_L6", "SNN_L6"], help="trained runs exported for Netron")
    args = parser.parse_args()

    config = DBTGeometryConfig()
    input_shape = (1, 1, config.num_views_full, config.det_col_count)

    if not args.skip_torchview:
        if shutil.which("dot") is None:
            sys.exit("graphviz `dot` not found: add it to the PATH (e.g. ~/miniconda3/envs/tex/bin) or use --skip-torchview")
        for name in args.torchview:
            export_torchview(name, input_shape)
    for run in args.onnx:
        export_netron(run, input_shape)


if __name__ == "__main__":
    main()
