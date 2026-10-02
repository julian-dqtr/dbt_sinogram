#!/usr/bin/env python3
"""Why the HLCC moment regression stops at order 4: empirical evidence on the test split.

    .venv/bin/python scripts/hlcc_order_study.py                         # 100 test samples, UNet2D and UNet2dHLCC
    .venv/bin/python scripts/hlcc_order_study.py --models UNet2D LinearInterp --max_order 7

The moment curve a_n(theta) of order n has n + 1 unknown coefficients (src_2D/utils/hlcc.py). The
regression of Huang et al. (2017) fits them on the 51 acquired (noisy) views and evaluates the curve on
the 129 missing ones. Order by order, the three panels show:

  (a) how much the regression amplifies the noise of the measured moments (pure geometry, no data);
  (b) how accurate the regressed moment curves are on the missing views, next to the moments of the
      sinograms completed by the models: the regression is only worth using while it is the more
      accurate of the two;
  (c) what happens to the wedge MSE of a model when its moments of orders 0..K are replaced by the
      regressed ones (the "_P<K>" models of src_2D/models/factory.py).

Outputs:
    outputs/figures/hlcc_order_selection.pdf / .png   the three panels
    outputs/figures/hlcc_moments_by_order.pdf         panel (b) alone (thesis figure)
    outputs/evaluation/hlcc_order_study.csv           every plotted number
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.models.factory import BASELINES, get_model, parse_model_name
from src_2D.utils.hlcc import HLCCMomentProjection, harmonic_basis

torch.backends.cudnn.enabled = False  # Tesla K80, as in evaluate_all.py

# Same visual language as results/model_analysis.ipynb: colour = model family, line style = variant,
# references in ink. Text always uses the ink tokens, never a series colour.
INK, SECONDARY, GRID, AXIS, BAND = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#f0efec"
FAMILY_COLORS = {"Baseline": "#52514e", "U-Net": "#2a78d6", "GCN": "#eb6834", "SNN": "#1baf7a"}
VARIANT_STYLES = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "v")]
REGRESSION = "HLCC regression (from the 51 measured views)"


def family(model_name: str) -> str:
    base, _ = parse_model_name(model_name)
    return "Baseline" if base in BASELINES else "U-Net" if base.startswith("UNet") else base


def series_styles(model_names) -> dict:
    """Colour of the family; the models of one family differ by line style and marker."""
    styles, seen = {}, {}
    for name in model_names:
        index = seen.setdefault(family(name), 0)
        seen[family(name)] += 1
        linestyle, marker = VARIANT_STYLES[index % len(VARIANT_STYLES)]
        styles[name] = {"color": FAMILY_COLORS.get(family(name), SECONDARY), "linestyle": linestyle, "marker": marker}
    return styles


@torch.no_grad()
def predict(model, incomplete: torch.Tensor, device, batch_size: int = 25) -> torch.Tensor:
    batches = [model(incomplete[i:i + batch_size].to(device)).float().cpu() for i in range(0, len(incomplete), batch_size)]
    return torch.cat(batches)


def noise_gain(projection: HLCCMomentProjection) -> np.ndarray:
    """Variance of a regressed moment on a missing view for a unit-variance white noise on the measured
    moments, averaged over the missing views: one number per order."""
    operators = projection.extrapolation[:, ~projection.acquired]  # [K+1, V_missing, V_acquired]
    return (operators**2).sum(dim=2).mean(dim=1).numpy()


def moment_error(moments: torch.Tensor, reference: torch.Tensor) -> np.ndarray:
    """Relative error of the moment curves, per order, pooled over samples and views: [K+1]."""
    return ((moments - reference).pow(2).sum(dim=(0, 1, 2)) / reference.pow(2).sum(dim=(0, 1, 2))).sqrt().numpy()


def signed(value: float, digits: int) -> str:
    """"+3.1 %" / "−9.8 %", with a typographic minus sign."""
    return f"{value:+.{digits}f} %".replace("-", "\u2212")


def style_axis(ax, max_order: int) -> None:
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=9)
    ax.set_xticks(range(max_order + 1))
    ax.set_xlim(-0.4, max_order + 0.4)


def mark_last_useful_order(ax, last_order: int, max_order: int, label: bool = True) -> None:
    """Grey out the orders where the regression is less accurate than the models."""
    if last_order >= max_order:
        return
    ax.axvspan(last_order + 0.5, max_order + 0.4, color=BAND, zorder=0)
    if label:
        ax.text(last_order + 0.6, 0.97, "regression less accurate\nthan the models", transform=ax.get_xaxis_transform(),
                va="top", ha="left", fontsize=8, color=SECONDARY)


def plot_moment_errors(ax, orders, regression_error, model_errors, styles, last_order) -> None:
    mark_last_useful_order(ax, last_order, orders[-1])
    for name, errors in model_errors.items():
        ax.plot(orders, errors, linewidth=1.8, markersize=6.5, markeredgecolor="white", markeredgewidth=1.2,
                label=name, **styles[name])
    ax.plot(orders, regression_error, color=INK, linewidth=2.0, marker="D", markersize=6.5, markeredgecolor="white",
            markeredgewidth=1.2, label=REGRESSION)
    ax.set_yscale("log")
    ax.set_xlabel("HLCC order n (degree of the Chebyshev polynomial along the detector)", fontsize=9, color=SECONDARY)
    ax.set_ylabel("relative error of the moment curve $a_n(\\theta)$\non the missing views", fontsize=9, color=SECONDARY)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    style_axis(ax, orders[-1])


def main():
    parser = argparse.ArgumentParser(description="Empirical choice of the highest order of the HLCC moment regression")
    parser.add_argument("--models", nargs="+", default=["UNet2D", "UNet2dHLCC"])
    parser.add_argument("--num_samples", type=int, default=100, help="Size of the test split")
    parser.add_argument("--max_order", type=int, default=7)
    parser.add_argument("--noise_level", type=float, default=1e5)
    parser.add_argument("--figures_dir", type=Path, default=PROJECT_ROOT / "outputs/figures")
    parser.add_argument("--out_csv", type=Path, default=PROJECT_ROOT / "outputs/evaluation/hlcc_order_study.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DBTGeometryConfig()
    geom = DBTGeometry.from_config(config)
    missing = torch.from_numpy(~geom.acquired_view_mask)
    orders = list(range(args.max_order + 1))

    # --- Data and predictions (inference only) ---
    dataset = SinogramCompletionDataset(args.num_samples, split="test", noise_level=args.noise_level, geometry_config=config)
    samples = [dataset[i] for i in range(len(dataset))]
    incomplete = torch.stack([s[0] for s in samples])  # [N, 1, V, D], noisy measurements
    full = torch.stack([s[1] for s in samples])        # [N, 1, V, D], clean target

    predictions = {}
    for name in args.models:
        try:
            model, _ = get_model(name, device, geometry_config=config)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"[NOT EVALUATED] {name}: {str(exc).splitlines()[0]}")
            continue
        predictions[name] = predict(model, incomplete, device)
    if not predictions:
        raise SystemExit("No model could be loaded: train them first (src_2D/train.py).")
    styles = series_styles(predictions)

    # --- (a) Geometry of the regression: conditioning and noise amplification per order ---
    projection = HLCCMomentProjection(geom, args.max_order)
    gain = noise_gain(projection)
    condition = np.array([torch.linalg.cond(harmonic_basis(geom.angles, n)[~missing]).item() for n in orders])

    # --- (b) Accuracy of the moment curves on the missing views ---
    true_moments = projection.moments(full)[:, :, missing]
    regression_error = moment_error(projection.regressed_moments(incomplete)[:, :, missing], true_moments)
    model_errors = {name: moment_error(projection.moments(pred)[:, :, missing], true_moments) for name, pred in predictions.items()}
    # Last order up to which the regression beats every model on every order.
    better = regression_error < np.min(list(model_errors.values()), axis=0)
    last_order = int(np.argmin(better)) - 1 if not better.all() else args.max_order

    # --- (c) Wedge MSE of every model after the projection of the orders 0..K ---
    def wedge_mse(sinograms: torch.Tensor) -> float:
        return ((sinograms - full)[:, :, missing] ** 2).mean().item()

    mse_before = {name: wedge_mse(pred) for name, pred in predictions.items()}
    mse_change = {name: np.array([100.0 * (wedge_mse(HLCCMomentProjection(geom, K)(pred, incomplete)) / mse_before[name] - 1.0)
                                  for K in orders]) for name, pred in predictions.items()}
    best_order = {name: int(np.argmin(change)) for name, change in mse_change.items()}

    # --- Table (also the accessible twin of the figure) ---
    table = pd.DataFrame({"order": orders, "condition_number": condition, "noise_gain": gain,
                          "moment_error_regression": regression_error})
    for name in predictions:
        table[f"moment_error_{name}"] = model_errors[name]
        table[f"wedge_mse_change_percent_{name}"] = mse_change[name]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    print(f"\n{len(dataset)} test samples, noise I0 = {args.noise_level:g}, acquired window +/-{config.angle_max_deg:g} deg")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3g}"))
    print(f"\nThe regression is more accurate than every model up to order {last_order}.")
    for name in predictions:
        print(f"{name}: wedge MSE {mse_before[name]:.3e}, lowest after projecting the orders 0..{best_order[name]} "
              f"({mse_change[name][best_order[name]]:+.1f} %)")

    # --- Figure ---
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_gain, ax_moments, ax_mse) = plt.subplots(1, 3, figsize=(16.5, 4.8), layout="constrained")

    mark_last_useful_order(ax_gain, last_order, args.max_order, label=False)
    ax_gain.axhline(1.0, color=AXIS, linewidth=1.0)
    ax_gain.text(args.max_order + 0.35, 1.0, "noise kept as it is", va="bottom", ha="right", fontsize=8, color=SECONDARY)
    ax_gain.plot(orders, gain, color=INK, linewidth=2.0, marker="D", markersize=6.5, markeredgecolor="white", markeredgewidth=1.2)
    ax_gain.set_yscale("log")
    ax_gain.set_xlabel("HLCC order n", fontsize=9, color=SECONDARY)
    ax_gain.set_ylabel("variance of a regressed moment / variance of a measured one", fontsize=9, color=SECONDARY)
    ax_gain.set_title("(a) Noise amplification of the regression", fontsize=10.5, color=INK, loc="left")
    style_axis(ax_gain, args.max_order)

    plot_moment_errors(ax_moments, orders, regression_error, model_errors, styles, last_order)
    ax_moments.set_title("(b) Error of the moment curves on the missing views", fontsize=10.5, color=INK, loc="left")

    mark_last_useful_order(ax_mse, last_order, args.max_order, label=False)
    lowest = min(change.min() for change in mse_change.values())
    top = max(5.0, -0.6 * lowest)
    ax_mse.axhline(0.0, color=AXIS, linewidth=1.0)
    for name, change in mse_change.items():
        # Values above the axis are written, not drawn: the line stops where it leaves the plot.
        drawn = np.where(change > top, np.nan, change)
        first_off = int(np.argmax(change > top)) if (change > top).any() else None
        if first_off is not None:
            drawn[first_off] = top
        # The lowest point of every model is named in the legend: labels next to the points collide.
        ax_mse.plot(orders, drawn, linewidth=1.8, markersize=6.5, markeredgecolor="white", markeredgewidth=1.2,
                    markevery=[K for K in orders if change[K] <= top],
                    label=f"{name}: lowest at K = {best_order[name]} ({signed(change[best_order[name]], 1)})", **styles[name])
    off_scale = [f"K = {K}:  " + "  /  ".join(signed(change[K], 0) for change in mse_change.values())
                 for K in orders if any(change[K] > top for change in mse_change.values())]
    if off_scale:
        ax_mse.text(0.02, 0.97, "\n".join(["above the axis (" + " / ".join(mse_change) + "):", *off_scale]),
                    transform=ax_mse.transAxes, ha="left", va="top", fontsize=8, color=SECONDARY, linespacing=1.5)
    ax_mse.set_ylim(1.3 * lowest, 1.05 * top)
    ax_mse.set_xlabel("highest projected order K (orders 0..K replaced by the regression)", fontsize=9, color=SECONDARY)
    ax_mse.set_ylabel("change of the wedge MSE (%)", fontsize=9, color=SECONDARY)
    ax_mse.set_title("(c) Wedge MSE after the projection (below 0: better)", fontsize=10.5, color=INK, loc="left")
    ax_mse.legend(frameon=False, fontsize=8.5, loc="lower left")
    style_axis(ax_mse, args.max_order)

    fig.suptitle(f"HLCC moment regression from a ±{config.angle_max_deg:g}° window: useful up to order {last_order} "
                 f"({len(dataset)} test samples)", fontsize=12, color=INK)
    for suffix in ("pdf", "png"):
        fig.savefig(args.figures_dir / f"hlcc_order_selection.{suffix}", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    plot_moment_errors(ax, orders, regression_error, model_errors, styles, last_order)
    fig.savefig(args.figures_dir / "hlcc_moments_by_order.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(args.figures_dir / "hlcc_moments_by_order.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved {args.figures_dir}/hlcc_order_selection.pdf|png, hlcc_moments_by_order.pdf|png and {args.out_csv}")


if __name__ == "__main__":
    main()
