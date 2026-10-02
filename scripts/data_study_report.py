#!/usr/bin/env python3
"""U-Net vs U-Net + HLCC with few and many training phantoms: paired comparison on the test split.

    MODELS=$(.venv/bin/python scripts/data_study_report.py --print_models)
    .venv/bin/python src_2D/evaluate_all.py --models $MODELS          # per-sample metrics of every run
    .venv/bin/python scripts/data_study_report.py                     # tables + figures
    .venv/bin/python scripts/data_study_report.py --seed 1            # the replicas of SEED=1 (runs <run>_S1)

The runs are those of scripts/launch_unet_hlcc_data_study.sh. For every training-set size, each HLCC
variant is compared with the plain U-Net TRAINED ON THE SAME PHANTOMS:

    + HLCC loss, orders 0-1           the penalty during training (UNet2dHLCC)
    + HLCC projection                 the moment regression of orders 0..4 after the network ("_P4")

The two models are scored on the same test samples, so the comparison is PAIRED: the effect is the
change of the mean metric, its 95 % interval is a bootstrap over the test samples (resampled in pairs)
and the p-value is a paired Wilcoxon test (not corrected for the number of comparisons).

What this interval does NOT contain: the run-to-run variability of a training. Two runs that differ
only by their seed also differ "significantly" on a fixed test set; an effect of a few percent must
keep its sign on the replicas (SEED=1 bash scripts/launch_unet_hlcc_data_study.sh) to be trusted.

Outputs:
    outputs/evaluation/data_study_summary.csv          every number of the tables
    outputs/figures/hlcc_data_study.pdf / .png         change of the wedge MSE vs the plain U-Net, per size
    outputs/figures/hlcc_data_study_curves.pdf / .png  validation curves of the runs (from history.json)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_2D.models.factory import CHECKPOINT_ROOT, run_name_of

# Same visual language as scripts/hlcc_order_study.py: colour = model family (here the U-Net blue),
# the variants differ by position, line style and marker. Text always uses the ink tokens.
INK, SECONDARY, GRID, AXIS, BAND = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#f0efec"
UNET_COLOR = "#2a78d6"
VARIANT_STYLES = [("-", "o"), ("--", "s"), (":", "^")]

BASELINES = ["ZeroFilling", "LinearInterp"]
PROJECTION_TAG = "P4"
# (run prefix, label of the training loss); the first one is the reference of the comparison.
LOSSES = [("UNet2D", "U-Net (MSE only)"), ("UNet2dHLCC", "+ HLCC loss, orders 0–1")]
PROJECTED_LABELS = ["+ HLCC projection", "+ HLCC loss 0–1 and projection"]
IMAGE_METRIC = "img_psnr"
TABLE_METRICS = ["mse_wedge", "psnr_wedge", "ssim_wedge", "img_psnr", "img_ssim"]


def model_name(base: str, size: int, n_high: int, seed: int, projected: bool = False) -> str:
    """Run name of scripts/launch_unet_hlcc_data_study.sh, plus the projection suffix of models/factory.py."""
    name = base + (f"_N{size}" if size != n_high else "") + (f"_S{seed}" if seed != 0 else "")
    return f"{name}_{PROJECTION_TAG}" if projected else name


def variants(size: int, n_high: int, seed: int) -> list:
    """(label, model name) of one training-set size: the reference first, then every HLCC variant."""
    trained = [(label, model_name(base, size, n_high, seed)) for base, label in LOSSES]
    projected = [(label, model_name(base, size, n_high, seed, projected=True)) for (base, _), label in zip(LOSSES, PROJECTED_LABELS)]
    return trained + projected


def paired_effect(candidate: np.ndarray, reference: np.ndarray, rng, relative: bool, n_boot: int = 10_000) -> dict:
    """Effect of a variant on the MEAN of a metric, both models being scored on the same samples.

    relative=True: mean(candidate) / mean(reference) - 1; otherwise mean(candidate) - mean(reference).
    The 95 % interval is a percentile bootstrap over the samples, resampled in pairs.
    """
    def effect(c: np.ndarray, r: np.ndarray):
        return c.mean(axis=-1) / r.mean(axis=-1) - 1.0 if relative else c.mean(axis=-1) - r.mean(axis=-1)

    draws = rng.integers(0, len(reference), size=(n_boot, len(reference)))
    low, high = np.percentile(effect(candidate[draws], reference[draws]), [2.5, 97.5])
    differences = candidate - reference
    p_value = float(wilcoxon(differences).pvalue) if np.any(differences != 0) else 1.0
    return {"effect": float(effect(candidate, reference)), "ci_low": float(low), "ci_high": float(high), "p_value": p_value}


def signed(value: float, digits: int = 1, unit: str = " %") -> str:
    """"+3.1 %" / "−9.8 %", with a typographic minus sign."""
    return f"{value:+.{digits}f}{unit}".replace("-", "−")


def compare_size(metrics: pd.DataFrame, size: int, n_high: int, seed: int, rng) -> pd.DataFrame:
    """One row per evaluated variant of a training-set size, the reference (plain U-Net) first."""
    per_sample = {col: metrics.pivot(index="Sample", columns="Model", values=col) for col in metrics.columns.drop(["Sample", "Model"])}
    names = variants(size, n_high, seed)
    reference = names[0][1]
    available = set(per_sample["mse_wedge"].columns)
    if reference not in available:
        return pd.DataFrame()

    rows = []
    for label, name in names:
        if name not in available:
            continue
        row = {"n_train": size, "variant": label, "model": name}
        row.update({col: per_sample[col][name].mean() for col in per_sample})
        if name != reference:
            mse = paired_effect(per_sample["mse_wedge"][name].to_numpy(), per_sample["mse_wedge"][reference].to_numpy(), rng, relative=True)
            image = paired_effect(per_sample[IMAGE_METRIC][name].to_numpy(), per_sample[IMAGE_METRIC][reference].to_numpy(), rng, relative=False)
            row.update({f"mse_wedge_change_percent{k}": 100.0 * mse[v] for k, v in (("", "effect"), ("_ci_low", "ci_low"), ("_ci_high", "ci_high"))})
            row["mse_wedge_p_value"] = mse["p_value"]
            row["samples_improved_percent"] = 100.0 * (per_sample["mse_wedge"][name] < per_sample["mse_wedge"][reference]).mean()
            row.update({f"{IMAGE_METRIC}_gain_db{k}": image[v] for k, v in (("", "effect"), ("_ci_low", "ci_low"), ("_ci_high", "ci_high"))})
            row[f"{IMAGE_METRIC}_p_value"] = image["p_value"]
        rows.append(row)
    return pd.DataFrame(rows)


def training_curves(size: int, n_high: int, seed: int) -> dict:
    """label -> (history, training stats) of the trained runs of one size whose history.json exists."""
    curves = {}
    for base, label in LOSSES:
        run_dir = CHECKPOINT_ROOT / run_name_of(model_name(base, size, n_high, seed))
        if (run_dir / "history.json").exists() and (run_dir / "training_stats.json").exists():
            curves[label] = (pd.DataFrame(json.loads((run_dir / "history.json").read_text())),
                             json.loads((run_dir / "training_stats.json").read_text()))
    return curves


def print_tables(summary: pd.DataFrame, n_samples: int) -> None:
    for size, table in summary.groupby("n_train", sort=False):
        reference = table.iloc[0]
        print(f"\n=== {size} training phantoms, {n_samples} test samples | reference: {reference['model']} ===")
        shown = pd.DataFrame({"variant": table["variant"], "model": table["model"]})
        shown["mse_wedge"] = table["mse_wedge"].map("{:.3e}".format)
        if len(table) > 1:
            compared = table["mse_wedge_change_percent"].notna()
            shown["change"] = np.where(compared, table["mse_wedge_change_percent"].map("{:+.1f} %".format), "reference")
            shown["95 % interval"] = np.where(compared, [f"[{low:+.1f}, {high:+.1f}]" for low, high in
                                                         zip(table["mse_wedge_change_percent_ci_low"], table["mse_wedge_change_percent_ci_high"])], "")
            shown["p"] = np.where(compared, table["mse_wedge_p_value"].map("{:.1e}".format), "")
            shown["better on"] = np.where(compared, table["samples_improved_percent"].map("{:.0f} %".format), "")
        for col, fmt in (("psnr_wedge", "{:.2f}"), ("ssim_wedge", "{:.4f}"), ("img_psnr", "{:.2f}"), ("img_ssim", "{:.4f}")):
            shown[col] = table[col].map(fmt.format)
        if len(table) > 1:
            shown[f"{IMAGE_METRIC} gain (dB)"] = np.where(compared, [f"{gain:+.2f} [{low:+.2f}, {high:+.2f}]" for gain, low, high in zip(
                table[f"{IMAGE_METRIC}_gain_db"], table[f"{IMAGE_METRIC}_gain_db_ci_low"], table[f"{IMAGE_METRIC}_gain_db_ci_high"])], "")
        print(shown.to_string(index=False))

        residual_cols = [col for col in table.columns if col.startswith("hlcc_residual_")]
        if residual_cols:
            print("HLCC residual per order (0: consistent sinogram)")
            print(table.set_index("model")[residual_cols].to_string(float_format=lambda value: f"{value:.2e}"))


def style_axis(ax) -> None:
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=9)


def plot_changes(summary: pd.DataFrame, n_samples: int, path: Path) -> None:
    """Change of the wedge MSE of every variant relative to the plain U-Net, one panel per training-set size."""
    sizes = list(summary["n_train"].unique())
    labels = [label for _, label in LOSSES[1:]] + PROJECTED_LABELS
    rows = {label: y for y, label in enumerate(labels)}
    compared = summary[summary["mse_wedge_change_percent"].notna()]
    reach = max(5.0, 1.15 * compared[["mse_wedge_change_percent_ci_low", "mse_wedge_change_percent_ci_high"]].abs().to_numpy().max())

    fig, axes = plt.subplots(1, len(sizes), figsize=(5.4 * len(sizes) + 2.2, 3.9), sharex=True, sharey=True,
                             squeeze=False, layout="constrained")
    for ax, size in zip(axes[0], sizes):
        table = summary[summary["n_train"] == size]
        reference = table.iloc[0]
        ax.axvline(0.0, color=AXIS, linewidth=1.0)
        for _, row in table.iloc[1:].iterrows():
            y = rows[row["variant"]]
            ax.plot([row["mse_wedge_change_percent_ci_low"], row["mse_wedge_change_percent_ci_high"]], [y, y],
                    color=UNET_COLOR, linewidth=2.0, solid_capstyle="round")
            ax.plot(row["mse_wedge_change_percent"], y, marker="o", markersize=8, color=UNET_COLOR,
                    markeredgecolor="white", markeredgewidth=1.5)
            ax.annotate(signed(row["mse_wedge_change_percent"]), (row["mse_wedge_change_percent"], y), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)
        ax.set_title(f"{size} training phantoms\nplain U-Net: wedge MSE {reference['mse_wedge']:.2e}",
                     fontsize=10.5, color=INK, loc="left")
        ax.set_xlim(-reach, reach)
        ax.set_ylim(len(labels) - 0.5, -0.7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9.5, color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_xlabel("change of the wedge MSE vs the plain U-Net (%)", fontsize=9, color=SECONDARY)
        ax.text(0.01, 0.01, "← better", transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=SECONDARY)
        ax.text(0.99, 0.01, "worse →", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=SECONDARY)
        style_axis(ax)
    fig.suptitle(f"Effect of the HLCC on a U-Net trained on the same phantoms "
                 f"(point: change of the mean, bar: 95 % interval over {n_samples} test samples)", fontsize=11.5, color=INK)
    for suffix in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_curves(curves: dict, path: Path) -> None:
    """Validation wedge MSE of the trained runs over the epochs, one panel per training-set size."""
    fig, axes = plt.subplots(1, len(curves), figsize=(5.6 * len(curves), 4.4), sharey=True, squeeze=False, layout="constrained")
    styles = {label: style for (_, label), style in zip(LOSSES, VARIANT_STYLES)}
    handles = {}
    for ax, (size, runs) in zip(axes[0], curves.items()):
        for index, (label, (history, _)) in enumerate(runs.items()):
            linestyle, marker = styles[label]
            # Staggered markers keep the curves apart where they overlap.
            handles[label], = ax.plot(history["epoch"], history["val/mse_wedge"], color=UNET_COLOR, linewidth=1.8,
                                      linestyle=linestyle, marker=marker, markersize=6, markeredgecolor="white",
                                      markeredgewidth=1.2, markevery=(10 + 10 * index, 30))
        interpolation = next(iter(runs.values()))[1]["baselines"]["LinearInterp"]["mse_wedge"]
        ax.axhline(interpolation, color=AXIS, linewidth=1.0)
        ax.text(1.0, interpolation, "linear interpolation", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=8, color=SECONDARY)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{size} training phantoms", fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("epoch", fontsize=9, color=SECONDARY)
        ax.grid(color=GRID, linewidth=0.8)
        style_axis(ax)
    axes[0, 0].set_ylabel("wedge MSE on the validation split", fontsize=9, color=SECONDARY)
    ordered = [label for _, label in LOSSES if label in handles]
    fig.legend([handles[label] for label in ordered], ordered, loc="outside lower center", ncol=len(ordered),
               frameon=False, fontsize=9, handlelength=3.5)
    for suffix in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_overfitting(curves: dict) -> None:
    rows = []
    for size, runs in curves.items():
        for label, (history, stats) in runs.items():
            last = history.iloc[-1]
            rows.append({"n_train": size, "loss": label, "best epoch": f"{stats['best']['best_epoch']} / {stats['epochs_run']}",
                         "val mse_wedge (best)": f"{stats['best']['mse_wedge']:.3e}",
                         "train mse (last)": f"{last['train/mse']:.3e}", "val mse (last)": f"{last['val/mse']:.3e}",
                         "val / train": f"{last['val/mse'] / last['train/mse']:.1f}"})
    print("\n=== Training curves (mse: all the views; a large val / train ratio means overfitting) ===")
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Paired comparison of the U-Net / HLCC data study on the test split")
    parser.add_argument("--metrics", type=Path, default=PROJECT_ROOT / "outputs/evaluation/model_comparison_metrics.csv",
                        help="Per-sample metrics written by src_2D/evaluate_all.py")
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 2000], help="Training-set sizes, in phantoms")
    parser.add_argument("--n_high", type=int, default=2000, help="Size of the runs without the _N<n> suffix")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the runs (another seed: runs <run>_S<seed>)")
    parser.add_argument("--print_models", action="store_true", help="Only print the model names to pass to evaluate_all.py")
    parser.add_argument("--figures_dir", type=Path, default=PROJECT_ROOT / "outputs/figures")
    parser.add_argument("--out_csv", type=Path, default=PROJECT_ROOT / "outputs/evaluation/data_study_summary.csv")
    args = parser.parse_args()

    expected = [name for size in args.sizes for _, name in variants(size, args.n_high, args.seed)]
    if args.print_models:
        print(" ".join(BASELINES + expected))
        return

    metrics = pd.read_csv(args.metrics)
    metrics = metrics[metrics["Sample"] >= 0]
    missing = [name for name in expected if name not in set(metrics["Model"])]
    if missing:
        print(f"[NOT IN {args.metrics.name}] {' '.join(missing)}")
    rng = np.random.default_rng(0)
    tables = [compare_size(metrics, size, args.n_high, args.seed, rng) for size in args.sizes]
    if all(table.empty for table in tables):
        raise SystemExit("No plain U-Net of the study in the metrics file: run src_2D/evaluate_all.py first (see --print_models).")
    summary = pd.concat(tables, ignore_index=True)
    n_samples = metrics["Sample"].nunique()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    print_tables(summary, n_samples)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    saved = [str(args.out_csv)]
    if "mse_wedge_change_percent" in summary and summary["mse_wedge_change_percent"].notna().any():
        plot_changes(summary, n_samples, args.figures_dir / "hlcc_data_study")
        saved.append(f"{args.figures_dir}/hlcc_data_study.pdf|png")
    curves = {size: runs for size in args.sizes if (runs := training_curves(size, args.n_high, args.seed))}
    if curves:
        print_overfitting(curves)
        plot_curves(curves, args.figures_dir / "hlcc_data_study_curves")
        saved.append(f"{args.figures_dir}/hlcc_data_study_curves.pdf|png")
    print("\nSaved " + ", ".join(saved))


if __name__ == "__main__":
    main()
