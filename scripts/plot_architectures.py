#!/usr/bin/env python3
"""Architecture figures of the learned models, drawn with PlotNeuralNet (LaTeX / TikZ).

    python scripts/plot_architectures.py                      # every figure
    python scripts/plot_architectures.py --model viewgraph    # one figure
    python scripts/plot_architectures.py --no-compile         # only write the .tex
    CUDA_VISIBLE_DEVICES=0 python scripts/plot_architectures.py --sinograms   # re-render the sinogram images first

- arch_unet:         SinogramUNet. UNet2D and UNet2dHLCC share this exact network (only the loss differs).
- arch_unet_hlcc_p4: UNet2dHLCC_P4, the U-Net trained with the HLCC penalty followed by the HLCC moment regression.
- arch_viewgraph:    ViewGraphNet. SinoGCN and SinoSheafNet share it, only the fixed transport T_ij differs.

The U-Net figures show the real input / outputs of test sample --sample (PNGs in
results/figures/architectures/sinograms/, rendered by --sinograms: needs ASTRA, a GPU and the checkpoints).
Without these PNGs, plain boxes are drawn instead.

Channel counts and tensor sizes are read from DEFAULT_MODEL_CONFIG and DBTGeometryConfig, so the
figures follow the models. PlotNeuralNet is not vendored, clone it once:

    git clone --depth 1 https://github.com/HarisIqbal88/PlotNeuralNet tools/PlotNeuralNet

The .tex sources are written to results/figures/architectures/ and compiled with the first of
tectonic / pdflatex found on the PATH (or --latex); the PDFs go to results/figures/.
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.models.factory import DEFAULT_MODEL_CONFIG

PLOTNN_ROOT = PROJECT_ROOT / "tools/PlotNeuralNet"
TEX_DIR = PROJECT_ROOT / "results/figures/architectures"
SINO_DIR = TEX_DIR / "sinograms"
PDF_DIR = PROJECT_ROOT / "results/figures"

if not (PLOTNN_ROOT / "pycore/tikzeng.py").exists():
    sys.exit(f"PlotNeuralNet not found in {PLOTNN_ROOT}, clone it first (see the docstring of this script).")
sys.path.insert(0, str(PLOTNN_ROOT))
from pycore.tikzeng import to_begin, to_connection, to_end, to_head, to_skip  # noqa: E402

COLORS = r"""
\usetikzlibrary{calc,decorations.pathreplacing,fit}
\def\IOColor{rgb:white,4;black,1}
\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ActColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\UpColor{rgb:blue,2;green,1;black,0.3}
\def\CopyColor{rgb:white,1;black,3}
\def\OutColor{rgb:magenta,5;black,7}
\def\GraphColor{rgb:blue,5;green,2.5;white,5}
\def\GraphActColor{rgb:blue,5;green,5;white,3}
\def\SumColor{rgb:blue,5;green,15}
\def\DCColor{rgb:red,2;blue,5;white,2}
\def\HLCCColor{rgb:green,3;blue,1;white,8}
"""

# Path from the input to the residual / data-consistency nodes, drawn below the network; operators of the
# 2D flowcharts.
BUS_STYLE = r"""
\tikzstyle{bus}=[ultra thick,-Stealth,draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7,rounded corners=4pt]
\tikzstyle{op}=[draw=black!60,fill=\HLCCColor,rounded corners=3pt,align=flush center,font=\small,inner sep=4pt]
"""


# ---------------------------------------------------------------------------
# TikZ helpers (same pics as pycore.tikzeng, with free colours and labels)
# ---------------------------------------------------------------------------

def labels(*values) -> str:
    """xlabel / width list of a PlotNeuralNet pic: {{"32","32"}}."""
    return "{{" + ",".join(f'"{v}"' for v in values) + ",}}"


def pic(kind: str, name: str, at: str, offset: Tuple[float, float, float] = (0, 0, 0), **opts) -> str:
    body = ",\n        ".join(f"{key}={value}" for key, value in opts.items())
    dx, dy, dz = offset
    return f"\n\\pic[shift={{({dx},{dy},{dz})}}] at ({at})\n    {{{kind}={{name={name},\n        {body}}}}};\n"


def box(name, at, channels="", fill=r"\ConvColor", width=2.0, height=40.0, depth=40.0, size="",
        caption="", offset=(0, 0, 0), opacity=0.6) -> str:
    return pic("Box", name, at, offset, caption=f"{{{caption}}}", xlabel=labels(channels), zlabel=f"{{{size}}}",
               fill=fill, opacity=opacity, height=height, width=width, depth=depth)


def banded_box(name, at, channels: Sequence, fill=r"\ConvColor", bandfill=r"\ActColor", width=2.0,
               height=40.0, depth=40.0, size="", caption="", offset=(0, 0, 0), opacity=0.4) -> str:
    """Conv(s) followed by a normalisation / activation band, one sub-box per entry of ``channels``."""
    widths = "{" + ",".join(str(width) for _ in channels) + "}"
    return pic("RightBandedBox", name, at, offset, caption=f"{{{caption}}}", xlabel=labels(*channels),
               zlabel=f"{{{size}}}", fill=fill, bandfill=bandfill, opacity=opacity, height=height, width=widths,
               depth=depth)


def ball(name, at, logo, fill=r"\SumColor", radius=2.5, offset=(0, 0, 0), caption="") -> str:
    return pic("Ball", name, at, offset, fill=fill, opacity=0.6, radius=radius, logo=f"{{{logo}}}",
               caption=f"{{{caption}}}")


def io_label(name: str, title: str, size: str) -> str:
    """Caption of an input / output box (the built-in caption wraps on thin boxes)."""
    return rf"""
\node[below=12pt,align=center,font=\small] at ({name}-nearsoutheast) {{\textbf{{{title}}}\\{size}}};
"""


def image_slab(name: str, at: str, png: Optional[Path], height: float, depth: float,
               offset=(0, 0, 0)) -> str:
    """Thin box whose east face shows a sinogram (views along the height, detector along the depth).
    ``png`` is None (or missing): plain box."""
    return slab_box(name, at, height, depth, offset) + slab_image(name, png, height, depth)


def slab_box(name: str, at: str, height: float, depth: float, offset=(0, 0, 0)) -> str:
    return box(name, at, fill=r"\IOColor", width=0.4, height=height, depth=depth, offset=offset)


def slab_image(name: str, png: Optional[Path], height: float, depth: float) -> str:
    """The sinogram on the east face of the slab ``name`` (drawn on top of what precedes it)."""
    if png is None or not png.exists():
        return ""
    # 0.2: scale of the PlotNeuralNet pics (box units -> cm)
    return rf"""
\begin{{scope}}[shift={{({name}-east)}}]
    \node[canvas is zy plane at x=0,transform shape,inner sep=0pt] at (0,0)
        {{\includegraphics[width={depth * 0.2:.3f}cm,height={height * 0.2:.3f}cm]{{{png.relative_to(TEX_DIR).as_posix()}}}}};
\end{{scope}}
"""


def sinogram(name: str) -> Optional[Path]:
    path = SINO_DIR / f"{name}.png"
    return path if path.exists() else None


def legend(anchor: str, entries: Sequence[Tuple[str, str, str]], columns: int = 4, col_width: float = 6.0) -> str:
    """Legend below the figure. entries: (kind, colour, text), kind in box / band / ball / arrow
    (colour "fill|band" for a band)."""
    lines = [f"\n\\begin{{scope}}[shift={{({anchor})}}]"]
    for idx, (kind, color, text) in enumerate(entries):
        x, y = (idx % columns) * col_width, -(idx // columns) * 0.8
        if kind == "box":
            lines.append(rf"\filldraw[fill={color},fill opacity=0.7,draw=black!60] ({x},{y}) rectangle ++(0.6,0.4);")
        elif kind == "band":
            fill, band = color.split("|")
            lines.append(rf"\filldraw[fill={fill},fill opacity=0.7,draw=black!60] ({x},{y}) rectangle ++(0.45,0.4);")
            lines.append(rf"\filldraw[fill={band},fill opacity=0.7,draw=black!60] ({x + 0.45},{y}) rectangle ++(0.15,0.4);")
        elif kind == "ball":
            lines.append(rf"\shade[ball color={color}] ({x + 0.3},{y + 0.2}) circle (0.22);")
        elif kind == "arrow":
            lines.append(rf"\draw[bus] ({x},{y + 0.2}) -- ++(0.6,0);")
        lines.append(rf"\node[anchor=west,font=\small] at ({x + 0.7},{y + 0.2}) {{{text}}};")
    lines.append(r"\end{scope}")
    return "\n".join(lines) + "\n"


def channel_width(channels: int, scale: float) -> float:
    """Box thickness grows with log2(channels) so that wide layers stay readable."""
    return round(max(0.6, scale * math.log2(max(channels, 1))), 2)


def times(rows: int, cols: int) -> str:
    return rf"${rows}\times{cols}$"


def document(body: List[str], tex_dir: Path) -> List[str]:
    layers_root = os.path.relpath(PLOTNN_ROOT, tex_dir)
    return [to_head(layers_root), COLORS, to_begin(), BUS_STYLE, *body, to_end()]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def unet_figure(filters: int, views: int, detectors: int) -> List[str]:
    """SinogramUNet: CustomUNet (3 poolings, Resize+Conv decoder), residual output, hard data consistency."""
    padded = math.ceil(views / 8) * 8  # CustomUNet pads the views to a multiple of 8 (replicate)
    levels = range(4)
    chans = [filters * 2**lvl for lvl in levels]
    size = [times(padded // 2**lvl, detectors // 2**lvl) for lvl in levels]
    height = [round(40 * 0.75**lvl, 1) for lvl in levels]
    depth = [round(h * detectors / padded, 1) for h in height]
    width = [channel_width(c, 0.75) for c in chans]

    arch = [image_slab("in", "0,0,0", sinogram("incomplete"), height[0], depth[0]),
            io_label("in", "Incomplete $y$", times(views, detectors))]

    # Encoder: ConvBlock (2 x [conv 3x3, InstanceNorm, LeakyReLU]) then max pooling
    prev = "in"
    for lvl in levels:
        name = f"enc{lvl}"
        arch.append(banded_box(name, f"{prev}-east", (chans[lvl], chans[lvl]), width=width[lvl], height=height[lvl],
                               depth=depth[lvl], size=size[lvl], offset=(2.2 if lvl == 0 else 1.2, 0, 0),
                               caption="Bottleneck" if lvl == 3 else ""))
        arch.append(to_connection(prev, name))
        prev = name
        if lvl < 3:
            arch.append(box(f"pool{lvl}", f"{name}-east", fill=r"\PoolColor", width=1, height=height[lvl + 1],
                            depth=depth[lvl + 1], opacity=0.5))
            prev = f"pool{lvl}"

    # Decoder: bilinear x2 + conv 3x3, concatenation with the skip, ConvBlock
    for lvl in (2, 1, 0):
        arch += [
            box(f"up{lvl}", f"{prev}-east", chans[lvl], fill=r"\UpColor", width=width[lvl], height=height[lvl],
                depth=depth[lvl], offset=(1.2, 0, 0), opacity=0.5),
            box(f"copy{lvl}", f"up{lvl}-east", chans[lvl], fill=r"\CopyColor", width=width[lvl], height=height[lvl],
                depth=depth[lvl], opacity=0.25),
            banded_box(f"dec{lvl}", f"copy{lvl}-east", (chans[lvl], chans[lvl]), width=width[lvl],
                       height=height[lvl], depth=depth[lvl]),
            to_connection(prev, f"up{lvl}"),
            to_skip(of=f"enc{lvl}", to=f"copy{lvl}", pos=1.25),
        ]
        prev = f"dec{lvl}"

    # Output: conv 1x1, residual on the input, data consistency (measured views copied back)
    arch += [
        box("head", "dec0-east", 1, fill=r"\OutColor", width=0.8, height=height[0], depth=depth[0],
            offset=(1.2, 0, 0)),
        ball("sum", "head-east", "$+$", offset=(1.6, 0, 0)),
        ball("dc", "sum-east", r"\scriptsize\textsf{DC}", fill=r"\DCColor", offset=(1.6, 0, 0)),
        image_slab("out", "dc-east", sinogram("unet2d"), height[0], depth[0], offset=(1.6, 0, 0)),
        io_label("out", r"Completed $\hat y$", times(views, detectors)),
        to_connection("dec0", "head"),
        to_connection("head", "sum"),
        to_connection("sum", "dc"),
        to_connection("dc", "out"),
        r"""
\draw[bus] ($(in-east)+(0.9,0)$) -- ++(0,-7.6) coordinate (bus) -- (bus -| sum-south)
    node[pos=0.5,below,font=\small,opacity=1] {measured sinogram $y$: residual learning and data consistency}
    -- (sum-south);
\draw[bus] (bus -| sum-south) -- (bus -| dc-south) -- (dc-south);
""",
        legend("$(in-west)+(0,-9.2)$", [
            ("band", r"\ConvColor|\ActColor", r"conv $3\times3$ + InstanceNorm + LeakyReLU"),
            ("box", r"\PoolColor", r"max pooling $2\times2$"),
            ("box", r"\UpColor", r"bilinear $\times2$ + conv $3\times3$"),
            ("box", r"\CopyColor", r"skip connection (concatenation)"),
            ("box", r"\OutColor", r"conv $1\times1$"),
            ("ball", r"\SumColor", r"residual: $y + f_\theta(y)$"),
            ("ball", r"\DCColor", r"data consistency (measured views kept)"),
            ("arrow", "", r"copy of the input"),
        ], columns=4, col_width=7.2),
    ]
    return arch


def mini_unet(prev: str, views: int, detectors: int, height0: float) -> List[str]:
    """Schematic SinogramUNet without labels (the detailed one is arch_unet): menc0..menc3, mdec2..mdec0."""
    padded = math.ceil(views / 8) * 8
    height = [round(height0 * 0.75**lvl, 1) for lvl in range(4)]
    depth = [round(h * detectors / padded, 1) for h in height]
    arch = []
    for lvl in range(4):
        name = f"menc{lvl}"
        arch += [banded_box(name, f"{prev}-east", ("", ""), width=0.6 + 0.3 * lvl, height=height[lvl],
                            depth=depth[lvl], offset=(2.0 if lvl == 0 else 0.7, 0, 0)),
                 to_connection(prev, name)]
        prev = name
    for lvl in (2, 1, 0):
        name = f"mdec{lvl}"
        arch += [banded_box(name, f"{prev}-east", ("", ""), width=0.6 + 0.3 * lvl, height=height[lvl],
                            depth=depth[lvl], offset=(0.7, 0, 0)),
                 to_connection(prev, name),
                 to_skip(of=f"menc{lvl}", to=name, pos=1.25)]
        prev = name
    return arch


def unet_hlcc_p4_figure(views: int, detectors: int, max_order: int) -> List[str]:
    """UNet2dHLCC_P<K>: U-Net trained with MSE + HLCC penalty (orders 0-1), then the HLCC moment regression of
    orders 0..K (``HLCCMomentProjection``: no trainable parameter, applied at inference only).

    Layout: the network on the first row (left to right), the moment regression below it (right to left),
    so that the output comes back under the input."""
    height = 24.0
    depth = round(height * detectors / views, 1)
    correction = sinogram(f"p{max_order}_correction_2d")
    image = (r"\\[3pt]\includegraphics[height=1.9cm]{" + correction.relative_to(TEX_DIR).as_posix() + "}"
             r"\\[-1pt]{\scriptsize $M\odot\delta$ on the test sample}") if correction else ""
    flowchart = r"""
% Network part: the bus from the input feeds the residual, the data consistency and the moment regression
\draw[decorate,decoration={brace,mirror,amplitude=6pt},thick]
    ($(menc0-west |- menc0-south)+(-0.7,-1.4)$) -- ($(mdec0-east |- menc0-south)+(0,-1.4)$)
    node[midway,below=8pt,font=\small] {U-Net $f_\theta$ (same network as UNet2D)};
\path ($(in-east)+(0.9,0)$) coordinate (b0);
\draw[bus] (b0) -- ++(0,-5.0) coordinate (b1) -| (sum-south);
\draw[bus] (b1 -| sum-south) -| (dc-south);
\node[anchor=west,align=left,font=\small] at ($(yhat-east)+(1.0,0)$) {\textbf{UNet2dHLCC}\\$\hat y$};

% Training loss (UNet2dHLCC only)
\node[draw=black!60,dashed,rounded corners=3pt,align=center,font=\small,anchor=south east] (loss)
    at ($(yhat-north)+(1.6,2.0)$)
    {\textbf{training only}: $\mathcal L=\mathrm{MSE}(\hat y,y^\star)+\alpha(t)\,(\lambda_0L_0+\lambda_1L_1)$\\
     HLCC penalty $L_n=\|(I-\Pi_{H_n})\,a^{\hat y}_n\|^2/V$, annealed by $\alpha(t)$};
\draw[bus,dashed] (yhat-north) -- (yhat-north |- loss.south);

% HLCC moment regression P_K: first row, left to right
\node[op,text width=4.8cm,anchor=north west] (my) at ($(in-west)+(-0.8,-7.0)$)
    {moments of the measured views\\
     $a^{y}_n(\theta)=\sum_s y(\theta,s)\,U_n(s/S)$\\
     $\theta\in\Theta_{\mathrm{acq}}$, $n=0,\dots,<K>$};
\draw[bus] (b1) -- (b1 |- my.north);
\node[op,text width=5.9cm,right=0.6cm of my] (reg)
    {harmonic regression\\ fit $\tilde a_n\in H_n$ on $\Theta_{\mathrm{acq}}$, evaluate on all views\\
     {\scriptsize $H_n=\mathrm{span}\{\cos m\theta,\sin m\theta\}_{m\le n,\ m\equiv n\,(2)}$}};
\node[op,text width=2.9cm] (myhat) at (yhat-south |- reg.center)
    {moments of $\hat y$\\ $a^{\hat y}_n(\theta)$, all the views};
\node[circle,draw=black!60,fill=\HLCCColor,inner sep=1pt,font=\small] (delta)
    at ($(reg.east)!0.5!(myhat.west)$) {$-$};
\draw[bus] (yhat-south) -- (myhat.north);
\draw[bus] (my) -- (reg);
\draw[bus] (reg) -- node[above,font=\small,opacity=1] {$\tilde a_n$} (delta);
\draw[bus] (myhat) -- node[above,font=\small,opacity=1] {$a^{\hat y}_n$} (delta);

% second row, right to left
\path (delta) ++(0,-4.1) node[op,text width=5.0cm] (corr)
    {correction of the missing views\\
     $\delta(\theta,s)=\sum_n c_n(\theta)\,W(s)\,U_n(s/S)$\\
     $W(s)=\sqrt{1-(s/S)^2}$\\ $c_n$: the corrected moments are $\tilde a_n$};
\draw[bus] (delta) -- node[right,font=\small,opacity=1] {$\Delta a_n=\tilde a_n-a^{\hat y}_n$} (corr);
\node[op,text width=4.6cm,left=0.6cm of corr] (upd)
    {$\hat y_{P_<K>}=\hat y+M\odot\delta$\\
     $M$: missing views inside the\\ shadow of the object\\ ($\hat y>0.03\max_s\hat y$)<IMAGE>};
\draw[bus] (corr) -- (upd);
\coordinate (title) at ($(my.north west)+(0,0.75)$);
"""
    flowchart = flowchart.replace("<K>", str(max_order)).replace("<IMAGE>", image)

    return [
        image_slab("in", "0,0,0", sinogram("incomplete"), height, depth),
        io_label("in", "Incomplete $y$", times(views, detectors)),
        *mini_unet("in", views, detectors, height),
        ball("sum", "mdec0-east", "$+$", offset=(1.4, 0, 0)),
        ball("dc", "sum-east", r"\scriptsize\textsf{DC}", fill=r"\DCColor", offset=(1.4, 0, 0)),
        slab_box("yhat", "dc-east", height, depth, offset=(1.4, 0, 0)),
        to_connection("mdec0", "sum"),
        to_connection("sum", "dc"),
        to_connection("dc", "yhat"),
        slab_image("yhat", sinogram("unet2dhlcc"), height, depth),
        flowchart,
        # Output under the input: the arrow arrives on the east face, under the image
        slab_box("yp4", "$(upd.west)+(-2.6,0)$", height, depth),
        r"""
\draw[connection] (upd.west) -- node {\midarrow} (yp4-east);
""",
        slab_image("yp4", sinogram(f"unet2dhlcc_p{max_order}"), height, depth),
        io_label("yp4", rf"UNet2dHLCC\_P{max_order} $\hat y_{{P_{max_order}}}$", times(views, detectors)),
        rf"""
\node[draw=black!50,dashed,rounded corners,inner sep=0.3cm,
    fit=(title)(my)(reg)(myhat)(corr)(upd)(yp4-farnorthwest)(yp4-nearsoutheast),
    label={{[anchor=north west,font=\small\bfseries,inner sep=8pt]north west:HLCC moment regression $P_{max_order}$
    (orders $0,\dots,{max_order}$, no trainable parameter, inference only)}}] {{}};
""",
    ]


def viewgraph_figure(num_stalks: int, k: int, views: int, detectors: int) -> List[str]:
    """ViewGraphNet: nodes = views, features [2F, V, D]; residual blocks of detector convs + graph conv."""
    channels, hidden = 2 * num_stalks, max(1, num_stalks // 2)
    height = 32.0
    depth = round(height * detectors / views, 1)
    w = {c: channel_width(c, 0.6) for c in (1, 2, hidden, num_stalks, channels)}
    common = dict(height=height, depth=depth)

    arch = [
        box("in", "0,0,0", 2, fill=r"\IOColor", width=w[2], **common),
        io_label("in", r"$y$ + acquired flag", times(views, detectors)),
        banded_box("enc", "in-east", (channels,), width=w[channels], offset=(2.4, 0, 0), caption="Encoder",
                   opacity=0.9, **common),
        to_connection("in", "enc"),
    ]

    # Residual ViewGraphBlocks: [conv 1x3, ViewNorm, GELU] [graph conv, ViewNorm, GELU] [conv 1x3] (+)
    prev = "enc"
    for idx, block in enumerate(("1", "2", "L")):
        b = f"b{block}"
        after_dots = block == "L"
        arch += [
            banded_box(f"{b}c1", f"{prev}-east", (channels,), width=w[channels],
                       offset=(3.2 if after_dots else 1.2, 0, 0), opacity=0.9, **common),
            banded_box(f"{b}g", f"{b}c1-east", (channels,), fill=r"\GraphColor", bandfill=r"\GraphActColor",
                       width=w[channels], caption=f"Block ${block}$", opacity=0.9, **common),
            box(f"{b}c2", f"{b}g-east", channels, width=w[channels], opacity=0.9, **common),
            ball(f"{b}s", f"{b}c2-east", "$+$", radius=1.8, offset=(1.0, 0, 0)),
            to_connection(f"{b}c2", f"{b}s"),
            rf"""
\draw[bus] ($({b}c1-west)+(-0.5,0)$) -- ++(0,{height * 0.1 + 0.9}) -| ({b}s-north);
""",
        ]
        if after_dots:
            arch.append(rf"""
\draw[connection] ({prev}-east) -- node[fill=white,text opacity=1,fill opacity=1] {{\Large$\cdots$}} ({b}c1-west);
""")
        else:
            arch.append(to_connection(prev, f"{b}c1"))
        prev = f"{b}s"

    arch += [
        rf"""
\draw[decorate,decoration={{brace,mirror,amplitude=8pt}},thick]
    ($(b1c1-west |- b1c1-south)+(-0.6,-2.3)$) -- ($(bLs-east |- b1c1-south)+(0.2,-2.3)$)
    node[midway,below=10pt,font=\small] {{$\times L$ blocks: angular reach $L\,k/2 = {k // 2}L$ views}};
""",
        # Decoder: [conv 1x3, ViewNorm, GELU] [conv 1x3, GELU] [conv 1x3]
        banded_box("dec1", "bLs-east", (num_stalks,), width=w[num_stalks], offset=(1.4, 0, 0), caption="Decoder",
                   opacity=0.9, **common),
        banded_box("dec2", "dec1-east", (hidden,), width=w[hidden], opacity=0.9, **common),
        box("dec3", "dec2-east", 1, width=w[1], opacity=0.9, **common),
        ball("dc", "dec3-east", r"\scriptsize\textsf{DC}", fill=r"\DCColor", offset=(1.6, 0, 0)),
        box("out", "dc-east", 1, fill=r"\IOColor", width=w[1], offset=(1.6, 0, 0), **common),
        io_label("out", "Completed", times(views, detectors)),
        to_connection("bLs", "dec1"),
        to_connection("dec3", "dc"),
        to_connection("dc", "out"),
        r"""
\draw[bus] (in-south) -- ++(0,-5.6) -| (dc-south)
    node[pos=0.25,below,font=\small,opacity=1] {measured sinogram $y$: data consistency};
""",
        # Inset: message passing of one graph conv, on the view graph
        rf"""
\begin{{scope}}[shift={{($(b1g-north)+(1.0,6.2)$)}}]
    \draw[dashed,rounded corners,black!50] (-5.0,-1.9) rectangle (5.0,2.6);
    \foreach \j in {{0,...,6}} {{
        \node[circle,draw=black!70,fill=\GraphColor,minimum size=0.42cm,inner sep=0pt] (v\j) at ({{1.15*(\j-3)}},0) {{}};
    }}
    \node[circle,draw=black,very thick,fill=\GraphActColor,minimum size=0.42cm,inner sep=0pt] at (v3) {{}};
    \node at (-4.3,0) {{$\cdots$}};  \node at (4.3,0) {{$\cdots$}};
    \foreach \j in {{0,1,2}} {{ \draw[-Stealth,thick,draw=\edgecolor] (v\j) to[bend left=50] (v3); }}
    \foreach \j in {{4,5,6}} {{ \draw[-Stealth,thick,draw=\edgecolor] (v\j) to[bend right=50] (v3); }}
    \node[below=1pt,font=\small] at (v3.south) {{$i$}};
    \node[font=\small] at (0,2.05) {{graph conv on the view graph ($k/2={k // 2}$ neighbours per side)}};
    \node[font=\small] at (0,-0.85) {{$m_i=\sum_{{j}}\hat A_{{ij}}\,T_{{ij}}\,x_j, \qquad x_i \leftarrow \mathrm{{Conv}}_{{1\times1}}([x_i;\,m_i])$}};
    \node[font=\small] at (0,-1.45) {{GCN: $T_{{ij}}=I$ \qquad SNN: $T_{{ij}}=S\big(t_f(\theta_i-\theta_j)\big)$ (shift along the detector)}};
\end{{scope}}
\draw[dashed,black!50] ($(b1g-north)+(1.0,4.3)$) -- (b1g-north);
""",
        legend("$(in-west)+(0,-10.4)$", [
            ("band", r"\ConvColor|\ActColor", r"conv $1\times3$ along the detector + (ViewNorm +) GELU"),
            ("band", r"\GraphColor|\GraphActColor", r"graph conv between views + ViewNorm + GELU"),
            ("ball", r"\SumColor", r"residual connection"),
            ("ball", r"\DCColor", r"data consistency (measured views kept)"),
            ("arrow", "", r"identity path"),
        ], columns=3, col_width=9.5),
    ]
    return arch


# ---------------------------------------------------------------------------
# Sinogram images
# ---------------------------------------------------------------------------

def render_sinograms(sample_index: int, noise_level: float, max_order: int) -> None:
    """PNGs of the test sample shown by the U-Net figures: input, outputs and the P_K correction.
    Needs ASTRA (GPU projector of the dataset) and the UNet2D / UNet2dHLCC checkpoints; the models run on CPU."""
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from src_2D.data.dataset_2d import SinogramCompletionDataset
    from src_2D.models.factory import get_model

    dataset = SinogramCompletionDataset(sample_index + 1, split="test", noise_level=noise_level,
                                        geometry_config=DBTGeometryConfig())
    incomplete, full, _ = dataset[sample_index]
    device = torch.device("cpu")
    unet, _ = get_model("UNet2D", device)
    hlcc_pk, _ = get_model(f"UNet2dHLCC_P{max_order}", device)
    with torch.no_grad():
        x = incomplete.unsqueeze(0)
        outputs = {"unet2d": unet(x), "unet2dhlcc": hlcc_pk.model(x), f"unet2dhlcc_p{max_order}": hlcc_pk(x)}

    def save(name: str, sino: torch.Tensor, slab: bool = True, **style) -> None:
        # [V, D] with the first view (-90 deg) at the bottom. On a slab the detector axis is mirrored: the depth
        # axis of PlotNeuralNet points towards the reader, which mirrors the image back.
        image = np.flipud(sino.squeeze().cpu().numpy())
        plt.imsave(SINO_DIR / f"{name}.png", np.fliplr(image) if slab else image, **style)
        print(f"wrote {(SINO_DIR / name).relative_to(PROJECT_ROOT)}.png")

    SINO_DIR.mkdir(parents=True, exist_ok=True)
    vmax = float(full.max())
    save("incomplete", incomplete, cmap="bone", vmin=0.0, vmax=vmax)
    for name, out in outputs.items():
        save(name, out, cmap="bone", vmin=0.0, vmax=vmax)
    correction = outputs[f"unet2dhlcc_p{max_order}"] - outputs["unet2dhlcc"]
    limit = float(correction.abs().max())
    save(f"p{max_order}_correction_2d", correction, slab=False, cmap="RdBu_r", vmin=-limit, vmax=limit)


# ---------------------------------------------------------------------------
# Generation / compilation
# ---------------------------------------------------------------------------

def find_latex(requested: str | None) -> str | None:
    if requested:
        return requested
    return shutil.which("tectonic") or shutil.which("pdflatex")


def compile_tex(tex_path: Path, latex: str) -> Path:
    """Compile in the .tex folder (the \\subimport path is relative to it) and move the PDF to PDF_DIR."""
    if Path(latex).name == "tectonic":
        cmd = [latex, "-X", "compile", tex_path.name]
    else:
        cmd = [latex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    result = subprocess.run(cmd, cwd=tex_path.parent, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
    for ext in (".aux", ".log"):
        tex_path.with_suffix(ext).unlink(missing_ok=True)
    pdf_path = PDF_DIR / tex_path.with_suffix(".pdf").name
    shutil.move(tex_path.with_suffix(".pdf"), pdf_path)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("unet", "unet_hlcc_p4", "viewgraph", "all"), default="all")
    parser.add_argument("--latex", default=None, help="LaTeX compiler (default: tectonic or pdflatex from the PATH)")
    parser.add_argument("--no-compile", action="store_true", help="only write the .tex sources")
    parser.add_argument("--sinograms", action="store_true", help="re-render the sinogram images (ASTRA + checkpoints)")
    parser.add_argument("--sample", type=int, default=0, help="test sample shown by the U-Net figures")
    parser.add_argument("--noise_level", type=float, default=1e5, help="as in evaluate_all.py")
    args = parser.parse_args()

    max_order = 4  # the "_P4" models of the protocol
    if args.sinograms:
        render_sinograms(args.sample, args.noise_level, max_order)

    geometry = DBTGeometryConfig()
    views, detectors = geometry.num_views_full, geometry.det_col_count
    unet_cfg, graph_cfg = DEFAULT_MODEL_CONFIG["UNet2D"], DEFAULT_MODEL_CONFIG["SNN"]
    figures = {
        "unet": lambda: unet_figure(unet_cfg["filters"], views, detectors),
        "unet_hlcc_p4": lambda: unet_hlcc_p4_figure(views, detectors, max_order),
        "viewgraph": lambda: viewgraph_figure(graph_cfg["num_stalks"], graph_cfg["k"], views, detectors),
    }
    selected = figures if args.model == "all" else {args.model: figures[args.model]}

    latex = None if args.no_compile else find_latex(args.latex)
    if not args.no_compile and latex is None:
        print("No LaTeX compiler found (tectonic / pdflatex): only the .tex sources are written.")

    TEX_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in selected.items():
        tex_path = TEX_DIR / f"arch_{name}.tex"
        tex_path.write_text("".join(document(build(), TEX_DIR)))
        print(f"wrote {tex_path.relative_to(PROJECT_ROOT)}")
        if latex is not None:
            print(f"wrote {compile_tex(tex_path, latex).relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
