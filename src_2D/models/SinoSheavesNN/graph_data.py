"""View graph of a sinogram: one node per projection angle.

The graph is fixed by the acquisition geometry (it does not depend on the data), so it is
built once and stored as dense [V, V] operators. With V = 180 views this is both exact and
much faster than sparse message passing (no per-edge [E, C, L] message tensor).

Shared by the graph models of the ablation:

- GCN  (transport="identity"):  m_i = sum_j  A_hat[i, j] *                                x_j
- SNN  (transport="shift"):     m_i = sum_j  A_hat[i, j] * S(t_f (theta_i - theta_j))   x_j
- first SNN (transport="so2"):  m_i = sum_j  A_hat[i, j] * R(theta_i - theta_j)          x_j

S(delta) translates a detector line by delta (stalk f has its own trace slope t_f, see
``build_shift_phases``); R is a rotation of every 2D stalk. All use exactly the same topology and
the same normalised weights A_hat.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch


def build_adjacency(
    angles_rad,
    k: int = 12,
    sigma_deg: Optional[float] = 5.0,
    graph_type: str = "knn",
    self_loops: bool = True,
) -> torch.Tensor:
    """Symmetric weighted adjacency W [V, V] of the view graph.

    Args:
        angles_rad: view angles, sorted and (approximately) uniformly spaced.
        k: number of angular neighbours of an interior node: the k/2 nearest views on each
            side (must be even). The graph is a path, NOT a cycle: views near +/-90 degrees
            simply have fewer neighbours, which keeps the graph symmetric and makes the
            reach of one layer exactly k/2 views everywhere.
        sigma_deg: standard deviation (degrees) of the Gaussian edge weights
            w_ij = exp(-(theta_i - theta_j)^2 / (2 sigma^2)). ``None`` gives binary weights.
        graph_type: "knn" (banded, default) or "full" (every pair of views).
        self_loops: add w_ii = 1 (Kipf & Welling style A + I).
    """
    angles = torch.as_tensor(np.asarray(angles_rad), dtype=torch.float64)
    num_views = angles.numel()
    idx = torch.arange(num_views)
    hops = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()

    if graph_type == "knn":
        if k < 2 or k % 2 != 0:
            raise ValueError(f"k must be a positive even number (k/2 neighbours per side), got {k}.")
        connected = (hops <= k // 2) & (hops > 0)
    elif graph_type == "full":
        connected = hops > 0
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}. Use 'knn' or 'full'.")

    if sigma_deg is None:
        weights = torch.ones(num_views, num_views, dtype=torch.float64)
    else:
        delta = angles.unsqueeze(1) - angles.unsqueeze(0)
        sigma_rad = np.deg2rad(sigma_deg)
        weights = torch.exp(-(delta**2) / (2.0 * sigma_rad**2))

    adjacency = weights * connected
    if self_loops:
        adjacency = adjacency + torch.eye(num_views, dtype=torch.float64)
    return adjacency


def normalize_adjacency(adjacency: torch.Tensor, normalization: str = "sym") -> torch.Tensor:
    """Normalise W: "sym" -> D^-1/2 W D^-1/2 (GCN), "rw" -> D^-1 W (weighted mean)."""
    degree = adjacency.sum(dim=1)
    if normalization == "sym":
        inv_sqrt = degree.pow(-0.5)
        return inv_sqrt.unsqueeze(1) * adjacency * inv_sqrt.unsqueeze(0)
    if normalization == "rw":
        return adjacency / degree.unsqueeze(1)
    raise ValueError(f"Unknown normalization: {normalization}. Use 'sym' or 'rw'.")


def build_transport_operators(
    angles_rad,
    transport: str,
    k: int = 12,
    sigma_deg: Optional[float] = 5.0,
    graph_type: str = "knn",
    normalization: str = "sym",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Dense message-passing operators (A_cos, A_sin), each [V, V], float32.

    The SO(2) restriction maps are hard-coded by the acquisition angles. For an edge j -> i
    the transport is the rotation R(theta_i - theta_j) = [[cos, -sin], [sin, cos]], applied
    to every 2D stalk (pair of channels) of the source node. Entry-wise:

        A_cos[i, j] = A_hat[i, j] * cos(theta_i - theta_j)
        A_sin[i, j] = A_hat[i, j] * sin(theta_i - theta_j)

    For transport="identity" (plain GCN) R = I, hence A_cos = A_hat and A_sin is None.
    """
    a_hat = normalize_adjacency(build_adjacency(angles_rad, k, sigma_deg, graph_type), normalization)

    if transport == "identity":
        return a_hat.float(), None
    if transport == "so2":
        angles = torch.as_tensor(np.asarray(angles_rad), dtype=torch.float64)
        delta = angles.unsqueeze(1) - angles.unsqueeze(0)  # delta[i, j] = theta_i - theta_j
        return (a_hat * torch.cos(delta)).float(), (a_hat * torch.sin(delta)).float()
    raise ValueError(f"Unknown transport: {transport}. Use 'identity' or 'so2' (the shift transport works in Fourier space: build_shift_phases).")


def shift_slopes(num_stalks: int, max_depth_mm: float) -> np.ndarray:
    """Trace slopes ds/dtheta (mm / rad) of the shift transport, one per stalk, linearly spaced in
    [-max_depth_mm, max_depth_mm].

    In parallel beam a point of the object at depth t along the rays of view theta (its coordinate
    along the ray direction, measured from the rotation centre) moves along the detector at the rate
    |ds/dtheta| = |t|: its trace in the sinogram is a sinusoid of local slope +/- t. The bank covers every
    depth of the reconstructed square (max_depth_mm = its half-diagonal), so every point has a stalk
    whose slope matches its trace to within half the spacing of the bank.
    """
    return np.linspace(-max_depth_mm, max_depth_mm, num_stalks)


def shift_padding(num_pixels: int, max_shift_px: float) -> int:
    """Samples appended to every detector line before the FFT of the shift transport.

    At least twice the largest translation of a message, so that whatever enters the detector comes from
    the half of the padding next to the edge it enters through; odd total length, so that the real FFT
    has no Nyquist bin and a fractional translation stays exactly invertible.
    """
    padding = 2 * math.ceil(max_shift_px) + 2
    return padding + 1 if (num_pixels + padding) % 2 == 0 else padding


def build_shift_phases(angles_rad, slopes_mm_per_rad, pixel_size_mm: float, padded_length: int) -> torch.Tensor:
    """Fourier phases of the shift transport, [F, V, W, 2] float32 (real, imaginary), W = padded_length // 2 + 1.

        phase[f, v, w] = exp(+i * omega_w * t_f * theta_v / pixel),   omega_w = 2 pi w / padded_length

    For an edge j -> i the restriction maps translate stalk f along the detector by t_f (theta_i - theta_j)
    (in mm, i.e. that divided by the pixel size in pixels): the content of view j is moved to where a point
    whose trace has the slope t_f appears in view i (to first order in theta_i - theta_j, the shift-and-add
    of tomosynthesis). The translation factorises as S(t_f theta_i) S(-t_f theta_j): multiplying the spectrum
    of view j by phase[:, j] and the aggregated spectrum of view i by conj(phase[:, i]) applies it exactly,
    circularly on the padded detector line. Like the SO(2) maps, it is a pure gauge (a flat connection), and
    since a circular translation is orthogonal, R_ji = R_ij^T.
    """
    angles = torch.as_tensor(np.asarray(angles_rad), dtype=torch.float64)
    slopes = torch.as_tensor(np.asarray(slopes_mm_per_rad), dtype=torch.float64)
    omega = 2.0 * np.pi * torch.arange(padded_length // 2 + 1, dtype=torch.float64) / padded_length
    gauge_shift_px = slopes[:, None] * angles[None, :] / pixel_size_mm  # [F, V]
    phase_angle = omega[None, None, :] * gauge_shift_px[:, :, None]     # [F, V, W], in float64: up to ~300 rad
    return torch.stack([torch.cos(phase_angle), torch.sin(phase_angle)], dim=-1).float()


def angular_reach_deg(num_layers: int, k: int, angle_step_deg: float, graph_type: str = "knn") -> float:
    """Angular distance (degrees) over which information can travel through the network.

    Every other operation of the model acts within a single view, so the angular receptive
    field is exactly ``num_layers * k/2`` views. This number is the core of the locality
    argument of the thesis: measured data cannot influence a missing view further away.
    """
    if graph_type == "full":
        return float("inf")
    return num_layers * (k // 2) * angle_step_deg
