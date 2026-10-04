from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src_2D.models.SinoSheavesNN.graph_data import build_shift_phases, build_transport_operators, shift_padding, shift_slopes

# Feature layout used throughout the graph models: [B, C, V, L]
#   B batch, C = 2 * num_stalks channels, V views (graph nodes), L detector pixels.
# Operations "within a view" are Conv2d with (1, n) kernels: they never mix views. (Without
# cuDNN on the K80s, this layout is ~5x faster than a Conv1d over a [B*V, C, L] batch.)


class ViewNorm(nn.Module):
    """LayerNorm of each view separately, over its (channels, detector) entries.

    Equivalent to GroupNorm(1, C) applied to every node: statistics are never shared between
    views (so the normalisation does not leak information along the angular axis), and the
    affine bias keeps constant (missing) views from being zeroed out.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 3), keepdim=True)
        var = x.var(dim=(1, 3), keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


def detector_conv(in_channels: int, out_channels: int) -> nn.Conv2d:
    """3-tap convolution along the detector axis only. Replicate padding: zero padding creates
    spurious edges on constant rows (acquired flag, missing views) that normalisation amplifies."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), padding_mode="replicate")


class ViewTransport(nn.Module):
    """
    Fixed (non-learned) angular aggregation of the view-graph models, shared by all their layers.

    Features [B, 2F, V, L] hold F stalks (consecutive channel pairs (2f, 2f+1)) of the L detector
    pixels of each of the V views:

        m_i = sum_j A_hat[i, j] * T_ij x_j

    - "identity" (GCN):  T_ij = I, a standard normalised neighbourhood aggregation.
    - "shift"    (SNN):  T_ij translates stalk f along the detector by t_f (theta_i - theta_j), t_f being
      the trace slope assigned to that stalk (``graph_data.shift_slopes``): neighbouring views are aligned
      on the points of depth t_f before being averaged (``graph_data.build_shift_phases``).
    - "so2" (first SNN): T_ij = R(theta_i - theta_j), the same rotation of every 2D stalk at every
      detector pixel. Gauge-equivalent to the GCN aggregation (docs/harmonic_sheaves.md, section 6);
      kept for the record, no model of the protocol uses it any more.

    Nothing is learned and every buffer is non-persistent: a pure function of the configuration stored
    in the checkpoint. Only A_hat mixes views, so the angular reach is the one of the graph.
    """

    def __init__(self, angles_rad, transport: str, num_stalks: int, k: int = 12, sigma_deg: Optional[float] = 5.0,
                 graph_type: str = "knn", normalization: str = "sym", num_pixels: Optional[int] = None,
                 pixel_size_mm: Optional[float] = None, max_depth_mm: Optional[float] = None):
        super().__init__()
        self.transport = transport
        self.num_stalks = num_stalks
        self.num_views = len(angles_rad)
        if transport in ("identity", "so2"):
            a_cos, a_sin = build_transport_operators(angles_rad, transport, k=k, sigma_deg=sigma_deg,
                                                     graph_type=graph_type, normalization=normalization)
            self.register_buffer("a_hat" if transport == "identity" else "a_cos", a_cos, persistent=False)
            if a_sin is not None:
                self.register_buffer("a_sin", a_sin, persistent=False)
        elif transport == "shift":
            if graph_type != "knn":
                raise ValueError("The shift transport is a first-order (small angle) alignment: it needs the local knn graph.")
            if num_pixels is None or pixel_size_mm is None or max_depth_mm is None:
                raise ValueError("The shift transport needs num_pixels, pixel_size_mm and max_depth_mm.")
            a_hat, _ = build_transport_operators(angles_rad, "identity", k=k, sigma_deg=sigma_deg,
                                                 graph_type=graph_type, normalization=normalization)
            angles = torch.as_tensor(np.asarray(angles_rad), dtype=torch.float64)
            max_edge_rad = ((angles[:, None] - angles[None, :]).abs() * (a_hat > 0)).max().item()
            self.slopes_mm_per_rad = shift_slopes(num_stalks, max_depth_mm)
            self.max_shift_px = max_depth_mm * max_edge_rad / pixel_size_mm
            self.num_pixels = num_pixels
            self.padded_length = num_pixels + shift_padding(num_pixels, self.max_shift_px)
            padding = self.padded_length - num_pixels
            self.register_buffer("a_hat", a_hat, persistent=False)
            self.register_buffer("phase", build_shift_phases(angles_rad, self.slopes_mm_per_rad, pixel_size_mm,
                                                             self.padded_length), persistent=False)
            ramp = torch.arange(1, padding + 1, dtype=torch.float32) / (padding + 1)
            self.register_buffer("ramp", ramp, persistent=False)
        else:
            raise ValueError(f"Unknown transport: {transport}. Use 'identity', 'shift' or 'so2'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transport == "identity":
            return torch.einsum("ij,bcjl->bcil", self.a_hat, x)
        if self.transport == "so2":
            return self._rotate_and_aggregate(x)
        return self._shift_and_aggregate(x)

    def _rotate_and_aggregate(self, x: torch.Tensor) -> torch.Tensor:
        B, C, V, L = x.shape
        stalks = x.reshape(B, self.num_stalks, 2, V, L)
        x_a, x_b = stalks[:, :, 0], stalks[:, :, 1]  # the two components of every stalk
        # [m_a]   [cos  -sin] [x_a]
        # [m_b] = [sin   cos] [x_b]   summed over the neighbours j with weights A_hat[i, j]
        m_a = torch.einsum("ij,bfjl->bfil", self.a_cos, x_a) - torch.einsum("ij,bfjl->bfil", self.a_sin, x_b)
        m_b = torch.einsum("ij,bfjl->bfil", self.a_sin, x_a) + torch.einsum("ij,bfjl->bfil", self.a_cos, x_b)
        return torch.stack([m_a, m_b], dim=2).reshape(B, C, V, L)

    def _shift_and_aggregate(self, x: torch.Tensor) -> torch.Tensor:
        B, C, V, L = x.shape
        if L != self.num_pixels:
            raise ValueError(f"The shift transport was built for {self.num_pixels} detector pixels, got {L}.")
        # Periodic extension without a jump: the padding ramps linearly from the last pixel back to the first,
        # so that a circular translation brings in values close to the edge it crosses (like replicate padding)
        # and the FFT does not ring on a wrap-around discontinuity.
        ramp = x[..., -1:] + (x[..., :1] - x[..., -1:]) * self.ramp
        spectrum = torch.fft.rfft(torch.cat([x, ramp], dim=-1), dim=-1)            # [B, C, V, W]
        phase = torch.view_as_complex(self.phase)                                   # [F, V, W]
        W = spectrum.size(-1)
        # Gauge of view j: S(-t_f theta_j) on every channel of stalk f ...
        gauged = (spectrum.reshape(B, self.num_stalks, 2, V, W) * phase[None, :, None]).reshape(B, C, V, W)
        # ... the plain GCN aggregation (A_hat is real: it acts on the real and imaginary parts alike) ...
        mixed = torch.einsum("ij,bcjwr->bciwr", self.a_hat, torch.view_as_real(gauged)).contiguous()
        # ... and back to the frame of view i: S(t_f theta_i). Net effect on an edge j -> i: S(t_f (theta_i - theta_j)).
        mixed = torch.view_as_complex(mixed).reshape(B, self.num_stalks, 2, V, W) * phase.conj()[None, :, None]
        return torch.fft.irfft(mixed.reshape(B, C, V, W), n=self.padded_length, dim=-1)[..., :L]


class ViewGraphConv(nn.Module):
    """
    Angular message passing between views, shared by the GCN baseline and the SNN.

        m_i   = sum_j A_hat[i, j] * T_ij x_j          (aggregation, fixed: ``ViewTransport``)
        out_i = Conv1x1([x_i ; m_i])                  (update, learned)

    The variants have exactly the same parameters; only the fixed transport T_ij differs.
    """

    def __init__(self, num_stalks: int):
        super().__init__()
        self.num_stalks = num_stalks
        self.conv_update = nn.Conv2d(num_stalks * 4, num_stalks * 2, kernel_size=1)

    def forward(self, x: torch.Tensor, transport: ViewTransport) -> torch.Tensor:
        return self.conv_update(torch.cat([x, transport(x)], dim=1))


class ViewGraphBlock(nn.Module):
    """
    Residual block alternating detector-axis and angular processing:
    1. detector conv + ViewNorm + GELU          (within each view)
    2. ViewGraphConv + ViewNorm + GELU          (between views)
    3. detector conv, residual connection       (within each view)

    Steps 1 and 3 never mix views, so the angular reach of a block is exactly the one of
    its single ViewGraphConv.
    """

    def __init__(self, num_stalks: int):
        super().__init__()
        channels = num_stalks * 2
        self.conv1 = detector_conv(channels, channels)
        self.norm1 = ViewNorm(channels)
        self.graph_conv = ViewGraphConv(num_stalks)
        self.norm2 = ViewNorm(channels)
        self.conv2 = detector_conv(channels, channels)

    def forward(self, x: torch.Tensor, transport: ViewTransport) -> torch.Tensor:
        res = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.graph_conv(x, transport)
        x = F.gelu(self.norm2(x))
        x = self.conv2(x)
        return x + res
