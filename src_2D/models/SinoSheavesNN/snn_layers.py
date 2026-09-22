from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ViewGraphConv(nn.Module):
    """
    Angular message passing between views, shared by the GCN baseline and the SNN.

    Features [B, 2F, V, L] hold F two-dimensional stalks (consecutive channel pairs
    (2f, 2f+1)) at each of the L detector pixels of each of the V views.

        m_i   = sum_j A_hat[i, j] * R_ij x_j          (aggregation)
        out_i = Conv1x1([x_i ; m_i])                  (update)

    - GCN: R_ij = I, i.e. a standard normalised neighbourhood aggregation.
    - SNN: R_ij = R(theta_i - theta_j) in SO(2), hard-coded by the acquisition angles and
      applied to every stalk. Nothing about the transport is learned.

    The two variants have exactly the same parameters; only the fixed operators differ.
    """

    def __init__(self, num_stalks: int):
        super().__init__()
        self.num_stalks = num_stalks
        self.conv_update = nn.Conv2d(num_stalks * 4, num_stalks * 2, kernel_size=1)

    def aggregate(self, x: torch.Tensor, a_cos: torch.Tensor, a_sin: Optional[torch.Tensor]) -> torch.Tensor:
        if a_sin is None:
            # R = I: plain weighted aggregation of the neighbours.
            return torch.einsum("ij,bcjl->bcil", a_cos, x)

        B, C, V, L = x.shape
        stalks = x.reshape(B, self.num_stalks, 2, V, L)
        x_a, x_b = stalks[:, :, 0], stalks[:, :, 1]  # the two components of every stalk
        # [m_a]   [cos  -sin] [x_a]
        # [m_b] = [sin   cos] [x_b]   summed over the neighbours j with weights A_hat[i, j]
        m_a = torch.einsum("ij,bfjl->bfil", a_cos, x_a) - torch.einsum("ij,bfjl->bfil", a_sin, x_b)
        m_b = torch.einsum("ij,bfjl->bfil", a_sin, x_a) + torch.einsum("ij,bfjl->bfil", a_cos, x_b)
        return torch.stack([m_a, m_b], dim=2).reshape(B, C, V, L)

    def forward(self, x: torch.Tensor, a_cos: torch.Tensor, a_sin: Optional[torch.Tensor]) -> torch.Tensor:
        messages = self.aggregate(x, a_cos, a_sin)
        return self.conv_update(torch.cat([x, messages], dim=1))


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

    def forward(self, x: torch.Tensor, a_cos: torch.Tensor, a_sin: Optional[torch.Tensor]) -> torch.Tensor:
        res = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.graph_conv(x, a_cos, a_sin)
        x = F.gelu(self.norm2(x))
        x = self.conv2(x)
        return x + res
