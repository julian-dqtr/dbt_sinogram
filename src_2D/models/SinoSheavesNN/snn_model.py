from typing import Optional, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.geometry.dbt_geometry_2d import DBTGeometry
from src_2D.models.SinoSheavesNN.graph_data import angular_reach_deg
from src_2D.models.SinoSheavesNN.snn_layers import ViewGraphBlock, ViewNorm, ViewTransport, detector_conv
from src_2D.utils.evaluation import apply_data_consistency, get_data_consistency_mask


class ViewGraphNet(nn.Module):
    """
    Graph network for sinogram completion. Nodes are the views, node features keep the
    detector axis: [B, 2 * num_stalks, V, D].

    The graph (topology, edge weights, restriction maps) and the data-consistency step
    live INSIDE the model, so training, validation and evaluation cannot diverge:

        completed = model(incomplete)          # [B, 1, V, D] -> [B, 1, V, D]

    Two variants share this exact architecture and parameter count:

    - ``SinoGCN``      (transport="identity"): standard GCN aggregation.
    - ``SinoSheafNet`` (transport="shift"): hard-coded restriction maps that translate every stalk
      along the detector by t_f (theta_i - theta_j), aligning neighbouring views on the points of
      depth t_f (one depth per stalk, spanning the object). See ``ViewTransport``.

    Args:
        transport: "identity", "shift" or "so2" (the first SNN, kept for the record).
        num_stalks: number F of stalks (2F feature channels; with the shift transport, F depths).
        num_layers: depth. The angular reach is num_layers * k/2 views (see
            ``angular_reach_deg``): with 1 degree steps and k = 12, the 6 / 12 / 18-layer
            models reach 36 / 72 / 108 degrees, while the farthest missing view is 65
            degrees away from the acquired window.
        k, sigma_deg, graph_type, normalization: see ``graph_data.build_adjacency``.
        data_consistency: copy the measured views back into the output (hard data consistency).
        blend_width_deg: > 0 restores the soft taper of the first runs (see ``get_data_consistency_mask``).
        grad_checkpoint: trade compute for memory in deep / wide models.
    """

    # Buffers registered in __init__, declared here so static checkers know their type.
    dc_mask: torch.Tensor
    acquired_flag: torch.Tensor

    def __init__(
        self,
        transport: str = "shift",
        num_stalks: int = 32,
        num_layers: int = 6,
        k: int = 12,
        sigma_deg: Optional[float] = 5.0,
        graph_type: str = "knn",
        normalization: str = "sym",
        data_consistency: bool = True,
        blend_width_deg: float = 0.0,
        grad_checkpoint: bool = False,
        geometry_config: Optional[DBTGeometryConfig] = None,
    ):
        super().__init__()
        self.geometry_config = geometry_config or DBTGeometryConfig()
        geom = DBTGeometry.from_config(self.geometry_config)

        self.transport = transport
        self.num_layers = num_layers
        self.k = k
        self.graph_type = graph_type
        self.data_consistency = data_consistency
        self.grad_checkpoint = grad_checkpoint

        # Fixed (non-learned) operators. Not persistent: they are a pure function of the
        # configuration stored in the checkpoint. The shift transport spans every depth of the
        # reconstructed square: up to its half-diagonal from the rotation centre.
        (y_min, y_max), (x_min, x_max) = self.geometry_config.image_extent_mm
        self.transport_op = ViewTransport(
            geom.angles, transport, num_stalks, k=k, sigma_deg=sigma_deg, graph_type=graph_type,
            normalization=normalization, num_pixels=self.geometry_config.det_col_count,
            pixel_size_mm=self.geometry_config.det_pixel_size_mm,
            max_depth_mm=float(np.hypot((y_max - y_min) / 2, (x_max - x_min) / 2)),
        )
        self.register_buffer(
            "dc_mask", get_data_consistency_mask(geom, torch.device("cpu"), blend_width_deg), persistent=False
        )
        acquired = torch.from_numpy(geom.acquired_view_mask).float().view(1, 1, -1, 1)
        self.register_buffer("acquired_flag", acquired, persistent=False)

        # Lift 2 channels (signal + acquired flag) to num_stalks 2D stalks
        self.encoder = detector_conv(2, num_stalks * 2)
        self.blocks = nn.ModuleList([ViewGraphBlock(num_stalks) for _ in range(num_layers)])
        hidden = max(1, num_stalks // 2)
        self.decoder = nn.Sequential(
            detector_conv(num_stalks * 2, num_stalks),
            ViewNorm(num_stalks),
            nn.GELU(),
            detector_conv(num_stalks, hidden),
            nn.GELU(),
            detector_conv(hidden, 1),
        )

    @property
    def angular_reach_deg(self) -> float:
        return angular_reach_deg(self.num_layers, self.k, self.geometry_config.angle_step_deg, self.graph_type)

    def forward(self, incomplete: torch.Tensor, apply_dc: Optional[bool] = None) -> torch.Tensor:
        if incomplete.dim() != 4 or incomplete.size(1) != 1:
            raise ValueError(f"Expected input shape [B, 1, Views, Detectors], got {tuple(incomplete.shape)}")
        B, _, V, D = incomplete.shape
        if V != self.transport_op.num_views:
            raise ValueError(f"The model graph has {self.transport_op.num_views} views, got a sinogram with {V}.")

        # Node features: measured signal + "this view was acquired" flag (from the geometry)
        x = torch.cat([incomplete, self.acquired_flag.expand(B, 1, V, D)], dim=1)  # [B, 2, V, D]
        x = F.gelu(self.encoder(x))                                                # [B, 2F, V, D]

        for block in self.blocks:
            if self.grad_checkpoint and self.training and x.requires_grad:
                x = cast(torch.Tensor, checkpoint(block, x, self.transport_op, use_reentrant=False))
            else:
                x = block(x, self.transport_op)

        out = self.decoder(x)                                                      # [B, 1, V, D]

        use_dc = self.data_consistency if apply_dc is None else apply_dc
        if use_dc:
            out = apply_data_consistency(incomplete, out, self.dc_mask)
        return out


class SinoSheafNet(ViewGraphNet):
    """SinoSheavesNN: view-graph network whose hard-coded restriction maps translate every stalk along the
    detector by t_f (theta_i - theta_j), one depth t_f per stalk (shift-and-add alignment of the views)."""

    def __init__(self, **kwargs):
        kwargs.pop("transport", None)
        super().__init__(transport="shift", **kwargs)


class SinoGCN(ViewGraphNet):
    """Baseline GCN: same graph, same layers, same parameters, but an identity transport (T = I)."""

    def __init__(self, **kwargs):
        kwargs.pop("transport", None)
        super().__init__(transport="identity", **kwargs)
