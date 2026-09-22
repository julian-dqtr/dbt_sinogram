"""The ASTRA simulation is a rotating parallel-beam Radon transform with the documented convention."""
import numpy as np
import torch

from src_2D.data.dataset_2d import SinogramCompletionDataset
from tests.conftest import analytic_disk_sinogram


def test_config_is_parallel_with_clean_angular_grid(config, geom):
    assert config.beam == "parallel"
    assert geom.get_astra_proj_geom()["type"] == "parallel"
    angles_deg = np.rad2deg(config.full_angles)
    np.testing.assert_allclose(angles_deg, np.arange(-90, 90), atol=1e-9)  # [-90, 90), 1 degree step
    assert config.num_views == 51 and geom.acquired_view_mask.sum() == 51  # -25..25 included
    assert np.isclose(angles_deg[geom.acquired_view_mask].mean(), 0.0)      # window centred on 0


def test_detector_covers_the_whole_object(config):
    half_detector = config.det_col_count * config.det_pixel_size_mm / 2.0
    half_diagonal = np.hypot(*[extent[1] for extent in config.image_extent_mm])
    assert half_detector >= half_diagonal  # no truncated view


def test_astra_projection_matches_analytic_radon_transform(config, geom):
    """Project a rasterised disk with ASTRA and compare with the closed-form Radon transform.
    This pins both the scaling (mm) and the sign / orientation convention of theta and s."""
    radius, center, density = 20.0, (15.0, -10.0), 0.5
    (row_min, row_max), (col_min, col_max) = config.image_extent_mm
    rows, cols = config.image_shape
    # ASTRA volume convention: x grows with the column index, y DEcreases with the row index.
    x = col_min + (np.arange(cols) + 0.5) * (col_max - col_min) / cols
    y = row_max - (np.arange(rows) + 0.5) * (row_max - row_min) / rows
    disk = density * ((x[None, :] - center[0]) ** 2 + (y[:, None] - center[1]) ** 2 <= radius**2)

    dataset = SinogramCompletionDataset(1, split="val", noise_level=0.0, geometry_config=config)
    astra_sino = dataset.projector_fn(torch.from_numpy(disk.astype(np.float32)))
    exact = analytic_disk_sinogram(geom, radius, center, density)

    rel_err = (astra_sino - exact).norm() / exact.norm()
    assert rel_err < 0.03, f"ASTRA vs analytic Radon transform: relative error {rel_err:.3f}"
