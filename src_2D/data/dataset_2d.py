from __future__ import annotations

from typing import Callable, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.geometry.dbt_geometry_2d import DBTGeometry

PhantomType = Literal["shepp_logan", "ellipses", "blobs", "rectangles", "mixed"]


# ----------------------------------------------------------------------
# Utility functions for generating random shapes in PyTorch.
# Every function draws from an explicit ``np.random.Generator`` so that a sample is a pure
# function of its seed and the global numpy / torch RNG states are never touched.
# ----------------------------------------------------------------------
def _random_ellipse_mask(shape: Tuple[int, int], n: int, rng: np.random.Generator) -> torch.Tensor:
    """Binary mask with n random ellipses (filled)."""
    H, W = shape
    mask = torch.zeros(H, W, dtype=torch.float32)
    for _ in range(n):
        cx = rng.uniform(-0.8, 0.8)
        cy = rng.uniform(-0.8, 0.8)
        rx = rng.uniform(0.1, 0.5)
        ry = rng.uniform(0.1, 0.5)
        angle = rng.uniform(0, 180)
        intensity = rng.uniform(0.1, 1.0)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H),
            torch.linspace(-1.0, 1.0, W),
            indexing="ij",
        )
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        x_rot = cos_a * (xx - cx) + sin_a * (yy - cy)
        y_rot = -sin_a * (xx - cx) + cos_a * (yy - cy)
        inside = (x_rot / rx) ** 2 + (y_rot / ry) ** 2 <= 1.0
        mask[inside] += intensity
    return torch.clamp(mask, 0.0, 1.0)


def _random_blobs_mask(shape: Tuple[int, int], n_blobs: int, rng: np.random.Generator) -> torch.Tensor:
    """Binary mask with Gaussian blobs at random positions."""
    H, W = shape
    mask = torch.zeros(H, W, dtype=torch.float32)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H),
        torch.linspace(-1.0, 1.0, W),
        indexing="ij",
    )
    for _ in range(n_blobs):
        cx = rng.uniform(-0.8, 0.8)
        cy = rng.uniform(-0.8, 0.8)
        sigma = rng.uniform(0.05, 0.2)
        intensity = rng.uniform(0.1, 1.0)
        blob = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
        mask += intensity * blob
    return torch.clamp(mask, 0.0, 1.0)


def _random_rectangles_mask(shape: Tuple[int, int], n: int, rng: np.random.Generator) -> torch.Tensor:
    """Binary mask with n random rectangles."""
    H, W = shape
    mask = torch.zeros(H, W, dtype=torch.float32)
    for _ in range(n):
        cx = rng.uniform(-0.8, 0.8)
        cy = rng.uniform(-0.8, 0.8)
        w = rng.uniform(0.1, 0.6)
        h = rng.uniform(0.1, 0.6)
        angle = rng.uniform(0, 180)
        intensity = rng.uniform(0.1, 1.0)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H),
            torch.linspace(-1.0, 1.0, W),
            indexing="ij",
        )
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        x_rot = cos_a * (xx - cx) + sin_a * (yy - cy)
        y_rot = -sin_a * (xx - cx) + cos_a * (yy - cy)
        inside = (x_rot.abs() <= w / 2) & (y_rot.abs() <= h / 2)
        mask[inside] += intensity
    return torch.clamp(mask, 0.0, 1.0)


def _shepp_logan_tensor(shape: Tuple[int, int]) -> torch.Tensor:
    """Shepp-Logan phantom from scikit-image, returned as a torch Tensor."""
    try:
        from skimage.data import shepp_logan_phantom
        img = shepp_logan_phantom()  # shape (400,400) par défaut
        img_t = torch.from_numpy(img.astype(np.float32))
        img_t = img_t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        img_t = F.interpolate(img_t, size=shape, mode='bilinear', align_corners=False)
        return img_t.squeeze(0).squeeze(0)  # (H,W)
    except ImportError:
        # Fallback : un cercle simple
        H, W = shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing="ij"
        )
        return (xx**2 + yy**2 <= 0.5).float()


# ----------------------------------------------------------------------
# Phantom generator class
# ----------------------------------------------------------------------
class PhantomGenerator:
    """Generates 2D phantoms of different types."""
    def __init__(self, device: torch.device):
        self.device = device

    def make_phantom(
        self,
        shape: Tuple[int, int],
        phantom_type: PhantomType = "ellipses",
        randomize_pose: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        rng = rng if rng is not None else np.random.default_rng()

        # 1. Generate base phantom (in normalized [-1,1] coordinates)
        if phantom_type == "mixed":
            phantom_type = ("ellipses", "blobs", "rectangles", "shepp_logan")[rng.integers(4)]

        if phantom_type == "shepp_logan":
            base = _shepp_logan_tensor(shape)
        elif phantom_type == "ellipses":
            base = _random_ellipse_mask(shape, n=int(rng.integers(2, 6)), rng=rng)
        elif phantom_type == "blobs":
            base = _random_blobs_mask(shape, n_blobs=int(rng.integers(3, 8)), rng=rng)
        elif phantom_type == "rectangles":
            base = _random_rectangles_mask(shape, n=int(rng.integers(2, 5)), rng=rng)
        else:
            raise ValueError(f"Unknown phantom_type: {phantom_type}")

        base = base.to(self.device)

        # 2. Apply random rigid transformation
        if randomize_pose:
            rotation_deg = rng.uniform(-180, 180)
            scale = rng.uniform(0.85, 1.0)
            tx = rng.uniform(-0.05, 0.05)
            ty = rng.uniform(-0.05, 0.05)

            rad = np.deg2rad(rotation_deg)
            cos_a, sin_a = np.cos(rad), np.sin(rad)

            theta = torch.eye(2, 3, device=self.device)
            theta[0, 0] = scale * cos_a
            theta[0, 1] = scale * -sin_a
            theta[1, 0] = scale * sin_a
            theta[1, 1] = scale * cos_a
            theta[0, 2] = tx
            theta[1, 2] = ty

            grid = F.affine_grid(
                theta.unsqueeze(0),
                [1, 1, *shape],
                align_corners=False,
            )
            base = base.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            base = F.grid_sample(base, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            base = base.squeeze(0).squeeze(0)

        return torch.clamp(base, 0.0, 1.0)


# =============================================================================
# SinogramCompletionDataset
# =============================================================================
# Each split owns a disjoint seed stream: sample ``i`` of a split is generated from
# ``SeedSequence([SPLIT_STREAMS[split], base_seed, i])``. The validation and test sets are
# therefore immutable, mutually disjoint, and independent of any global RNG state.
SPLIT_STREAMS = {"train": 0, "val": 1, "test": 2}


class SinogramCompletionDataset(Dataset):
    """Pairs of (incomplete, full) sinograms generated on the fly from random 2D phantoms.

    Args:
        n_samples: number of samples of the split.
        split: "train", "val" or "test". "val" and "test" are always deterministic.
        base_seed: changes the whole dataset (all splits) at once; keep the default to
            compare models on identical data.
        fixed_train_set: if True (default) the training split is also a fixed, finite set
            of ``n_samples`` phantoms, so that "trained on n_samples phantoms" is a true
            statement and every model sees the same data. If False, a fresh phantom is
            drawn at every access (infinite-data regime).
        noise_level: incident photon count I0 of the Poisson noise model (<= 0 disables it).
        noisy_target: if False (default) only the measured (incomplete) sinogram is noisy
            and the target is the clean full sinogram, which is the one that satisfies the
            consistency conditions. If True the target carries the same noise realisation.
        is_validation_or_test: deprecated alias of ``split="val"``.

    Returns per item: ``(incomplete [1, V, D], full [1, V, D], phantom [1, H, W])``, the
    sinograms being divided by ``geometry_config.sino_norm``.
    """

    def __init__(
        self,
        n_samples: int,
        phantom_type: PhantomType = "mixed",
        sinogram_shape: Optional[Tuple[int, int]] = None,
        image_shape: Optional[Tuple[int, int]] = None,
        device: str = "cpu",
        projector_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        geometry_config: Optional[DBTGeometryConfig] = None,
        noise_level: float = 100000.0,
        is_validation_or_test: bool = False,
        split: Optional[Literal["train", "val", "test"]] = None,
        base_seed: int = 0,
        fixed_train_set: bool = True,
        noisy_target: bool = False,
    ) -> None:
        if split is None:
            split = "val" if is_validation_or_test else "train"
        if split not in SPLIT_STREAMS:
            raise ValueError(f"Unknown split '{split}'. Use one of {tuple(SPLIT_STREAMS)}.")

        self.geometry_config = geometry_config or DBTGeometryConfig()
        self.n_samples = n_samples
        self.split = split
        self.base_seed = int(base_seed)
        self.fixed_train_set = fixed_train_set
        self.noisy_target = noisy_target
        self.sinogram_shape = sinogram_shape or (
            self.geometry_config.num_views_full,
            self.geometry_config.det_col_count,
        )
        self.image_shape = image_shape or self.geometry_config.image_shape
        self.device = torch.device(device)
        self.projector_fn = projector_fn or self._build_projector()
        self._acquired_view_indices = self._match_acquired_views()
        self.phantom_gen = PhantomGenerator(torch.device("cpu"))
        self.phantom_type: PhantomType = phantom_type
        self.noise_level = noise_level

    @property
    def is_deterministic(self) -> bool:
        return self.split != "train" or self.fixed_train_set

    def _sample_rng(self, index: int) -> np.random.Generator:
        if not self.is_deterministic:
            return np.random.default_rng()
        return np.random.default_rng(np.random.SeedSequence([SPLIT_STREAMS[self.split], self.base_seed, int(index)]))

    def _add_poisson_noise(self, sinogram: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Add Poisson noise to a raw (un-normalised, CPU) sinogram.

        ``self.noise_level`` is I0, the incident photon count per ray (1e4: noisy, 1e6: low
        noise). Line integrals (mm) are scaled by 0.1 before the exponential so that the
        largest attenuations (~80 mm -> 8) stay physically plausible.
        """
        if self.noise_level <= 0.0:
            return sinogram

        I0 = self.noise_level
        scale_factor = 0.1
        expected_photons = I0 * torch.exp(-sinogram * scale_factor)
        noisy_photons = torch.poisson(expected_photons, generator=generator)
        # Prevent log(0) if any pixel receives 0 photons
        noisy_photons = torch.clamp(noisy_photons, min=1.0)
        return -torch.log(noisy_photons / I0) / scale_factor

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 0 <= index < self.n_samples:
            raise IndexError(index)

        rng = self._sample_rng(index)
        phantom = self.phantom_gen.make_phantom(self.image_shape, self.phantom_type, rng=rng)

        clean_sinogram = self.projector_fn(phantom).cpu()
        noise_generator = torch.Generator(device="cpu")
        noise_generator.manual_seed(int(rng.integers(0, 2**62)))
        noisy_sinogram = self._add_poisson_noise(clean_sinogram, noise_generator)

        norm = self.geometry_config.sino_norm
        full_sinogram = (noisy_sinogram if self.noisy_target else clean_sinogram) / norm
        incomplete_sinogram = self._crop_to_acquired_views(noisy_sinogram / norm)

        return (
            incomplete_sinogram.unsqueeze(0).to(self.device),
            full_sinogram.unsqueeze(0).to(self.device),
            phantom.unsqueeze(0).to(self.device),
        )

    # ------------------------------------------------------------------
    # ASTRA projector
    # ------------------------------------------------------------------
    def _build_projector(self) -> Callable[[torch.Tensor], torch.Tensor]:
        # No silent fallback: a missing / broken ASTRA install must stop the run instead
        # of training every model on a fake projector.
        import astra

        geometry = DBTGeometry.from_config(self.geometry_config)
        proj_geom = geometry.get_astra_proj_geom()

        rows, cols = self.image_shape
        row_min, row_max = self.geometry_config.image_extent_mm[0]
        col_min, col_max = self.geometry_config.image_extent_mm[1]
        vol_geom = astra.create_vol_geom(rows, cols, col_min, col_max, row_min, row_max)
        projector_id = astra.create_projector("cuda", proj_geom, vol_geom)

        def projector(phantom: torch.Tensor) -> torch.Tensor:
            image_array = phantom.detach().cpu().numpy().astype(np.float32)
            if image_array.ndim == 3 and image_array.shape[0] == 1:
                image_array = image_array[0]
            image_array = np.ascontiguousarray(image_array)
            if image_array.shape != (rows, cols):
                raise ValueError(f"Expected a phantom of shape {(rows, cols)}, got {image_array.shape}")

            vol_id = astra.data2d.create("-vol", vol_geom, image_array)
            sino_id = astra.data2d.create("-sino", proj_geom)
            cfg = astra.astra_dict("FP_CUDA")
            cfg["ProjectorId"] = projector_id
            cfg["VolumeDataId"] = vol_id
            cfg["ProjectionDataId"] = sino_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            sino_array = astra.data2d.get(sino_id)
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(vol_id)
            astra.data2d.delete(sino_id)

            return torch.from_numpy(np.asarray(sino_array, dtype=np.float32)).float()

        return projector

    def _match_acquired_views(self) -> torch.Tensor:
        """Indices of the full-range views whose angle falls inside the acquired window."""
        return torch.from_numpy(np.flatnonzero(self.geometry_config.acquired_view_mask)).long()

    def _crop_to_acquired_views(self, full_sinogram: torch.Tensor) -> torch.Tensor:
        """Zero out every view outside the acquired +/-angle window."""
        incomplete = torch.zeros_like(full_sinogram)
        indices = self._acquired_view_indices.to(full_sinogram.device)
        incomplete[indices] = full_sinogram[indices]
        return incomplete
