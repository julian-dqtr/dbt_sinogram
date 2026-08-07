from __future__ import annotations

from typing import Callable, Optional, Tuple, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from geometry.DBT_Geometry import DBTGeometry
from conf.geometry import DBTGeometryConfig


# ----------------------------------------------------------------------
# Utility functions for generating random 3D shapes in PyTorch
# ----------------------------------------------------------------------
def _random_ellipsoid_mask(shape: Tuple[int, int, int], n: int = 3) -> torch.Tensor:
    """Binary mask with n random 3D ellipsoids (filled)."""
    D, H, W = shape
    mask = torch.zeros(D, H, W, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, D),
        torch.linspace(-1.0, 1.0, H),
        torch.linspace(-1.0, 1.0, W),
        indexing="ij",
    )
    for _ in range(n):
        cz = np.random.uniform(-0.8, 0.8)
        cy = np.random.uniform(-0.8, 0.8)
        cx = np.random.uniform(-0.8, 0.8)
        rz = np.random.uniform(0.1, 0.5)
        ry = np.random.uniform(0.1, 0.5)
        rx = np.random.uniform(0.1, 0.5)
        intensity = np.random.uniform(-1.0, 1.0)
        
        # Simple axis-aligned for 3D to keep it fast
        inside = ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        mask[inside] += intensity
    return torch.clamp(mask, -1.0, 1.0)


def _random_blobs_mask_3d(shape: Tuple[int, int, int], n_blobs: int = 5) -> torch.Tensor:
    """Binary mask with Gaussian blobs at random positions."""
    D, H, W = shape
    mask = torch.zeros(D, H, W, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, D),
        torch.linspace(-1.0, 1.0, H),
        torch.linspace(-1.0, 1.0, W),
        indexing="ij",
    )
    for _ in range(n_blobs):
        cz = np.random.uniform(-0.8, 0.8)
        cy = np.random.uniform(-0.8, 0.8)
        cx = np.random.uniform(-0.8, 0.8)
        sigma = np.random.uniform(0.05, 0.2)
        intensity = np.random.uniform(-1.0, 1.0)
        blob = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) / (2 * sigma**2))
        mask += intensity * blob
    return torch.clamp(mask, -1.0, 1.0)


def _random_cuboids_mask(shape: Tuple[int, int, int], n: int = 4) -> torch.Tensor:
    """Binary mask with n random 3D cuboids."""
    D, H, W = shape
    mask = torch.zeros(D, H, W, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, D),
        torch.linspace(-1.0, 1.0, H),
        torch.linspace(-1.0, 1.0, W),
        indexing="ij",
    )
    for _ in range(n):
        cz = np.random.uniform(-0.8, 0.8)
        cy = np.random.uniform(-0.8, 0.8)
        cx = np.random.uniform(-0.8, 0.8)
        dz = np.random.uniform(0.1, 0.6)
        dy = np.random.uniform(0.1, 0.6)
        dx = np.random.uniform(0.1, 0.6)
        intensity = np.random.uniform(-1.0, 1.0)
        
        inside = (xx - cx).abs() <= dx / 2
        inside &= (yy - cy).abs() <= dy / 2
        inside &= (zz - cz).abs() <= dz / 2
        mask[inside] += intensity
    return torch.clamp(mask, -1.0, 1.0)


def _shepp_logan_tensor_3d(shape: Tuple[int, int, int]) -> torch.Tensor:
    """3D Extruded Shepp-Logan phantom."""
    D, H, W = shape
    try:
        from skimage.data import shepp_logan_phantom
        img = shepp_logan_phantom()  # shape (400,400)
        img_t = torch.from_numpy(img.astype(np.float32))
        # Extrude to 3D
        img_t = img_t.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,H,W)
        img_t = img_t.expand(1, 1, D, -1, -1)
        img_t = F.interpolate(img_t, size=(D, H, W), mode='trilinear', align_corners=False)
        return img_t.squeeze(0).squeeze(0)  # (D,H,W)
    except ImportError:
        # Fallback : un cylindre simple
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, D),
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing="ij"
        )
        return (xx**2 + yy**2 <= 0.5).float()


# ----------------------------------------------------------------------
# Phantom generator class
# ----------------------------------------------------------------------
class PhantomGenerator:
    """Generates 3D phantoms of different types."""
    def __init__(self, device: torch.device):
        self.device = device

    def make_phantom(
        self,
        shape: Tuple[int, int, int],
        phantom_type: Literal["shepp_logan", "ellipses", "blobs", "rectangles", "mixed"] = "ellipses",
        randomize_pose: bool = True,
    ) -> torch.Tensor:
        # 1. Generate base phantom (in normalized [-1,1] coordinates)
        if phantom_type == "shepp_logan":
            base = _shepp_logan_tensor_3d(shape)
        elif phantom_type == "ellipses":
            base = _random_ellipsoid_mask(shape, n=np.random.randint(2, 6))
        elif phantom_type == "blobs":
            base = _random_blobs_mask_3d(shape, n_blobs=np.random.randint(3, 8))
        elif phantom_type == "rectangles":
            base = _random_cuboids_mask(shape, n=np.random.randint(2, 5))
        else:  # mixed
            funcs = [_random_ellipsoid_mask, _random_blobs_mask_3d, _random_cuboids_mask, _shepp_logan_tensor_3d]
            chosen = np.random.choice(funcs)
            if chosen == _shepp_logan_tensor_3d:
                base = _shepp_logan_tensor_3d(shape)
            elif chosen == _random_blobs_mask_3d:
                base = _random_blobs_mask_3d(shape, n_blobs=np.random.randint(3, 8))
            else:
                n = np.random.randint(2, 6)
                base = chosen(shape, n=n)

        base = base.to(self.device)

        # 2. Apply random rigid transformation (simplified for 3D: just translation and scale)
        if randomize_pose:
            scale = np.random.uniform(0.85, 1.0)
            tz = np.random.uniform(-0.05, 0.05)
            ty = np.random.uniform(-0.05, 0.05)
            tx = np.random.uniform(-0.05, 0.05)

            theta = torch.zeros(3, 4, device=self.device)
            theta[0, 0] = scale
            theta[1, 1] = scale
            theta[2, 2] = scale
            theta[0, 3] = tz
            theta[1, 3] = ty
            theta[2, 3] = tx

            grid = F.affine_grid(
                theta.unsqueeze(0),
                torch.Size([1, 1, *shape]),
                align_corners=False,
            )
            base = base.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
            base = F.grid_sample(base, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            base = base.squeeze(0).squeeze(0)

        return torch.clamp(base, -1.0, 1.0)


# =============================================================================
# SinogramCompletionDataset
# =============================================================================
class SinogramCompletionDataset(Dataset):
    """Pairs of incomplete/full 3D sinograms with varied 3D phantoms."""

    def __init__(
        self,
        n_samples: int,
        phantom_type: Literal["shepp_logan", "ellipses", "blobs", "rectangles", "mixed"] = "mixed",
        sinogram_shape: Optional[Tuple[int, int, int]] = None,
        image_shape: Optional[Tuple[int, int, int]] = None,
        device: str = "cpu",
        projector_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        geometry_config: Optional[DBTGeometryConfig] = None,
        noise_level: float = 0.0,
    ) -> None:
        self.geometry_config = geometry_config or DBTGeometryConfig()
        self.n_samples = n_samples
        self.sinogram_shape = sinogram_shape or (
            self.geometry_config.num_views_full,
            self.geometry_config.det_row_count,
            self.geometry_config.det_col_count,
        )
        self.image_shape = image_shape or self.geometry_config.image_shape
        self.device = torch.device(device)
        self.projector_fn = projector_fn or self._build_projector()
        self._acquired_view_indices = self._match_acquired_views()
        self.phantom_gen = PhantomGenerator(self.device)
        self.phantom_type = phantom_type
        self.noise_level = noise_level

    def _add_poisson_noise(self, sinogram: torch.Tensor) -> torch.Tensor:
        if self.noise_level <= 0.0:
            return sinogram
        return sinogram

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phantom = self.phantom_gen.make_phantom(self.image_shape, self.phantom_type)
        full_sinogram = self.projector_fn(phantom).to(self.device)
        full_sinogram = self._add_poisson_noise(full_sinogram)
        incomplete_sinogram = self._crop_to_acquired_views(full_sinogram)

        return (
            incomplete_sinogram.unsqueeze(0),
            full_sinogram.unsqueeze(0),
            phantom.unsqueeze(0),
        )

    # ------------------------------------------------------------------
    # ASTRA 3D projector
    # ------------------------------------------------------------------
    def _build_projector(self) -> Callable[[torch.Tensor], torch.Tensor]:
        try:
            import astra
        except ImportError:
            return self._default_projector

        try:
            geometry = DBTGeometry(
                angles=self.geometry_config.full_angles,
                src_radius=self.geometry_config.src_radius_mm,
                det_radius=self.geometry_config.det_radius_mm,
                det_row_count=self.geometry_config.det_row_count,
                det_col_count=self.geometry_config.det_col_count,
                det_pixel_size=self.geometry_config.det_pixel_size_mm,
            )
            proj_geom = geometry.get_astra_proj_geom()

            depth, rows, cols = self.image_shape
            z_min, z_max = self.geometry_config.image_extent_mm[0]
            y_min, y_max = self.geometry_config.image_extent_mm[1]
            x_min, x_max = self.geometry_config.image_extent_mm[2]
            
            # ASTRA create_vol_geom 3D: (Y, X, Z, minX, maxX, minY, maxY, minZ, maxZ)
            # Y corresponds to rows, X to cols, Z to slices.
            vol_geom = astra.create_vol_geom(rows, cols, depth, x_min, x_max, y_min, y_max, z_min, z_max)
            projector_id = astra.create_projector("cuda3d", proj_geom, vol_geom)

            def projector(phantom: torch.Tensor) -> torch.Tensor:
                image_array = phantom.detach().cpu().numpy().astype(np.float32)
                if image_array.ndim == 4 and image_array.shape[0] == 1:
                    image_array = image_array[0]
                image_array = np.asarray(image_array)
                # Ensure shape is (Z, Y, X) for ASTRA 3D
                if image_array.shape != (depth, rows, cols):
                    raise ValueError(f"Expected shape {(depth, rows, cols)}, got {image_array.shape}")

                vol_id = astra.data3d.create("-vol", vol_geom, image_array)
                sino_id = astra.data3d.create("-sino", proj_geom)
                cfg = astra.astra_dict("FP3D_CUDA")
                cfg["ProjectorId"] = projector_id
                cfg["VolumeDataId"] = vol_id
                cfg["ProjectionDataId"] = sino_id
                alg_id = astra.algorithm.create(cfg)
                astra.algorithm.run(alg_id)
                sino_array = astra.data3d.get(sino_id)
                astra.algorithm.delete(alg_id)
                astra.data3d.delete(vol_id)
                astra.data3d.delete(sino_id)

                # sino_array is (det_row_count, num_views, det_col_count)
                # we transpose to (num_views, det_row_count, det_col_count)
                sino_array = np.transpose(sino_array, (1, 0, 2))
                return torch.from_numpy(np.asarray(sino_array, dtype=np.float32)).float()

            return projector
        except Exception as e:
            print(f"Warning: Falling back to default projector. ASTRA 3D init failed: {e}")
            return self._default_projector

    def _default_projector(self, phantom: torch.Tensor) -> torch.Tensor:
        """Simple fallback projector."""
        base = phantom.squeeze(0).float()
        reduced = torch.nn.functional.interpolate(
            base.unsqueeze(0).unsqueeze(0),
            size=(1, self.sinogram_shape[1], self.sinogram_shape[2]),
            mode="trilinear",
        ).squeeze(0).squeeze(0).squeeze(0)
        return reduced.unsqueeze(0).repeat(self.sinogram_shape[0], 1, 1).float()

    def _match_acquired_views(self) -> torch.Tensor:
        full_angles_deg = np.rad2deg(self.geometry_config.full_angles)
        in_window = (full_angles_deg >= self.geometry_config.angle_min_deg) & (
            full_angles_deg <= self.geometry_config.angle_max_deg
        )
        return torch.from_numpy(np.flatnonzero(in_window)).long()

    def _crop_to_acquired_views(self, full_sinogram: torch.Tensor) -> torch.Tensor:
        incomplete = torch.zeros_like(full_sinogram)
        indices = self._acquired_view_indices.to(full_sinogram.device)
        incomplete[indices] = full_sinogram[indices]
        return incomplete