from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from typing import Callable, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src_2D.conf.geometry_conf_2d import DBTGeometryConfig
from src_2D.geometry.dbt_geometry_2d import DBTGeometry


# ----------------------------------------------------------------------
# Utility functions for generating random shapes in PyTorch
# ----------------------------------------------------------------------
def _random_ellipse_mask(shape: Tuple[int, int], n: int = 3) -> torch.Tensor:
    """Binary mask with n random ellipses (filled)."""
    H, W = shape
    mask = torch.zeros(H, W, dtype=torch.float32)
    for _ in range(n):
        cx = np.random.uniform(-0.8, 0.8)
        cy = np.random.uniform(-0.8, 0.8)
        rx = np.random.uniform(0.1, 0.5)
        ry = np.random.uniform(0.1, 0.5)
        angle = np.random.uniform(0, 180)
        intensity = np.random.uniform(0.1, 1.0)
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


def _random_blobs_mask(shape: Tuple[int, int], n_blobs: int = 5) -> torch.Tensor:
    """Binary mask with Gaussian blobs at random positions."""
    H, W = shape
    mask = torch.zeros(H, W, dtype=torch.float32)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H),
        torch.linspace(-1.0, 1.0, W),
        indexing="ij",
    )
    for _ in range(n_blobs):
        cx = np.random.uniform(-0.8, 0.8)
        cy = np.random.uniform(-0.8, 0.8)
        sigma = np.random.uniform(0.05, 0.2)
        intensity = np.random.uniform(0.1, 1.0)
        blob = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
        mask += intensity * blob
    return torch.clamp(mask, 0.0, 1.0)


def _random_rectangles_mask(shape: Tuple[int, int], n: int = 4) -> torch.Tensor:
    """Binary mask with n random rectangles."""
    H, W = shape
    mask = torch.zeros(H, W, dtype=torch.float32)
    for _ in range(n):
        cx = np.random.uniform(-0.8, 0.8)
        cy = np.random.uniform(-0.8, 0.8)
        w = np.random.uniform(0.1, 0.6)
        h = np.random.uniform(0.1, 0.6)
        angle = np.random.uniform(0, 180)
        intensity = np.random.uniform(0.1, 1.0)
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
        phantom_type: Literal["shepp_logan", "ellipses", "blobs", "rectangles", "mixed"] = "ellipses",
        randomize_pose: bool = True,
    ) -> torch.Tensor:
        # 1. Generate base phantom (in normalized [-1,1] coordinates)
        if phantom_type == "shepp_logan":
            base = _shepp_logan_tensor(shape)
        elif phantom_type == "ellipses":
            base = _random_ellipse_mask(shape, n=np.random.randint(2, 6))
        elif phantom_type == "blobs":
            base = _random_blobs_mask(shape, n_blobs=np.random.randint(3, 8))
        elif phantom_type == "rectangles":
            base = _random_rectangles_mask(shape, n=np.random.randint(2, 5))
        else:  # mixed
            funcs = [_random_ellipse_mask, _random_blobs_mask, _random_rectangles_mask, _shepp_logan_tensor]
            chosen = np.random.choice(funcs)
            if chosen == _shepp_logan_tensor:
                base = _shepp_logan_tensor(shape)
            elif chosen == _random_blobs_mask:
                base = _random_blobs_mask(shape, n_blobs=np.random.randint(3, 8))
            else:
                n = np.random.randint(2, 6)
                base = chosen(shape, n=n)

        base = base.to(self.device)

        # 2. Apply random rigid transformation
        if randomize_pose:
            rotation_deg = np.random.uniform(-180, 180)
            scale = np.random.uniform(0.85, 1.0)
            tx = np.random.uniform(-0.05, 0.05)
            ty = np.random.uniform(-0.05, 0.05)

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
                torch.Size([1, 1, *shape]),
                align_corners=False,
            )
            base = base.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            base = F.grid_sample(base, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            base = base.squeeze(0).squeeze(0)

        return torch.clamp(base, 0.0, 1.0)


# =============================================================================
# SinogramCompletionDataset – version finale
# =============================================================================
class SinogramCompletionDataset(Dataset):
    """Pairs of incomplete/full sinograms with varied 2D phantoms."""

    def __init__(
        self,
        n_samples: int,
        phantom_type: Literal["shepp_logan", "ellipses", "blobs", "rectangles", "mixed"] = "mixed",
        sinogram_shape: Optional[Tuple[int, int]] = None,
        image_shape: Optional[Tuple[int, int]] = None,
        device: str = "cpu",
        projector_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        geometry_config: Optional[DBTGeometryConfig] = None,
        noise_level: float = 0.0,
        is_validation_or_test: bool = False,
    ) -> None:
        self.geometry_config = geometry_config or DBTGeometryConfig()
        self.n_samples = n_samples
        self.sinogram_shape = sinogram_shape or (
            self.geometry_config.num_views_full,
            self.geometry_config.det_col_count,
        )
        self.image_shape = image_shape or self.geometry_config.image_shape
        self.device = torch.device(device)
        self.projector_fn = projector_fn or self._build_projector()
        self._acquired_view_indices = self._match_acquired_views()
        self.phantom_gen = PhantomGenerator(self.device)
        self.phantom_type = phantom_type
        self.noise_level = noise_level
        self.is_validation_or_test = is_validation_or_test

    def _add_poisson_noise(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Add Poisson noise to the sinogram. A placeholder for future implementation."""
        if self.noise_level <= 0.0:
            return sinogram
        
        # Placeholder: When you're ready, implement Poisson noise here using self.noise_level.
        # e.g., noisy = torch.poisson(sinogram * lambda) / lambda
        return sinogram

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_validation_or_test:
            np.random.seed(index)
            torch.manual_seed(index)

        phantom = self.phantom_gen.make_phantom(self.image_shape, self.phantom_type)

        if self.is_validation_or_test:
            # Reset seeds so we don't break global randomness for other tasks
            np.random.seed()
            torch.seed()

        full_sinogram = self.projector_fn(phantom).to(self.device)
        full_sinogram = self._add_poisson_noise(full_sinogram)
        
        # Global normalization to approximately [-1, 1] based on theoretical maximum
        # For a 128x128 image with max density 1.0, max projection is ~128
        global_sino_norm = 100.0 
        full_sinogram = full_sinogram / global_sino_norm
            
        incomplete_sinogram = self._crop_to_acquired_views(full_sinogram)

        return (
            incomplete_sinogram.unsqueeze(0),
            full_sinogram.unsqueeze(0),
            phantom.unsqueeze(0),
        )

    # ------------------------------------------------------------------
    # ASTRA projector (unchanged from original)
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
                det_col_count=self.geometry_config.det_col_count,
                det_pixel_size=self.geometry_config.det_pixel_size_mm,
            )
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
                image_array = np.asarray(image_array)
                if image_array.shape != (rows, cols):
                    image_array = np.reshape(image_array, (rows, cols))

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
        except Exception:
            return self._default_projector

    def _default_projector(self, phantom: torch.Tensor) -> torch.Tensor:
        """Simple fallback projector used when ASTRA is unavailable."""
        base = phantom.squeeze(0).float()
        reduced = torch.nn.functional.interpolate(
            base.unsqueeze(0).unsqueeze(0),
            size=(1, max(1, self.sinogram_shape[1])),
            mode="bilinear",
        ).squeeze(0).squeeze(0)
        return reduced.reshape(1, -1).repeat(self.sinogram_shape[0], 1).float()

    def _match_acquired_views(self) -> torch.Tensor:
        """Indices of the full-arc views whose angle falls inside the acquired window."""
        full_angles_deg = np.rad2deg(self.geometry_config.full_angles)
        in_window = (full_angles_deg >= self.geometry_config.angle_min_deg) & (
            full_angles_deg <= self.geometry_config.angle_max_deg
        )
        return torch.from_numpy(np.flatnonzero(in_window)).long()

    def _crop_to_acquired_views(self, full_sinogram: torch.Tensor) -> torch.Tensor:
        """Zero out every view outside the acquired +/-angle window."""
        incomplete = torch.zeros_like(full_sinogram)
        indices = self._acquired_view_indices.to(full_sinogram.device)
        incomplete[indices] = full_sinogram[indices]
        return incomplete

from src_2D.data.dataset_2d import SinogramCompletionDataset
from src_2D.models.Unet2D.evaluate_2d import (generate_example_figure,
                                              resolve_compute_device,
                                              save_random_dataset_preview,
                                              save_training_curve)
from src_2D.models.Unet2D.unet_2d import SinogramUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 2D sinogram completion model.")
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0042275432102115195)
    parser.add_argument("--filters", type=int, default=32)
    parser.add_argument("--optimizer", type=str, choices=("Adam", "AdamW"), default="AdamW")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "outputs/2d/checkpoints")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "outputs/2d/visualisation")
    parser.add_argument("--reconstruct-iters", type=int, default=15)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="dbt-sinogram-completion-2D")
    parser.add_argument("--wandb-name", type=str, default=None, help="Name of the wandb run")
    parser.add_argument("--wandb-mode", type=str, choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_compute_device()
    geometry = DBTGeometryConfig()

    train_dataset = SinogramCompletionDataset(n_samples=args.n_samples, phantom_type="mixed", device=str(device))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = SinogramCompletionDataset(n_samples=max(1, args.n_samples // 5), phantom_type="mixed", device=str(device), is_validation_or_test=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    save_random_dataset_preview(train_dataset, geometry, args)

    model = SinogramUNet(in_channels=1, out_channels=1, filters=args.filters).to(device)

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    run = None
    if args.use_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args), mode=args.wandb_mode)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float("inf")
    epochs_without_improvement = 0
    start_time = time.time()
    train_losses = []

    model.train()
    
    epoch_iter = tqdm(range(args.max_epochs), desc="training", unit="epoch")
    try:
        for epoch in epoch_iter:
            running_loss = 0.0
            
            batch_iter = tqdm(train_loader, desc=f"epoch {epoch + 1}", leave=False, unit="batch")
            for incomplete, target, _ in batch_iter:
                incomplete = incomplete.to(device)
                target = target.to(device)

                refined_output = model(incomplete)
                loss = criterion(refined_output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batch_iter.set_postfix(unet=f"{loss.item():.4f}")

            train_loss = running_loss / max(1, len(train_loader))
            
            # Validation loop
            model.eval()
            val_loss = 0.0
            val_psnr = 0.0
            val_ssim = 0.0
            with torch.no_grad():
                for incomplete, target, _ in val_loader:
                    incomplete = incomplete.to(device)
                    target = target.to(device)
                    refined_output = model(incomplete)
                    
                    val_loss += criterion(refined_output, target).item()
                    
                    # Calculate metrics on CPU
                    target_np = target.cpu().numpy()
                    output_np = refined_output.cpu().numpy()
                    
                    batch_psnr = 0.0
                    batch_ssim = 0.0
                    for i in range(target_np.shape[0]):
                        t = target_np[i, 0]
                        o = output_np[i, 0]
                        batch_psnr += psnr(t, o, data_range=2.0)
                        batch_ssim += ssim(t, o, data_range=2.0)
                    
                    val_psnr += batch_psnr / target_np.shape[0]
                    val_ssim += batch_ssim / target_np.shape[0]

            val_loss /= max(1, len(val_loader))
            val_psnr /= max(1, len(val_loader))
            val_ssim /= max(1, len(val_loader))
            model.train()

            train_losses.append(train_loss)
            epoch_iter.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", ssim=f"{val_ssim:.4f}")

            if run is not None:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "loss/train_unet": train_loss,
                    "loss/val_unet": val_loss,
                    "metrics/val_psnr": val_psnr,
                    "metrics/val_ssim": val_ssim,
                    "loss/best_val": min(best_loss, val_loss),
                })

            if val_loss < best_loss - args.min_delta:
                best_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), args.checkpoint_dir / "best_model.pt")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"\nNo improvement for {args.patience} epochs, stopping early at epoch {epoch + 1}.")
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user, saving current model state.")
        torch.save(model.state_dict(), args.checkpoint_dir / "interrupted_model.pt")

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed / 60:.1f} min. Best loss (U-Net): {best_loss:.4f}")
    
    # Reload best model for saving curve/examples (optional, but good for validation example)
    if (args.checkpoint_dir / "best_model.pt").exists():
        model.load_state_dict(torch.load(args.checkpoint_dir / "best_model.pt", map_location=device, weights_only=True))

    save_training_curve(train_losses, args)
    # Generate an example from validation set to see training result
    generate_example_figure(model, val_dataset, device, geometry, args, run)
        
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
