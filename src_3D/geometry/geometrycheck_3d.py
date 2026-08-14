import sys
from pathlib import Path

# Add project root to sys.path if not present
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import astra
import matplotlib.pyplot as plt
import numpy as np

try:
    from src_3D.conf.geometry_conf_3d import DBTGeometryConfig
    from src_3D.geometry.dbt_geometry_3d import DBTGeometry
except ImportError:
    from conf.geometry_conf_3d import DBTGeometryConfig
    from dbt_geometry_3d import DBTGeometry

# 1. Setup parameters from configuration
cfg_geom = DBTGeometryConfig()

angles = cfg_geom.angles
num_views = cfg_geom.num_views
src_radius = cfg_geom.src_radius_mm
det_radius = cfg_geom.det_radius_mm
det_row_count = cfg_geom.det_row_count
det_col_count = cfg_geom.det_col_count
det_pixel_size = cfg_geom.det_pixel_size_mm

# Build 3D cone-beam stationary-detector geometry used by the pipeline
dbt_geometry = DBTGeometry(
    angles=angles,
    src_radius=src_radius,
    det_radius=det_radius,
    det_row_count=det_row_count,
    det_col_count=det_col_count,
    det_pixel_size=det_pixel_size,
)
proj_geom = dbt_geometry.get_astra_proj_geom()

depth, rows, cols = cfg_geom.image_shape
(z_min, z_max), (y_min, y_max), (x_min, x_max) = cfg_geom.image_extent_mm
vol_geom = astra.create_vol_geom(rows, cols, depth, x_min, x_max, y_min, y_max, z_min, z_max)

# 2. Create a single point in 3D volume
# ASTRA 3D volume data is indexed as (depth/slices, rows/Y, cols/X)
vol_data = np.zeros((depth, rows, cols), dtype=np.float32)

# Place a point at physical coordinates: X = 20.0 mm, Y = 10.0 mm, Z = 30.0 mm
px_target, py_target, pz_target = 20.0, 10.0, 30.0
z_idx = int((pz_target - z_min) / (z_max - z_min) * depth)
y_idx = int((py_target - y_min) / (y_max - y_min) * rows)
x_idx = int((px_target - x_min) / (x_max - x_min) * cols)

# Actual center of the voxel
px = x_min + (x_idx + 0.5) * (x_max - x_min) / cols
py = y_min + (y_idx + 0.5) * (y_max - y_min) / rows
pz = z_min + (z_idx + 0.5) * (z_max - z_min) / depth

vol_data[z_idx, y_idx, x_idx] = 1.0

# 3. ASTRA 3D forward projection
projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
vol_id = astra.data3d.create('-vol', vol_geom, vol_data)
sino_id = astra.data3d.create('-sino', proj_geom)

cfg = astra.astra_dict('FP3D_CUDA')
cfg['ProjectorId'] = projector_id
cfg['VolumeDataId'] = vol_id
cfg['ProjectionDataId'] = sino_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)

sino_arr = astra.data3d.get(sino_id)  # Shape: (det_row_count, num_views, det_col_count)

astra.algorithm.delete(alg_id)
astra.data3d.delete(sino_id)
astra.data3d.delete(vol_id)
astra.projector.delete(projector_id)

# 4. Mathematical ray-tracing calculation of expected detector hits
expected_u_pixels = []
expected_v_pixels = []

for theta in angles:
    Sx = src_radius * np.sin(theta)
    Sy = src_radius * np.cos(theta)
    Sz = 0.0

    Vx = px - Sx
    Vy = py - Sy
    Vz = pz - Sz

    # Intersection with stationary detector plane at Y = -det_radius:
    t = (-det_radius - Sy) / Vy
    hit_x = Sx + t * Vx
    hit_z = Sz + t * Vz

    # Convert physical mm to detector pixel index (detector centered at 0)
    hit_u = hit_x / det_pixel_size + (det_col_count - 1) / 2.0
    hit_v = hit_z / det_pixel_size + (det_row_count - 1) / 2.0

    expected_u_pixels.append(hit_u)
    expected_v_pixels.append(hit_v)

# 5. Visualization of results
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Detector U (columns) across views
sino_u_proj = np.max(sino_arr, axis=0)  # (num_views, det_cols)
axes[0].imshow(sino_u_proj.T, cmap='bone', origin='lower', aspect='auto')
axes[0].plot(np.arange(num_views), expected_u_pixels, 'r--', linewidth=2, label='Math Ray-Tracing (U)')
axes[0].set_title('ASTRA Simulation vs Math: Detector U (Col) vs View')
axes[0].set_xlabel('View Index (Angle)')
axes[0].set_ylabel('Detector Col Index (U)')
axes[0].legend()

# Subplot 2: Detector V (rows) across views
sino_v_proj = np.max(sino_arr, axis=2)  # (det_rows, num_views)
axes[1].imshow(sino_v_proj, cmap='bone', origin='lower', aspect='auto')
axes[1].plot(np.arange(num_views), expected_v_pixels, 'r--', linewidth=2, label='Math Ray-Tracing (V)')
axes[1].set_title('ASTRA Simulation vs Math: Detector V (Row) vs View')
axes[1].set_xlabel('View Index (Angle)')
axes[1].set_ylabel('Detector Row Index (V)')
axes[1].legend()

plt.tight_layout()
output_path = Path(__file__).parent / 'geometry_check_3d_result.png'
plt.savefig(output_path, dpi=300)
print(f"Plot saved successfully as {output_path}")