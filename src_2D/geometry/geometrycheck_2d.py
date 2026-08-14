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
    from src_2D.geometry.dbt_geometry_2d import DBTGeometry
except ImportError:
    from dbt_geometry_2d import DBTGeometry

# Setup parameters

num_views = 25
angles = np.linspace(-25 * np.pi / 180, 25 * np.pi / 180, num_views)
src_radius = 590.0
det_radius = 60.0
det_col_count = 128
det_pixel_size = 2.5

# Build the same stationary-detector geometry used by the training pipeline.
dbt_geometry = DBTGeometry(
    angles=angles,
    src_radius=src_radius,
    det_radius=det_radius,
    det_col_count=det_col_count,
    det_pixel_size=det_pixel_size,
)
proj_geom = dbt_geometry.get_astra_proj_geom()
vol_geom = astra.create_vol_geom(32, 128, -120, 120, 0, 60)

# 2 Create a single point

# Empty image (ASTRA expects data in row, col order Z, X)
vol_data = np.zeros((32, 128), dtype=np.float32)

# Let's place a point at physical coordinates: X = 40mm, Z = 30mm
# Convert physical mm to pixel indices:
# X ranges from -120 to 120 over 128 pixels. So 40mm is at index ~85
# Z ranges from 0 to 60 over 32 pixels. So 30mm is at index 16

x_pt, z_pt = 40.3125, 30.9375
x_idx, z_idx = 85, 16

# Place the point using Z, X indexing
vol_data[z_idx, x_idx] = 1.0  # A single bright point

# 3 ASTRA forward projection

projector_id = astra.create_projector('cuda', proj_geom, vol_geom)
vol_id = astra.data2d.create('-vol', vol_geom, vol_data)
sino_id = astra.data2d.create('-sino', proj_geom)

cfg = astra.astra_dict('FP_CUDA')
cfg['ProjectorId'] = projector_id
cfg['VolumeDataId'] = vol_id
cfg['ProjectionDataId'] = sino_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)

sino_arr = astra.data2d.get(sino_id)
astra.algorithm.delete(alg_id)

# 4 Maths Computation of Expected Detector Hits

# We calculate exactly where the ray passing through (40, 30) should hit the detector at Z = -60
expected_u_pixels = []

for theta in angles:
    # 1. Source position
    S_x = src_radius * np.sin(theta)
    S_z = src_radius * np.cos(theta)
    
    # 2. Vector from Source to our Point P(40, 30)
    V_x = x_pt - S_x
    V_z = z_pt - S_z
    
    # 3. Find intersection with the detector plane (Z = -60)
    # S_z + t * V_z = -det_radius  =>  t = (-det_radius - S_z) / V_z
    t = (-det_radius - S_z) / V_z
    
    # 4. Calculate X coordinate on the detector
    hit_x_mm = S_x + t * V_x
    
    # 5. Convert mm to detector pixel index (Detector width is 128*2.5 = 320mm, from -160 to +160)
    hit_pixel = hit_x_mm / det_pixel_size + (det_col_count - 1) / 2.0
    expected_u_pixels.append(hit_pixel)



# 5 Visualization of Results


plt.figure(figsize=(12, 6))

# Plot ASTRA's 2D sinogram directly (shape: views x cols)
plt.imshow(sino_arr.T, cmap='bone', origin='lower', aspect='auto')

# Overlay our mathematical calculation as a red dotted line
plt.plot(np.arange(num_views), expected_u_pixels, 'r--', linewidth=2, label='Mathematical Ray-Tracing')

plt.title('ASTRA Simulation vs Math (Stationary Detector, 2D)')
plt.xlabel('View Index (Angle)')
plt.ylabel('Detector X Pixel Index')
plt.legend()
plt.tight_layout()

plt.savefig('geometry_check_result.png', dpi=300)
print("Plot saved successfully as geometry_check_result.png")

# Cleanup
astra.data2d.delete(sino_id)
astra.data2d.delete(vol_id)
astra.projector.delete(projector_id)