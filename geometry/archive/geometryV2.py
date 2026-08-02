import numpy as np
import astra
from DBT_Geometry import DBTGeometry

# 1. Define geometry parameters
num_views = 25
angles = np.linspace(-25 * np.pi / 180, 25 * np.pi / 180, num_views)
src_radius = 590.0
det_radius = 60.0
det_col_count = 2816  # X pixels
det_row_count = 3584  # Y pixels
det_pixel_size = 0.085

# 2. Initialize the custom geometry wrapper
dbt_geometry = DBTGeometry(
    angles=angles, 
    src_radius=src_radius, 
    det_radius=det_radius,
    det_row_count=det_row_count,
    det_col_count=det_col_count,
    det_pixel_size=det_pixel_size
)

# Extract the ASTRA-compatible projection geometry
proj_geom = dbt_geometry.get_astra_proj_geom()

# 3. Define the reconstruction volume geometry for ASTRA
# Format: (Y_voxels, X_voxels, Z_voxels, x_min, x_max, y_min, y_max, z_min, z_max)
vol_geom = astra.create_vol_geom(500, 400, 100, -120, 120, -150, 150, 0, 60)

# 4. Setup ASTRA Projector and Algorithm (Replacing ODL workflow)
# Create a 3D projector using CUDA
projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)

# Create a dummy volume data to project (replace with your phantom array)
vol_data = np.zeros((500, 400, 100), dtype=np.float32)
vol_id = astra.data3d.create('-vol', vol_geom, vol_data)

# Create a data object to hold the sinogram
sino_id = astra.data3d.create('-sino', proj_geom)

# Forward projection (Ray Transform equivalent)
astra.algorithm.create_and_run({
    'proj_type': 'cuda3d',
    'sino_id': sino_id,
    'vol_id': vol_id,
    'type': 'FP3D_CUDA'
})

# Get the generated sinogram data
proj_data = astra.data3d.get(sino_id)

# Clean up memory
astra.data3d.delete(vol_id)
astra.data3d.delete(sino_id)
astra.projector.delete(projector_id)

