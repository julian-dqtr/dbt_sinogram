import numpy as np
import odl
import astra
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CUSTOM ASTRA GEOMETRY GENERATORS
# ==============================================================================
def get_rotating_vectors(angles, src_radius, det_radius, px_size):
    """ Simulated standard CT scanner (Source and Detector rotate together) """
    vecs = np.zeros((len(angles), 12))
    for i, a in enumerate(angles):
        # Source position
        vecs[i, 0:3] = [src_radius * np.sin(a), 0.0, src_radius * np.cos(a)]
        # Detector center (opposite to source)
        vecs[i, 3:6] = [-det_radius * np.sin(a), 0.0, -det_radius * np.cos(a)]
        # Detector U vector (Width, rotates)
        vecs[i, 6:9] = [px_size * np.cos(a), 0.0, -px_size * np.sin(a)]
        # Detector V vector (Height, stays on Y axis)
        vecs[i, 9:12] = [0.0, px_size, 0.0]
    return vecs

def get_stationary_vectors(angles, src_radius, det_radius, px_size):
    """ Simulated DBT (Source rotates, Detector stays completely flat at the bottom) """
    vecs = np.zeros((len(angles), 12))
    for i, a in enumerate(angles):
        # Source position (moves)
        vecs[i, 0:3] = [src_radius * np.sin(a), 0.0, src_radius * np.cos(a)]
        # Detector center (FIXED below isocenter)
        vecs[i, 3:6] = [0.0, 0.0, -det_radius]
        # Detector U and V vectors (FIXED and aligned with X and Y axes)
        vecs[i, 6:9] = [px_size, 0.0, 0.0]
        vecs[i, 9:12] = [0.0, px_size, 0.0]
    return vecs

# ==============================================================================
# 2. DEFINITIONS & PHANTOM
# ==============================================================================
# The reco space (Physical volume in mm: 240 x 300 x 60)
reco_space = odl.uniform_discr(
    min_pt=[-120, -150, 0], max_pt=[120, 150, 60], shape=[128, 160, 32],
    dtype='float32')

# Create Phantom in ODL, then extract to ASTRA memory layout (Z, Y, X)
phantom_odl = odl.core.phantom.shepp_logan(reco_space, True)
phantom_array = phantom_odl.asarray() # ODL shape: (X=128, Y=160, Z=32)
phantom_astra = np.transpose(phantom_array, (2, 1, 0)).copy() # ASTRA shape: (32, 160, 128)

# ASTRA Volume Geometry definition: (Y_rows, X_cols, Z_slices, x_min, x_max, y_min, y_max, z_min, z_max)
vol_geom = astra.create_vol_geom(160, 128, 32, -120, 120, -150, 150, 0, 60)

# ==============================================================================
# 3. SETTINGS & PROJECTIONS
# ==============================================================================
det_col_count = 128
det_row_count = 160
# INCREASED pixel size to 2.5 mm. 
# Detector is now 320x400 mm, big enough to catch the magnified shadow of the phantom! No more cut edges.
det_pixel_size = 2.5 

angles_full = np.linspace(-90 * np.pi / 180, 90 * np.pi / 180, 180)
angles_dbt = np.linspace(-25 * np.pi / 180, 25 * np.pi / 180, 25)

vecs_full = get_rotating_vectors(angles_full, 590.0, 60.0, det_pixel_size)
vecs_dbt = get_stationary_vectors(angles_dbt, 590.0, 60.0, det_pixel_size)

geom_full = astra.create_proj_geom('cone_vec', det_row_count, det_col_count, vecs_full)
geom_dbt = astra.create_proj_geom('cone_vec', det_row_count, det_col_count, vecs_dbt)

# --- Helper function to project using ASTRA ---
def project_volume(geom):
    proj_id = astra.create_projector('cuda3d', geom, vol_geom)
    vol_id = astra.data3d.create('-vol', vol_geom, phantom_astra)
    sino_id = astra.data3d.create('-sino', geom)
    
    cfg = astra.astra_dict('FP3D_CUDA')
    cfg['ProjectorId'] = proj_id
    cfg['VolumeDataId'] = vol_id
    cfg['ProjectionDataId'] = sino_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    
    sino_arr = astra.data3d.get(sino_id)
    
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(sino_id)
    astra.data3d.delete(vol_id)
    astra.projector.delete(proj_id)
    return sino_arr

sino_full_arr = project_volume(geom_full)
sino_dbt_arr = project_volume(geom_dbt)

# ==============================================================================
# 4. ASTRA ITERATIVE RECONSTRUCTION (SIRT) FOR DBT
# ==============================================================================
print("Running SIRT reconstruction (DBT)...")
proj_id = astra.create_projector('cuda3d', geom_dbt, vol_geom)
sino_id = astra.data3d.create('-sino', geom_dbt, sino_dbt_arr)
reco_id = astra.data3d.create('-vol', vol_geom)

cfg = astra.astra_dict('SIRT3D_CUDA')
cfg['ProjectorId'] = proj_id
cfg['ProjectionDataId'] = sino_id
cfg['ReconstructionDataId'] = reco_id
sirt_id = astra.algorithm.create(cfg)

# 20 iterations are enough for a small volume to converge nicely
astra.algorithm.run(sirt_id, 20) 
reco_astra = astra.data3d.get(reco_id)

# Transpose reconstruction back to ODL layout (X, Y, Z) for normal visualization
reco_arr = np.transpose(reco_astra, (2, 1, 0))

# Cleanup
astra.algorithm.delete(sirt_id)
astra.data3d.delete(reco_id)
astra.data3d.delete(sino_id)
astra.projector.delete(proj_id)

# ==============================================================================
# 5. VISUALIZATION
# ==============================================================================
mid_z = 16
mid_y = int(160 / 2)
physical_det_width = 128 * det_pixel_size / 2.0

plt.figure(figsize=(16, 10))

# 1. Original Phantom
plt.subplot(2, 2, 1)
plt.imshow(phantom_array[:, :, mid_z].T, cmap='gray', origin='lower', extent=[-120, 120, -150, 150])
plt.title(f'1. Original Phantom (Z={mid_z})')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')

# 2. Full Sinogram (Rotating CT)
plt.subplot(2, 2, 2)
plt.imshow(sino_full_arr[mid_y, :, :].T, cmap='bone', origin='lower', aspect='auto', extent=[-90, 90, -physical_det_width, physical_det_width])
plt.title('2. Full Sinogram (Rotating CT, -90° to +90°)')
plt.xlabel(r'Angle $\phi$ (Degrees)')
plt.ylabel('Detector Width u (mm)')

# 3. Limited Sinogram (Stationary DBT)
ax_lim = plt.subplot(2, 2, 3)
plt.imshow(sino_dbt_arr[mid_y, :, :].T, cmap='bone', origin='lower', aspect='auto', extent=[-25, 25, -physical_det_width, physical_det_width])
plt.title(r'3. DBT Sinogram (Stationary Detector, $\pm 25^\circ$)')
plt.xlabel(r'Angle $\phi$ (Degrees)')
plt.ylabel('Detector Width u (mm)')
ax_lim.set_xlim(-90, 90) # Force visual scale to match image 2 for direct comparison
ax_lim.set_facecolor('black')

# 4. SIRT Reconstruction
plt.subplot(2, 2, 4)
plt.imshow(reco_arr[:, :, mid_z].T, cmap='gray', origin='lower', vmin=0, vmax=np.max(phantom_array), extent=[-120, 120, -150, 150])
plt.title('4. SIRT Reconstruction (DBT)')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')

plt.tight_layout()
plt.show()