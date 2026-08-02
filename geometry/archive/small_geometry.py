import numpy as np
import odl

# 1. Define the reconstruction space (volume of the compressed breast)
# Physical dimensions in mm are kept the same, but the shape (resolution) is downscaled.
# Original shape: [400, 500, 100] -> Mini shape: [128, 160, 32]
reco_space = odl.uniform_discr(
    min_pt=[-120, -150, 0], max_pt=[120, 150, 60], shape=[128, 160, 32],
    dtype='float32')

# Limited Angle partition: 25 views from -25 degrees to +25 degrees Around pi/2
angle_min = (90-25) * np.pi / 180
angle_max = (90+25) * np.pi / 180
num_views = 25
angle_partition = odl.uniform_partition(angle_min, angle_max, num_views)

# Full Angle partition:

full_angle_partition = odl.uniform_partition(0, np.pi, 180)

# 2. Detector partition: Reduced pixel count, kept physical size
# Physical dimensions are roughly 240mm x 300mm
det_width_half = 120.0
det_height_half = 150.0

# Original shape: [2816, 3584] -> Mini shape: [128, 160]
detector_partition = odl.uniform_partition(
    [-det_width_half, -det_height_half], 
    [det_width_half, det_height_half], 
    [128, 160]
)

# Geometry distances (in mm) - Kept identical to preserve the physics of the system
det_radius = 60.0 
src_radius = 590.0

# Initialize Cone Beam Geometry (Rotation axis Y)
geometry = odl.applications.tomo.ConeBeamGeometry(
    angle_partition, detector_partition, 
    src_radius=src_radius, det_radius=det_radius,
    axis=[0, 1, 0]
)

geometry_full = odl.applications.tomo.ConeBeamGeometry(
    full_angle_partition, detector_partition, 
    src_radius=src_radius, det_radius=det_radius,
    axis=[0, 1, 0]
)

# Ray transform (= forward projection)
ray_transfo = odl.applications.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
ray_transfo_full = odl.applications.tomo.RayTransform(reco_space, geometry_full, impl='astra_cuda')

# Create a discrete Shepp-Logan phantom
phantom = odl.core.phantom.shepp_logan(reco_space, True)

# Create projection data (clean sinograms)
proj_data = ray_transfo(phantom)           
proj_data_full = ray_transfo_full(phantom) 

# # 3. Add noise (Poisson + Gaussian) to make the limited views realistic
# proj_array = proj_data.asarray()
# incident_photons = 1e5  
# proj_intensity = np.exp(-proj_array) * incident_photons
# noisy_intensity = np.random.poisson(proj_intensity)
# noisy_intensity[noisy_intensity == 0] = 1 # Prevent log(0)
# noisy_proj_array = -np.log(noisy_intensity / incident_photons)

# gaussian_noise = np.random.normal(0.0, 0.05, proj_array.shape)
# final_noisy_proj = noisy_proj_array + gaussian_noise

# # Update ODL element with the newly calculated noisy data
# proj_data = ray_transfo.range.element(final_noisy_proj)

# # 4. Setup the Filtered Back-Projection (FDK) operator 
# # Added a 'Hann' filter to slightly smooth the image and suppress the high-frequency noise
fbp_operator = odl.applications.tomo.fbp_op(ray_transfo)

# Back-projection using ASTRA FDK on GPU
backproj_fdk = fbp_operator(proj_data)

# # 5. Display Settings
# # Define the middle index of the detector's Y-axis (Physical center is 0.0 mm)
# mid_det_y = 0.0  

# # Adjusted Z slice index: the volume has 32 slices now, so the middle is around 16
# slice_z = 30
# # Show outputs
# phantom.show(coords=[None, None, slice_z], title=f'1. Original Phantom (Z={slice_z})')

# # Now, when you look at this full sinogram, you will see the -25 to +25 range sits perfectly in the middle
# proj_data_full.show(coords=[None, None, mid_det_y], title='2. Full Sinogram (-90 to +90)')

# proj_data.show(coords=[None, None, mid_det_y], title='3. Noisy Limited Sinogram (-25 to +25)')
# backproj_fdk.show(coords=[None, None, slice_z], title='4. FDK Reconstruction (DBT)', force_show=True, clim=[0, 1])


# 8. Display Settings
# Define the middle index of the detector's Y-axis (Physical center is 0.0 mm)
mid_det_y = 0.0  

# Fixed Z slice index: the volume has 32 slices now, so the middle is around 16.
# Index 32 would throw an 'Out of bounds' error since indices are 0 to 31.
slice_z = 16

# Show outputs
phantom.show(coords=[None, None, slice_z], title=f'1. Original Phantom (Z={slice_z})')

# Show the full sinogram from 0 to pi
proj_data_full.show(coords=[None, None, mid_det_y], title='2. Full Sinogram (0 to $\pi$)')

# Show the limited sinogram, perfectly cropped around the pi/2 center
# We grab the figure object to manipulate the Matplotlib axes directly
fig_lim = proj_data.show(coords=[None, None, mid_det_y], title='3. Noisy Limited Sinogram (Centered)')
ax_lim = fig_lim.gca()

# Force the X-axis (angles) to match the full sinogram limits (0 to pi)
ax_lim.set_xlim(0, np.pi)

# Set the background to black to create the "cropped" effect you wanted
ax_lim.set_facecolor('black')

# Show the reconstruction
backproj_fdk.show(coords=[None, None, slice_z], title='4. FDK Reconstruction (DBT)', force_show=True, clim=[0, 1])