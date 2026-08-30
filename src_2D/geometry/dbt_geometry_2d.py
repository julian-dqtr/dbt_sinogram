import astra
import numpy as np


class DBTGeometry:
    """Stationary-detector 2D fan-beam geometry for a single DBT slice acquisition.
    
    This models the same rotation plane as the original 3D cone-beam geometry (the axis
    perpendicular to this plane was stationary/decoupled anyway, so a single 2D slice
    reconstruction is physically equivalent to one row of the 3D volume).
    """

    def __init__(self, angles, src_radius, det_radius, det_col_count, det_pixel_size):
        # Store configuration parameters
        self.angles = angles
        self.num_views = len(angles)
        self.src_radius = src_radius
        self.det_radius = det_radius
        self.det_col_count = det_col_count
        self.det_pixel_size = det_pixel_size
        
        # Generate and store the geometry vectors upon initialization
        self.vectors = self._generate_vectors()
        
    def _generate_vectors(self):
        # Initialize a custom vector geometry array for ASTRA (num_views, 6)
        # Row format: [src_x, src_y, det_x, det_y, u_x, u_y]
        vectors = np.zeros((self.num_views, 6))
        
        for i, angle in enumerate(self.angles):
            # Calculate moving source position (rotating around the isocenter)
            vectors[i, 0] = self.src_radius * np.sin(angle)
            vectors[i, 1] = self.src_radius * np.cos(angle)
            
            # Keep the detector position perfectly stationary below the isocenter
            vectors[i, 2:4] = [0.0, -self.det_radius]
            
            # Set the detector orientation vector (U)
            # Must be scaled by the pixel size for accurate physical dimensions in ASTRA
            vectors[i, 4:6] = [self.det_pixel_size, 0.0]  # U vector (Detector X-axis)
            
        return vectors
        
    def get_astra_proj_geom(self):
        # Return the initialized ASTRA projection geometry object
        return astra.create_proj_geom('fanflat_vec', self.det_col_count, self.vectors)