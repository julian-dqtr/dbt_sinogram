import astra
import numpy as np


class DBTGeometry:
    """Stationary-detector 3D cone-beam geometry for digital breast tomosynthesis (DBT) acquisition.

    The source rotates in the X-Y plane around the isocenter while the detector
    remains stationary below the isocenter at Y = -det_radius.
    """

    def __init__(self, angles, src_radius, det_radius, det_row_count, det_col_count, det_pixel_size):
        # Store configuration parameters
        self.angles = angles
        self.num_views = len(angles)
        self.src_radius = src_radius
        self.det_radius = det_radius
        self.det_row_count = det_row_count
        self.det_col_count = det_col_count
        self.det_pixel_size = det_pixel_size
        
        # Generate and store the geometry vectors upon initialization
        self.vectors = self._generate_vectors()
        
    def _generate_vectors(self):
        # Initialize a custom vector geometry array for ASTRA (num_views, 12)
        # Row format: [srcX, srcY, srcZ, detX, detY, detZ, uX, uY, uZ, vX, vY, vZ]
        vectors = np.zeros((self.num_views, 12))
        
        for i, angle in enumerate(self.angles):
            # Calculate moving source position (rotating around the isocenter)
            vectors[i, 0] = self.src_radius * np.sin(angle)
            vectors[i, 1] = self.src_radius * np.cos(angle)
            vectors[i, 2] = 0.0
            
            # Keep the detector position perfectly stationary below the isocenter
            vectors[i, 3:6] = [0.0, -self.det_radius, 0.0]
            
            # Set the detector orientation vectors (U and V)
            # U vector: along the X-axis (columns)
            vectors[i, 6:9] = [self.det_pixel_size, 0.0, 0.0]
            # V vector: along the Z-axis (rows)
            vectors[i, 9:12] = [0.0, 0.0, self.det_pixel_size]
            
        return vectors
        
    def get_astra_proj_geom(self):
        # Return the initialized ASTRA projection geometry object
        return astra.create_proj_geom('cone_vec', self.det_row_count, self.det_col_count, self.vectors)