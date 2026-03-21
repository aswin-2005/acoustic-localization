import torch
import numpy as np
import torch.nn.functional as F
from doa.model.main import SSLModel

class DOAPredictor:
    """
    A lightweight API for the Direction of Arrival (DOA) pipeline.
    This class handles model loading and predicts direction from 
    pre-computed GCC-PHAT correlation vectors.
    """
    def __init__(self, weights_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Initialize and load model
        self.model = SSLModel()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device).eval()
        
    def predict(self, correlation_vectors):
        """
        Input: 
            correlation_vectors: np.ndarray of shape (3, 33) 
                                 representing GCC-PHAT for 3 mic pairs.
        Output:
            az_deg: float, Azimuth in degrees [-180, 180]
            el_deg: float, Elevation in degrees [-90, 90]
        """
        # Ensure correct shape and type
        if correlation_vectors.shape != (3, 33):
            raise ValueError(f"Expected shape (3, 33), got {correlation_vectors.shape}")
            
        feat_tensor = torch.from_numpy(correlation_vectors.astype(np.float32)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Model returns unit vector [x, y, z]
            pred_vec = self.model(feat_tensor).cpu().numpy()[0]
            
        x, y, z = pred_vec
        
        # Convert Cartesian to Spherical
        # r is implicitly 1.0 due to L2-normalization in the model
        az_rad = np.arctan2(y, x)
        el_rad = np.arcsin(np.clip(z, -1.0, 1.0))
        
        az_deg = np.rad2deg(az_rad)
        el_deg = np.rad2deg(el_rad)
        
        return az_deg, el_deg

# Usage Example:
# predictor = DOAPredictor("weights/ssl_model.pth")
# az, el = predictor.predict(my_gcc_phat_vectors)
