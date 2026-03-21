import numpy as np
import os
from doa.api import DOAPredictor

def test_inference_api():
    weights = r"f:\codes\PROJECTS\mini-project\v7\doa\model\weights\ssl_model.pth"
    test_feature_path = r"f:\codes\PROJECTS\mini-project\v7\doa\data\test\features\sample_10000.npy"
    
    if not os.path.exists(weights):
        print("Model weights not found.")
        return
    if not os.path.exists(test_feature_path):
        print("Test feature file not found.")
        return
        
    # 1. Initialize the predictor
    predictor = DOAPredictor(weights)
    
    # 2. Load a pre-computed GCC-PHAT vector (shape: 3, 33)
    # This simulates what your actual DOA pipeline would provide
    my_gcc_phat = np.load(test_feature_path)
    
    # 3. Predict direction
    az, el = predictor.predict(my_gcc_phat)
    
    print("\n--- Test result for Inference API ---")
    print(f"Input Feature Shape: {my_gcc_phat.shape}")
    print(f"Predicted Azimuth:   {az:.2f}°")
    print(f"Predicted Elevation: {el:.2f}°")

if __name__ == "__main__":
    test_inference_api()
