import os
import sys
from config import SED_MODEL_WEIGHTS, SED_LABELS_CSV
from sed.api import SEDPredictor

def test_sed_api():
    # Paths from config
    weights = SED_MODEL_WEIGHTS
    labels_csv = SED_LABELS_CSV
    
    # Check if files exist
    if not os.path.exists(weights):
        print(f"Error: Weights not found at {weights}")
        return
    if not os.path.exists(labels_csv):
        print(f"Error: Labels not found at {labels_csv}")
        return
        
    # Pick a sample audio file from the DOA data 
    # (Glass breaking should be easily recognized by PANNs)
    sample_audio = r"f:\codes\PROJECTS\mini-project\v7\samples\11325622-glass-breaking-sound-effect-240679.mp3"
    
    if not os.path.exists(sample_audio):
        print(f"Error: Sample audio not found at {sample_audio}")
        return

    print("Initializing SED Predictor...")
    predictor = SEDPredictor(weights, labels_csv)
    
    print(f"Analyzing: {os.path.basename(sample_audio)}")
    top_classes = predictor.predict(sample_audio, top_k=5)
    
    print("\n--- SED Inference Results ---")
    for i, (label, prob) in enumerate(top_classes):
        print(f"{i+1}. {label}: {prob:.4f}")

if __name__ == "__main__":
    test_sed_api()
