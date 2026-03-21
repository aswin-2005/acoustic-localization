import os
import json
import numpy as np
import torch
from tqdm import tqdm
import time
from datetime import datetime
import librosa

# Core Package
from config import *
from dsp.main import compute_gcc_phat, cartesian_to_spherical

# Model
from doa.model.main import SSLModel

def run_evaluation():
    test_dir = os.path.join(DOA_DATA_DIR, "test")
    labels_path = os.path.join(test_dir, "labels.json")
    weights_path = DOA_MODEL_WEIGHTS
    log_path = os.path.join(ROOT_DIR, "doa", "_debug", "evaluation_report.log")
    
    if not os.path.exists(labels_path) or not os.path.exists(weights_path):
        print(f"Required files missing. Check {labels_path} and {weights_path}")
        return

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSLModel()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    
    with open(labels_path, 'r') as f:
        test_samples = json.load(f)
        
    print(f"Running evaluation on {len(test_samples)} samples...")
    results = []
    start_time = time.time()
    
    for sample in tqdm(test_samples):
        sample_id = sample['id']
        wav_path = os.path.join(test_dir, "audio", f"{sample_id}.wav")
        
        if not os.path.exists(wav_path): continue
            
        y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=False)
        y = y[:, :WINDOW_SAMPLES]
        
        # Features
        features = []
        for m in range(1, 4):
            cc = compute_gcc_phat(y[m], y[0], SAMPLE_RATE, MAX_LAG)
            features.append(cc)
        
        feat_tensor = torch.from_numpy(np.array(features).astype(np.float32)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_vec = model(feat_tensor).cpu().numpy()[0]
            
        # Metrics
        true_az, true_el = sample['azimuth_rad'], sample['elevation_rad']
        true_vec = np.array([
            np.cos(true_el) * np.cos(true_az),
            np.cos(true_el) * np.sin(true_az),
            np.sin(true_el)
        ])
        
        cos_sim = np.clip(np.dot(pred_vec, true_vec), -1.0, 1.0)
        ang_err = np.rad2deg(np.arccos(cos_sim))
        
        results.append({
            "ang_err": ang_err,
            "snr": sample['snr_db']
        })
        
    total_time = time.time() - start_time
    
    # Report
    mean_ang = np.mean([r['ang_err'] for r in results])
    acc_10 = sum(1 for r in results if r['ang_err'] < 10) / len(results) * 100
    
    report = f"""
====================================================
          SSL MODEL EVALUATION REPORT
          Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
====================================================
Total Samples: {len(results)}
Mean Angular Error: {mean_ang:.4f}°
Accuracy (< 10°):  {acc_10:.2f}%
====================================================
"""
    print(report)
    with open(log_path, "w") as f:
        f.write(report)

if __name__ == "__main__":
    run_evaluation()
