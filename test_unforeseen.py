import os
import json
import random
import numpy as np
import torch
from tqdm import tqdm
import librosa
import soundfile as sf
import pyroomacoustics as pra

# Core
from config import *
from dsp.main import compute_gcc_phat, get_onset_centered_window, spherical_to_cartesian

# Architecture
from doa.model.main import SSLModel

def generate_unforeseen_dataset(source_dir, output_root, num_samples=100):
    audio_out = os.path.join(output_root, "audio")
    feat_out = os.path.join(output_root, "features")
    os.makedirs(audio_out, exist_ok=True)
    os.makedirs(feat_out, exist_ok=True)
    
    source_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith('.mp3')]
    metadata = []
    
    for idx in tqdm(range(num_samples), desc="Generating Unforeseen"):
        source_file = random.choice(source_files)
        y, sr = librosa.load(source_file, sr=SAMPLE_RATE)
        y_snippet = get_onset_centered_window(y, sr, WINDOW_SAMPLES)
        
        # New Room instance
        room_dim = np.array([random.uniform(3, 12), random.uniform(3, 10), random.uniform(2.5, 5)])
        room = pra.ShoeBox(room_dim, fs=SAMPLE_RATE, max_order=15, absorption=0.2)
        
        mic_center = room_dim / 2
        mic_pos = MIC_POSITIONS_LOCAL + mic_center[:, None]
        room.add_microphone_array(mic_pos)
        
        az, el = np.deg2rad(random.uniform(-180, 180)), np.deg2rad(random.uniform(-45, 45))
        source_pos = mic_center + spherical_to_cartesian(az, el) * 2.5
        
        room.add_source(np.clip(source_pos, 0.1, room_dim-0.1), signal=y_snippet)
        room.simulate()
        
        mics_signals = room.mic_array.signals[:, :WINDOW_SAMPLES]
        
        # Save
        prefix = f"unforeseen_{idx:03d}"
        sf.write(os.path.join(audio_out, f"{prefix}.wav"), mics_signals.T, SAMPLE_RATE)
        
        features = []
        for m in range(1, 4):
            cc = compute_gcc_phat(mics_signals[m], mics_signals[0], SAMPLE_RATE, MAX_LAG)
            features.append(cc)
        np.save(os.path.join(feat_out, f"{prefix}.npy"), np.array(features))
        
        metadata.append({
            "id": prefix,
            "azimuth_rad": az,
            "elevation_rad": el,
            "cartesian_unit": spherical_to_cartesian(az, el).tolist()
        })
        
    with open(os.path.join(output_root, "labels.json"), "w") as f:
        json.dump(metadata, f, indent=4)

def evaluate(data_root, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSLModel()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    
    with open(os.path.join(data_root, "labels.json"), "r") as f:
        samples = json.load(f)
        
    errors = []
    for s in tqdm(samples, desc="Evaluating"):
        feat = np.load(os.path.join(data_root, "features", f"{s['id']}.npy")).astype(np.float32)
        with torch.no_grad():
            pred = model(torch.from_numpy(feat).unsqueeze(0).to(device)).cpu().numpy()[0]
        
        true_vec = np.array(s['cartesian_unit'])
        ang_err = np.rad2deg(np.arccos(np.clip(np.dot(pred, true_vec), -1.0, 1.0)))
        errors.append(ang_err)
        
    print(f"\nUnforeseen Mean Error: {np.mean(errors):.4f}°")

if __name__ == "__main__":
    out_root = os.path.join(DOA_DATA_DIR, "dataset", "unforeseen")
    generate_unforeseen_dataset(SAMPLES_DIR, out_root, num_samples=50)
    evaluate(out_root, DOA_MODEL_WEIGHTS)
