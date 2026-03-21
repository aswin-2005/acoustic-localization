import os
import json
import random
import numpy as np
import librosa
import pyroomacoustics as pra
import soundfile as sf
from tqdm import tqdm

# Root Core Utility
from config import *
from dsp.main import compute_gcc_phat, get_onset_centered_window, spherical_to_cartesian

# Local Training Configuration
from doa.training_pipeline.train_config import *

def generate_sample(sample_idx, source_audio_files, is_train=True):
    """
    Generates a single SSL training sample (4ch audio + 3 GCC-PHAT features).
    """
    source_file = random.choice(source_audio_files)
    y, sr = librosa.load(source_file, sr=SAMPLE_RATE)
    
    # Gain
    y *= random.uniform(*GAIN_RANGE)
    # Pitch
    n_steps = random.uniform(*PITCH_SHIFT_STEPS)
    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    # Stretch
    rate = random.uniform(*TIME_STRETCH_RATES)
    y = librosa.effects.time_stretch(y, rate=rate)
    
    y_snippet = get_onset_centered_window(y, sr, WINDOW_SAMPLES)
    
    # Room
    room_dim = np.array([random.uniform(ROOM_DIM_MIN[i], ROOM_DIM_MAX[i]) for i in range(3)])
    t60 = random.uniform(T60_MIN, T60_MAX)
    
    vol = np.prod(room_dim)
    surf = 2 * (room_dim[0]*room_dim[1] + room_dim[1]*room_dim[2] + room_dim[2]*room_dim[0])
    avg_absorption = np.clip(1 - np.exp(-0.161 * vol / (surf * t60)), 0.05, 0.95)
    
    room = pra.ShoeBox(room_dim, fs=SAMPLE_RATE, max_order=ISM_MAX_ORDER, absorption=avg_absorption)
    
    # Mic
    mic_center = np.array([random.uniform(WALL_MARGIN, room_dim[i] - WALL_MARGIN) for i in range(3)])
    mic_pos = MIC_POSITIONS_LOCAL + mic_center[:, None]
    room.add_microphone_array(mic_pos)
    
    # Source
    az = np.deg2rad(random.uniform(-180, 180))
    el = np.deg2rad(random.uniform(-60, 60))
    dist = random.uniform(SOURCE_DIST_MIN, SOURCE_DIST_MAX)
    source_pos = mic_center + spherical_to_cartesian(az, el) * dist
    
    if not np.all((source_pos >= 0.1) & (source_pos <= room_dim - 0.1)):
        return None
        
    room.add_source(source_pos, signal=y_snippet)
    room.simulate()
    
    mics_signals = room.mic_array.signals[:, :WINDOW_SAMPLES]
    
    # Noise
    snr_db = random.uniform(SNR_MIN_DB, SNR_MAX_DB)
    sig_power = np.mean(mics_signals**2)
    noise = np.random.normal(0, np.sqrt(sig_power / (10**(snr_db / 10))), mics_signals.shape)
    mics_signals = mics_signals + noise
    
    # Features
    features = []
    for m in range(1, 4):
        cc = compute_gcc_phat(mics_signals[m], mics_signals[0], SAMPLE_RATE, MAX_LAG)
        features.append(cc)
    
    features = np.array(features)
    
    # Save
    split = "train" if is_train else "test"
    out_dir = os.path.join(DOA_DATA_DIR, "dataset", split)
    os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "features"), exist_ok=True)
    
    prefix = f"sample_{sample_idx:05d}"
    sf.write(os.path.join(out_dir, "audio", f"{prefix}.wav"), mics_signals.T, SAMPLE_RATE)
    np.save(os.path.join(out_dir, "features", f"{prefix}.npy"), features)
    
    return {
        "id": prefix,
        "azimuth_rad": az,
        "elevation_rad": el,
        "cartesian_unit": spherical_to_cartesian(az, el).tolist(),
        "snr_db": snr_db
    }

def main():
    source_files = [os.path.join(SAMPLES_DIR, f) for f in os.listdir(SAMPLES_DIR) if f.endswith('.mp3')]
    
    # Train
    train_meta = []
    pbar = tqdm(total=NUM_TRAIN_SAMPLES, desc="Generating Train Set")
    while len(train_meta) < NUM_TRAIN_SAMPLES:
        meta = generate_sample(len(train_meta), source_files, True)
        if meta:
            train_meta.append(meta)
            pbar.update(1)
    pbar.close()
    
    with open(os.path.join(DOA_DATA_DIR, "dataset", "train", "labels.json"), "w") as f:
        json.dump(train_meta, f, indent=4)
        
    # Test
    test_meta = []
    pbar = tqdm(total=NUM_TEST_SAMPLES, desc="Generating Test Set")
    while len(test_meta) < NUM_TEST_SAMPLES:
        meta = generate_sample(len(test_meta) + NUM_TRAIN_SAMPLES, source_files, False)
        if meta:
            test_meta.append(meta)
            pbar.update(1)
    pbar.close()
    
    with open(os.path.join(DOA_DATA_DIR, "dataset", "test", "labels.json"), "w") as f:
        json.dump(test_meta, f, indent=4)

if __name__ == "__main__":
    main()
