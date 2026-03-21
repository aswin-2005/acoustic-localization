"""
Core Digital Signal Processing (DSP) Module
Contains logic for correlation (GCC-PHAT), coordinate mappings, and audio windowing.
"""

import numpy as np
import librosa
from scipy.fft import rfft, irfft

def compute_gcc_phat(sig, ref_sig, fs, max_lag):
    """
    Generalized Cross Correlation with Phase Transform.
    Core algorithm for TDOA extraction.
    """
    n = len(sig) + len(ref_sig)
    X = rfft(sig, n=n)
    Y = rfft(ref_sig, n=n)
    R = X * np.conj(Y)
    R /= (np.abs(R) + 1e-12) # Whitening
    cc = irfft(R, n=n)
    # Extract lag window
    cc = np.concatenate((cc[-(max_lag):], cc[:max_lag+1]))
    return cc

def cartesian_to_spherical(x, y, z):
    """Convert Cartesian coordinates to (Azimuth, Elevation) in radians."""
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)
    el = np.arcsin(np.clip(z / r, -1.0, 1.0))
    return az, el

def spherical_to_cartesian(az, el):
    """Convert (Azimuth, Elevation) in radians to a Cartesian unit vector [x, y, z]."""
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    return np.array([x, y, z])

def get_onset_centered_window(y, sr, window_size):
    """
    Extracts a snippet of audio centered on a detected onset.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    
    if len(onset_frames) > 0:
        onset_sample = librosa.frames_to_samples(onset_frames[0])
    else:
        onset_sample = np.argmax(np.abs(y))
        
    margin = int(0.1 * sr)
    onset_pos = np.random.randint(margin, window_size - margin)
    start_sample = onset_sample - onset_pos
    
    if start_sample < 0:
        snippet = np.pad(y[:max(0, start_sample + window_size)], (abs(start_sample), 0))
    elif start_sample + window_size > len(y):
        snippet = np.pad(y[start_sample:], (0, max(0, start_sample + window_size - len(y))))
    else:
        snippet = y[start_sample : start_sample + window_size]
        
    return snippet
