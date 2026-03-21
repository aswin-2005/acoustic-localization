"""
Global System Configuration
Source of truth for hardware, physical constants, and production paths.
"""

import os
import numpy as np

# --- Physical Constants ---
SAMPLE_RATE     = 16000           # Hz (16kHz for DOA, SED resamples internally)
SPEED_OF_SOUND  = 343.0           # m/s

# --- Microphone Array Geometry (Tetrahedron) ---
ARRAY_RADIUS = 0.05   # metres

# Canonical tetrahedral unit vectors (each row is one mic)
_TETRA_UNIT = np.array([
    [ 1,  1, -1],
    [-1, -1, -1],
    [ 1, -1,  1],
    [-1,  1,  1],
], dtype=np.float64)
_TETRA_UNIT /= np.linalg.norm(_TETRA_UNIT[0])          # normalise to unit sphere

# Export Mic Positions (centered)
MIC_POSITIONS_LOCAL = (_TETRA_UNIT * ARRAY_RADIUS).T    # shape [3, 4]

# --- DSP Constraints ---
MAX_MIC_SEP = np.max(
    np.linalg.norm(
        MIC_POSITIONS_LOCAL[:, :, None] - MIC_POSITIONS_LOCAL[:, None, :], axis=0
    )
)
# GCC-PHAT Lag Window (samples)
MAX_LAG = int(np.ceil(MAX_MIC_SEP / SPEED_OF_SOUND * SAMPLE_RATE)) * 3 + 4
GCC_PHAT_LEN = 2 * MAX_LAG + 1

# --- Windowing ---
WINDOW_DURATION = 1.0
WINDOW_SAMPLES  = int(SAMPLE_RATE * WINDOW_DURATION)

# --- Stream / Event Generation ---
EVENT_PROBABILITY = 0.8  # Probability (0.0–1.0) that a stream block contains a sound event

# --- Room Geometry (shared by generator and dashboard) ---
ROOM_DIM        = np.array([8.0, 6.0, 3.5])   # metres  [W, L, H]
MIC_ARRAY_CENTER = np.array([ROOM_DIM[0]/2, ROOM_DIM[1]/2, 1.5])

# --- Global Paths ---
ROOT_DIR     = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR  = os.path.join(ROOT_DIR, "samples")
DOA_DATA_DIR = os.path.join(ROOT_DIR, "doa", "data")
# Weights
DOA_MODEL_WEIGHTS = os.path.join(ROOT_DIR, "doa", "model", "weights", "ssl_model.pth")
SED_MODEL_WEIGHTS = os.path.join(ROOT_DIR, "sed", "model", "weights", "custom_Cnn14.pth")
SED_LABELS_CSV    = os.path.join(ROOT_DIR, "sed", "model", "labels", "custom_class_labels_indices.csv")

# Trigger Event Classes and their specific probability thresholds
LOCALIZATION_TRIGGERS = {
    "Glassbreak": 0.70,
    "Gunshot": 0.70,
    "explosion": 0.40,
    "whack": 0.70,
    "siren": 0.70,
}

# --- Dashboard / Camera Visualization ---
CAMERA_INIT_AZ    = 0.0    # Starting azimuth (degrees)
CAMERA_INIT_EL    = 0.0    # Starting elevation (degrees)
CAMERA_TURN_SPEED = 30.0   # Fixed rotation speed in degrees/second
