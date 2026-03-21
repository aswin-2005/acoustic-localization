"""
Configuration for Model Training and Retraining.
Contains hyperparameters and training-specific directory settings.
"""

from config import DOA_DATA_DIR

# --- Hyperparameters ---
LEARning_RATE = 0.001
BATCH_SIZE    = 32
NUM_EPOCHS    = 50

# --- Dataset Generation (For Retraining) ---
NUM_TRAIN_SAMPLES = 10000
NUM_TEST_SAMPLES  = 2000

# ISM Simulation complexity
ISM_MAX_ORDER = 17

# --- Augmentation Ranges ---
PITCH_SHIFT_STEPS = (-2, 2)
TIME_STRETCH_RATES = (0.8, 1.2)
GAIN_RANGE = (0.6, 1.4)

# --- Room Randomization (Training Environment) ---
ROOM_DIM_MIN = [3.0, 3.0, 2.5]
ROOM_DIM_MAX = [12.0, 10.0, 5.0]

T60_MIN = 0.10
T60_MAX = 0.80

SOURCE_DIST_MIN = 0.5
SOURCE_DIST_MAX = 4.0

WALL_MARGIN = 0.5

# --- Noise (Training) ---
SNR_MIN_DB = 10.0
SNR_MAX_DB = 35.0

