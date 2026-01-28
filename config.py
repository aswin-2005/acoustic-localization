# config.py
import numpy as np

# =========================
# Sampling
# =========================
FS = 32000
FRAME_SEC = 1.0
FRAME_SAMPLES = int(FS * FRAME_SEC)

# =========================
# Room
# =========================
ROOM_DIMS = np.array([6.0, 5.0, 3.0])
ABSORPTION = 0.7
MAX_ORDER = 0

# =========================
# Microphone Array
# =========================
MIC_CENTER = np.array([3.0, 2.5, 1.5])
d = 0.1 / np.sqrt(8)

MIC_POSITIONS = np.array([
    MIC_CENTER + [ d,  d,  d],
    MIC_CENTER + [ d, -d, -d],
    MIC_CENTER + [-d,  d, -d],
    MIC_CENTER + [-d, -d,  d],
])

# =========================
# Audio Sources
# =========================
EVENT_DIR = "audio/events"
NOISE_DIR = "audio/noise"
EVENT_PROBABILITY = 0.1

# =========================
# SED
# ======
# ===================
FS_PANNS = 32000

# Global fallback
SED_THRESHOLD = 0.05

# Per-class thresholds (IMPORTANT)
SED_THRESHOLDS = {
    "Glass": 0.01,
    "Breaking": 0.01,
    "Gunshot, gunfire": 0.08,
    "Explosion": 0.1,
    "Thump": 0.05,
    "Impact": 0.05,
    "Slap, smack": 0.05,
    "Whack, thwack": 0.05,
}
