# stream.py
import os
import random
import numpy as np
import librosa
import soundfile as sf
import pyroomacoustics as pra

# -------------------------
# Module-level state
# -------------------------
_cfg = None
_audio_files = []
_noise_files = []
_frame_counter = 0


def load_config(cfg):
    global _cfg, _audio_files, _noise_files, _frame_counter

    _cfg = cfg
    _frame_counter = 0

    # -------------------------
    # Load event audio
    # -------------------------
    _audio_files = []
    for fname in os.listdir(cfg.EVENT_DIR):
        if not fname.lower().endswith(".wav"):
            continue

        data, fs = sf.read(os.path.join(cfg.EVENT_DIR, fname))
        if data.ndim > 1:
            data = data.mean(axis=1)

        if fs != cfg.FS:
            data = librosa.resample(
                y=data, orig_sr=fs, target_sr=cfg.FS
            )

        _audio_files.append((fname, data.astype(np.float32)))

    if not _audio_files:
        raise RuntimeError("No event WAV files found")

    # -------------------------
    # Load noise audio
    # -------------------------
    _noise_files = []
    for fname in os.listdir(cfg.NOISE_DIR):
        if not fname.lower().endswith(".wav"):
            continue

        data, fs = sf.read(os.path.join(cfg.NOISE_DIR, fname))
        if data.ndim > 1:
            data = data.mean(axis=1)

        if fs != cfg.FS:
            data = librosa.resample(
                y=data, orig_sr=fs, target_sr=cfg.FS
            )

        _noise_files.append((fname, data.astype(np.float32)))

    if not _noise_files:
        raise RuntimeError("No noise WAV files found")


def _random_position(min_distance=2.0):
    mic_center = _cfg.MIC_POSITIONS.mean(axis=0)

    while True:
        pos = np.array([
            random.uniform(0.5, _cfg.ROOM_DIMS[0] - 0.5),
            random.uniform(0.5, _cfg.ROOM_DIMS[1] - 0.5),
            random.uniform(0.5, _cfg.ROOM_DIMS[2] - 0.5),
        ])

        if np.linalg.norm(pos - mic_center) >= min_distance:
            return pos


def simulate_one_second():
    global _frame_counter

    if _cfg is None:
        raise RuntimeError("load_config() must be called first")

    # -------------------------
    # Create a FRESH room (critical)
    # -------------------------
    room = pra.ShoeBox(
        _cfg.ROOM_DIMS,
        fs=_cfg.FS,
        absorption=_cfg.ABSORPTION,
        max_order=_cfg.MAX_ORDER,
    )

    mic_array = pra.MicrophoneArray(
        _cfg.MIC_POSITIONS.T, _cfg.FS
    )
    room.add_microphone_array(mic_array)

    # -------------------------
    # Select audio source
    # -------------------------
    if random.random() < _cfg.EVENT_PROBABILITY:
        name, audio = random.choice(_audio_files)
    else:
        name, audio = random.choice(_noise_files)

    # -------------------------
    # Prepare 1-second signal
    # -------------------------
    signal = np.zeros(_cfg.FRAME_SAMPLES, dtype=np.float32)
    audio = audio[:_cfg.FRAME_SAMPLES]
    signal[:len(audio)] = audio

    # -------------------------
    # Add source
    # -------------------------
    pos = _random_position()
    room.add_source(pos, signal)

    # -------------------------
    # Simulate
    # -------------------------
    room.simulate()
    signals = room.mic_array.signals.copy()

    _frame_counter += 1
    return signals, name, pos
