import numpy as np
import soundfile as sf
import json

def generate_multichannel_audio(
    source_signal,
    mic_positions,
    direction,
    fs,
    sound_speed=343.0
):
    delays = mic_positions @ direction / sound_speed
    delays -= delays.min()

    n = len(source_signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    SRC = np.fft.rfft(source_signal)

    signals = np.zeros((mic_positions.shape[0], n))

    for i, d in enumerate(delays):
        SIG = SRC * np.exp(-2j * np.pi * freqs * d)
        signals[i] = np.fft.irfft(SIG, n=n)

    return signals, delays


def write_multichannel_wav(
    filename,
    signals,
    fs
):
    sf.write(filename, signals.T, fs)


def print_metadata(
    fs,
    mic_positions,
    direction,
    delays,
    filename="meta.json"
):
    duration_sec = delays.shape[0] / fs

    metadata = {
        "sampling_rate_hz": fs,
        "duration_sec": round(duration_sec, 6),

        "geometry": {
            "mic_positions_m": mic_positions.tolist(),
            "direction_unit_vector": direction.tolist(),
        },

        "timing": {
            "delays_sec": delays.tolist(),
            "tdoa_usec_ref0": [
                round((delays[i] - delays[0]) * 1e6, 6)
                for i in range(len(delays))
            ],
        },
    }

    with open(filename, "w") as f:
        json.dump(metadata, f, indent=2)

