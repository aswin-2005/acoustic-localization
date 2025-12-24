import numpy as np
import os

from models.audio import (
    generate_multichannel_audio,
    write_multichannel_wav,
    print_metadata,
)
from utils.transforms import angle_to_unit_vector

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # =====================
    # Configuration
    # =====================
    fs = 48_000                 # Sampling rate (Hz)
    duration = 1.0              # seconds
    azimuth = 40.0              # degrees
    elevation = 20.0            # degrees
    output_sound_file = f"{OUTPUT_DIR}/sample.wav"
    output_metadata_file = f"{OUTPUT_DIR}/meta.json"

    # =====================
    # Microphone geometry
    # =====================
    mic_positions = np.array([
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0],
        [0.0, 0.05, 0.0],
        [0.0, 0.0, 0.05],
    ])

    # =====================
    # Generate source signal
    # =====================
    source_signal = np.random.randn(int(fs * duration))
    source_signal /= np.max(np.abs(source_signal))


    # =====================
    # Direction vector
    # =====================
    direction = angle_to_unit_vector(azimuth, elevation)

    # =====================
    # Synthesize multichannel audio
    # =====================
    mic_signals, delays = generate_multichannel_audio(
        source_signal=source_signal,
        mic_positions=mic_positions,
        direction=direction,
        fs=fs,
    )

    # =====================
    # Write WAV file
    # =====================
    write_multichannel_wav(
        filename=output_sound_file,
        signals=mic_signals,
        fs=fs,
    )

    # =====================
    # Print metadata
    # =====================
    print_metadata(
        fs=fs,
        mic_positions=mic_positions,
        direction=direction,
        delays=delays,
        filename=output_metadata_file
    )

    print(f"\nAudio written to: {output_sound_file}")
    print(f"Metadata written to: {output_metadata_file}")


if __name__ == "__main__":
    main()
