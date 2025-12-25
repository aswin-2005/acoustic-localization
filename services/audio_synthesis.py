import numpy as np
import os

from models.audio import (
    generate_audio,
    write_file,
    save_metadata,
)
from utils.transforms import angle_to_unit_vector

def synthesize_audio(config):
    OUTPUT_DIR = config.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fs = config.sample_rate
    duration = config.duration
    output_sound_file = f"{OUTPUT_DIR}/{config.audio_file}"
    output_metadata_file = f"{OUTPUT_DIR}/{config.metadata_file}"

    azimuth = config.azimuth              # degrees
    elevation = config.elevation            # degrees
    mic_positions = config.mic_positions  # numpy array of shape (num_mics, 3)

    source_signal = np.random.randn(int(fs * duration))
    source_signal /= np.max(np.abs(source_signal))

    direction = angle_to_unit_vector(azimuth, elevation)

    mic_signals, delays = generate_audio(
        source_signal=source_signal,
        mic_positions=mic_positions,
        direction=direction,
        fs=fs,
    )

    write_file(
        filename=output_sound_file,
        signals=mic_signals,
        fs=fs,
    )

    save_metadata(
        fs=fs,
        mic_positions=mic_positions,
        direction=direction,
        delays=delays,
        filename=output_metadata_file
    )

    print(f"\nAudio written to: {output_sound_file}")
    print(f"Metadata written to: {output_metadata_file}")

