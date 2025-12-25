import numpy as np
import os

from models.audio import (
    generate_audio,
    write_file,
    save_metadata,
)
from utils.transforms import angle_to_unit_vector

def synthesize_audio(config, source_signal=None):
    os.makedirs(config.output_dir, exist_ok=True)
    fs = config.sample_rate
    output_sound_file = f"{config.output_dir}/{config.audio_file}"
    output_metadata_file = f"{config.output_dir}/{config.metadata_file}"

    azimuth = config.azimuth
    elevation = config.elevation
    mic_positions = config.mic_positions

    direction = angle_to_unit_vector(azimuth, elevation)

    if source_signal is None:
        raise ValueError("source_signal must be provided")


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

