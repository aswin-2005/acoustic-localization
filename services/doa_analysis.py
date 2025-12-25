import json
import numpy as np
import soundfile as sf

from models.dsp import (
    estimate_tdoa,
    estimate_direction_vector,
)
from utils.transforms import unit_vector_to_angles
from utils.evaluations import angular_error


def analyze_doa(config):
    wav_file = config.audio_file
    meta_file = config.metadata_file

    with open(meta_file, "r") as f:
        meta = json.load(f)

    fs = meta["sampling_rate_hz"]
    mic_positions = np.array(meta["geometry"]["mic_positions_m"])
    true_direction = np.array(meta["geometry"]["direction_unit_vector"])

    signals, fs_read = sf.read(wav_file)
    if fs_read != fs:
        raise ValueError(f"Sampling rate mismatch: {fs_read} != {fs}")

    mic_signals = signals.T

    tdoa_matrix = estimate_tdoa(
        mic_signals,
        fs=fs,
        interp=config.interp,
    )

    est_direction = estimate_direction_vector(
        mic_positions,
        tdoa_matrix
    )

    true_az, true_el = unit_vector_to_angles(true_direction)
    est_az, est_el = unit_vector_to_angles(est_direction)

    ang_err = angular_error(true_direction, est_direction)


    print("\nTrue Direction Vector:")
    print(true_direction)

    print("\nEstimated Direction Vector:")
    print(est_direction)

    print("\nTrue Direction Angles (degrees):")
    print(f"  Azimuth      : {true_az:.2f}")
    print(f"  Elevation    : {true_el:.2f}")
    print("\nEstimated Direction Angles (degrees):")
    print(f"  Azimuth : {est_az:.2f}")
    print(f"  Elevation   : {est_el:.2f}")

    print("\nAccuracy:")
    print(f"  Angular Error     : {ang_err:.4f} degrees")

    if config.print_tdoa:
        print("\nTDOA Matrix (microseconds):")
        print(np.round(tdoa_matrix * 1e6, 3))