import soundfile as sf
import os

def load_audio(filename):
    if not os.path.exists(filename):
        raise ValueError("Audio file does not exist:", filename)
    signal, sample_rate = sf.read(filename)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    duration = len(signal) / sample_rate
    return signal, sample_rate, duration