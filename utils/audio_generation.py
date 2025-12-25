
import numpy as np

def generate_sample_noise(fs, duration):
    source_signal = np.random.randn(int(fs * duration))
    source_signal /= np.max(np.abs(source_signal))
    return source_signal

def generate_sample_pure_tone(fs, duration, frequency=440):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    source_signal = 0.5 * np.sin(2 * np.pi * frequency * t)
    return source_signal

def generate_sample_tone_burst(fs, duration, frequency=440, burst_duration=0.1):
    if burst_duration >= duration:
        raise ValueError("burst_duration must be less than total duration")
    n_total = int(fs * duration)
    signal = np.zeros(n_total)
    silence_duration = duration - burst_duration
    start_time = silence_duration / 2
    burst_samples = int(fs * burst_duration)
    start_sample = int(fs * start_time)
    end_sample = start_sample + burst_samples
    t_burst = np.arange(burst_samples) / fs
    tone = np.sin(2 * np.pi * frequency * t_burst)
    window = np.hanning(burst_samples)
    tone *= window
    signal[start_sample:end_sample] = tone
    return signal