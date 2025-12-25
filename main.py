from utils.configurations import (
    load_synthesis_config,
    load_doa_config,
)
from utils.audio_preprocessing import load_audio
from utils.audio_generation import(
    generate_sample_noise,
    generate_sample_pure_tone,
    generate_sample_tone_burst,
)
from utils.configurations import SynthesisConfig
from services.audio_synthesis import synthesize_audio
from services.doa_analysis import analyze_doa

def build_source_signal(cfg: SynthesisConfig):
    src = cfg.source

    if src.type == "noise":
        signal = generate_sample_noise(cfg.sample_rate, cfg.duration)

    elif src.type == "pure_tone":
        signal = generate_sample_pure_tone(
            cfg.sample_rate,
            cfg.duration,
            src.frequency,
        )

    elif src.type == "tone_burst":
        signal = generate_sample_tone_burst(
            cfg.sample_rate,
            cfg.duration,
            src.frequency,
            src.burst_duration,
        )

    elif src.type == "file":
        signal, sr, duration = load_audio(src.file_path)
        cfg.sample_rate = sr
        cfg.duration = duration

    else:
        raise ValueError(f"Unsupported source type: {src.type}")

    print(
        f"Source type: {src.type} | "
        f"sample rate: {cfg.sample_rate} Hz | "
        f"duration: {cfg.duration:.2f}s"
    )

    if src.type == "file":
        print(f"Loaded audio file: {src.file_path}")
    elif src.type in ["pure_tone", "tone_burst"]:
        print(f"Frequency: {src.frequency} Hz")
        if src.type == "tone_burst":
            print(f"Burst duration: {src.burst_duration:.2f} s")
    elif src.type == "noise":
        print("Generated white noise signal")
    

    return signal




def main():
    synth_cfg = load_synthesis_config()
    doa_cfg = load_doa_config()
    source_signal = build_source_signal(synth_cfg)
    synthesize_audio(synth_cfg, source_signal)
    analyze_doa(doa_cfg)


if __name__ == "__main__":
    main()
