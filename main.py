from utils.data_config import (
    load_synthesis_config,
    load_doa_config,
)
from services.audio_synthesis import synthesize_audio
from services.doa_analysis import analyze_doa


def main():
    synth_cfg = load_synthesis_config()
    doa_cfg = load_doa_config()

    synthesize_audio(synth_cfg)
    analyze_doa(doa_cfg)


if __name__ == "__main__":
    main()
