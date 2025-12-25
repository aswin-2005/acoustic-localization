from dataclasses import dataclass
from pathlib import Path
import yaml
import numpy as np

@dataclass
class SynthesisConfig:
    sample_rate: int
    duration: float
    azimuth: float
    elevation: float
    mic_positions: np.ndarray
    output_dir: str
    audio_file: str
    metadata_file: str


@dataclass
class DOAConfig:
    audio_file: str
    metadata_file: str
    interp: int = 8
    print_tdoa: bool = False

def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_synthesis_config(
    config_dir: str = "config",
    filename: str = "synthesis_config.yaml",
) -> SynthesisConfig:
    cfg_path = Path(config_dir) / filename
    y = _load_yaml(cfg_path)

    return SynthesisConfig(
        sample_rate=y["audio"]["sample_rate"],
        duration=y["audio"]["duration"],
        azimuth=y["geometry"]["azimuth"],
        elevation=y["geometry"]["elevation"],
        mic_positions=np.array(y["geometry"]["mic_positions"], dtype=float),
        output_dir=y["output"]["directory"],
        audio_file=y["output"]["audio_file"],
        metadata_file=y["output"]["metadata_file"],
    )


def load_doa_config(
    config_dir: str = "config",
    filename: str = "doa_config.yaml",
) -> DOAConfig:
    cfg_path = Path(config_dir) / filename
    y = _load_yaml(cfg_path)

    return DOAConfig(
        audio_file=y["input"]["audio_file"],
        metadata_file=y["input"]["metadata_file"],
        interp=y.get("gcc_phat", {}).get("interp", 8),
        print_tdoa=y.get("output", {}).get("print_tdoa", False),
    )
