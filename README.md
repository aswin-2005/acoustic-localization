# Acoustic Localization

A Python project for acoustic source localization using time-difference-of-arrival (TDOA) estimation.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python main.py
```

## Data Flow

1. Load synthesis and DOA configurations from `config/`
2. Synthesize multichannel audio data and save to `data/`
3. Analyze audio for direction-of-arrival (DOA) estimation using models and utilities
4. Output estimation results

## Project Structure

- `main.py`: Entry point for the application
- `config/`: Configuration files (DOA and synthesis settings)
- `data/`: Audio data and metadata
- `models/`: Core model classes (audio processing, DSP)
- `services/`: Service modules (audio synthesis, DOA analysis)
- `utils/`: Utility functions (analysis, data config, transforms)