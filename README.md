# Acoustic Localization

A Python project for acoustic source localization using time-difference-of-arrival (TDOA) estimation.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/aswin-2005/acoustic-localization.git
   cd acoustic-localization
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Generate sample data:
   ```bash
   python synthesize.py
   ```

2. Analyze the audio to estimate source direction and compare with true data:
   ```bash
   python analyze.py
   ```

## Project Structure

- `analyze.py`: Main analysis script for direction estimation
- `synthesize.py`: Generates synthetic multichannel audio data
- `models/`: Core algorithms (DSP, audio processing)
- `utils/`: Utility functions (transforms)
- `data/`: Generated audio files and metadata