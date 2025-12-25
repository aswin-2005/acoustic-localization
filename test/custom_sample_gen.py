import numpy as np
import soundfile as sf


def main():
    '''
    This function should be used to generate custom sample audio files for testing.
    The generated audio will be saved in the 'samples' directory with the specified filename.
    '''
    # Filename for the generated audio file
    filename = "300-3khz-chirp"
    # Sample rate and audio generation
    sample_rate = 44100

    # write your custom audio generation code here
    #----------------------------------------------------------------------------

    silence_duration = 0.25
    tone_duration = 0.5
    silence = np.zeros(int(sample_rate * silence_duration))
    t = np.linspace(0, tone_duration, int(sample_rate * tone_duration), False)
    chirp = np.sin(2 * np.pi * np.linspace(300, 3000, len(t)) * t / tone_duration)
    audio = np.concatenate([silence, chirp, silence])
    audio = audio / np.max(np.abs(audio))

    #---------------------------------------------------------------------------
    sf.write(f"samples/test_gen_{filename}.wav", audio, 44100)

if __name__ == "__main__":
    main()