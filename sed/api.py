import os
import torch
import numpy as np
import librosa
import pandas as pd
from sed.model.main import Cnn14

class SEDPredictor:
    def __init__(self, checkpoint_path, labels_csv, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Audio parameters from PANNs
        self.sample_rate = 32000
        self.window_size = 1024
        self.hop_size = 320
        self.mel_bins = 64
        self.fmin = 50
        self.fmax = 14000
        
        # Load labels
        df = pd.read_csv(labels_csv)
        self.labels = df['display_name'].values
        self.classes_num = len(self.labels)
        
        # Initialize model
        self.model = Cnn14(
            sample_rate=self.sample_rate,
            window_size=self.window_size,
            hop_size=self.hop_size,
            mel_bins=self.mel_bins,
            fmin=self.fmin,
            fmax=self.fmax,
            classes_num=self.classes_num
        )
        
        # Load weights
        print(f"Loading SED model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, audio_path, top_k=5):
        """
        Predict sound events from an audio file.
        """
        # Load and resample audio
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        audio = audio[None, :]  # Add batch dimension
        
        input_tensor = torch.from_numpy(audio).to(self.device).float()
        
        with torch.no_grad():
            output_dict = self.model(input_tensor)
            
        clipwise_output = output_dict['clipwise_output'].cpu().numpy()[0]
        
        # Get top-k results
        sorted_indexes = np.argsort(clipwise_output)[::-1]
        
        results = []
        for i in range(top_k):
            idx = sorted_indexes[i]
            results.append((self.labels[idx], clipwise_output[idx]))
            
        return results

    def predict_from_buffer(self, audio_buffer, top_k=5):
        """
        Predict from a numpy array buffer (mono, at self.sample_rate).
        """
        if len(audio_buffer.shape) == 1:
            audio_buffer = audio_buffer[None, :]
            
        input_tensor = torch.from_numpy(audio_buffer).to(self.device).float()
        
        with torch.no_grad():
            output_dict = self.model(input_tensor)
            
        clipwise_output = output_dict['clipwise_output'].cpu().numpy()[0]
        sorted_indexes = np.argsort(clipwise_output)[::-1]
        
        results = []
        for i in range(top_k):
            idx = sorted_indexes[i]
            results.append((self.labels[idx], clipwise_output[idx]))
            
        return results
